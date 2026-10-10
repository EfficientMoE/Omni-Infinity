#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Phase-1 per-stack compile-blocks parity check (upgrade matrix row).

One process, one GPU, one model load: an sm120 + opt-in fp8 smoke, an
eager ``resident``-profile generation, then
``compile_repeated_blocks(fullgraph=True, dynamic=True)`` exactly as the
runner applies it and a second generation. Reports three latent-level
parities: eager vs the recorded goldens (stack drift of the eager path),
compiled vs the goldens (the production golden gate), and compiled vs
the same-process eager latents (the stack's compile fidelity, which is
what a Phase-2 golden re-baseline could recover). The JSON row feeds the
Phase-1 matrix in
``docs/superpowers/plans/2026-10-09-p2-compile-blocks-parity-newer-stack.md``;
the abort gate reads ``gate_compile_vs_eager_2e2``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))

import profile_denoise_step as pds  # noqa: E402  (sibling script reuse)

COMPILE_KWARGS = {"fullgraph": True, "dynamic": True}


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def sm120_fp8_smoke() -> dict[str, Any]:
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "arch_list": torch.cuda.get_arch_list(),
    }
    try:
        lhs = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        rhs = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        product = lhs @ rhs
        attention = torch.nn.functional.scaled_dot_product_attention(
            torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16),
        )
        torch.cuda.synchronize()
        report["bf16_matmul_sdpa"] = bool(
            torch.isfinite(product).all() and torch.isfinite(attention).all()
        )
    except Exception as error:  # noqa: BLE001 - smoke must record, not raise
        report["bf16_matmul_sdpa"] = False
        report["bf16_error"] = repr(error)
    try:
        from omni_infinity import kernels

        weight = torch.randn(384, 512, device="cuda", dtype=torch.bfloat16)
        quantized, scale = kernels.quantize_block_fp8(weight)
        activation = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
        fused = kernels.fused_fp8_gemm(activation, quantized, scale)
        reference = kernels.fused_fp8_gemm_reference(
            activation, quantized, scale
        )
        torch.cuda.synchronize()
        rel = (
            (fused.float() - reference.float()).norm()
            / reference.float().norm()
        ).item()
        report["fp8_triton_vs_reference_rel"] = rel
        report["fp8_ok"] = bool(rel < 2e-2)
    except Exception as error:  # noqa: BLE001 - smoke must record, not raise
        report["fp8_ok"] = False
        report["fp8_error"] = repr(error)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=pds.DEFAULT_CHECKPOINT
    )
    parser.add_argument("--store-dir", type=Path, default=pds.DEFAULT_STORE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stack-label", required=True)
    parser.add_argument(
        "--first-frame", type=Path, default=pds.DEFAULT_FIRST_FRAME
    )
    parser.add_argument("--goldens", type=Path, default=pds.DEFAULT_GOLDENS)
    parser.add_argument("--prompt", default="a red ball bouncing")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--resolution", default="256p")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pds._validate_environment()

    import diffusers

    from omni_infinity.registry import resolve_profile
    from omni_infinity.runner import _transformer_component

    results: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "phase1 upgrade-matrix stack check",
        "stack_label": args.stack_label,
        "compile_kwargs": {
            key: str(val) for key, val in COMPILE_KWARGS.items()
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "diffusers": diffusers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "torchinductor_cache_dir": os.environ.get(
                "TORCHINDUCTOR_CACHE_DIR", ""
            ),
            "checkpoint": str(args.checkpoint),
            "store_dir": str(args.store_dir),
        },
        "parameters": {
            "prompt": args.prompt,
            "steps": args.steps,
            "resolution": args.resolution,
            "frames": args.frames,
            "seed": args.seed,
        },
    }
    try:
        import transformers

        results["environment"]["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001 - version report only
        pass

    log(f"stack={args.stack_label} torch={torch.__version__}")
    results["smoke"] = sm120_fp8_smoke()
    log(f"smoke: {results['smoke']}")

    resolved = resolve_profile(
        "h3-dense", ("adaln-host-cache",), checkpoint=str(args.checkpoint)
    )
    runner_kwargs = dict(resolved.runner_kwargs)
    runner_kwargs.update(
        {
            "offload": True,
            "store_dir": str(args.store_dir),
            "store_components": ("transformer", "vae", "audio_vae"),
        }
    )
    log("loading resident profile (adaln-host-cache)")
    with pds.patched_transformer_config_lookup(args.checkpoint):
        runner = resolved.runner.from_pretrained(
            resolved.checkpoint,
            device="cuda",
            torch_dtype=torch.bfloat16,
            **runner_kwargs,
        )
    from PIL import Image

    first_frame = Image.open(args.first_frame).convert("RGB")

    def generate() -> tuple[torch.Tensor, float]:
        started = time.perf_counter()
        result = runner.generate(
            args.prompt,
            image=first_frame,
            seed=args.seed,
            num_inference_steps=args.steps,
            resolution=args.resolution,
            num_frames=args.frames,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        latents = result.latents.detach().to("cpu", copy=True)
        del result
        torch.cuda.empty_cache()
        return latents, elapsed

    log("eager generation")
    eager_latents, eager_seconds = generate()
    results["eager_wall_seconds"] = eager_seconds

    golden = torch.load(args.goldens, map_location="cpu", weights_only=False)
    results["golden_provenance"] = pds.golden_provenance(
        golden,
        torch_version=torch.__version__,
        diffusers_version=diffusers.__version__,
        gpu_name=torch.cuda.get_device_name(0),
        first_frame_sha256=hashlib.sha256(
            args.first_frame.read_bytes()
        ).hexdigest(),
        prompt=args.prompt,
        seed=args.seed,
        steps=args.steps,
        resolution=args.resolution,
        frames=args.frames,
    )
    results["eager_vs_golden"] = pds.latent_parity(
        eager_latents, golden["latents"]
    )
    log(
        "eager vs golden: "
        f"tier={results['eager_vs_golden']['tier']} "
        f"rms_rel={results['eager_vs_golden']['rms_rel']:.6g}"
    )

    log("applying compile_repeated_blocks(fullgraph=True, dynamic=True)")
    from torch._dynamo.utils import counters

    counters.clear()
    transformer = _transformer_component(runner.pipeline)
    transformer.compile_repeated_blocks(**COMPILE_KWARGS)
    log("compiled generation (includes cold-start compile)")
    compiled_latents, compiled_seconds = generate()
    results["compiled_wall_seconds_cold"] = compiled_seconds
    results["dynamo_unique_graphs"] = int(counters["stats"]["unique_graphs"])

    results["compiled_vs_golden"] = pds.latent_parity(
        compiled_latents, golden["latents"]
    )
    results["compiled_vs_eager"] = pds.latent_parity(
        compiled_latents, eager_latents
    )
    results["gate_compile_vs_eager_2e2"] = bool(
        results["compiled_vs_eager"]["allclose"]["rtol=atol=2e-2"]
    )
    results["gate_compile_vs_golden_2e2"] = bool(
        results["compiled_vs_golden"]["allclose"]["rtol=atol=2e-2"]
    )
    for name in ("eager_vs_golden", "compiled_vs_golden", "compiled_vs_eager"):
        log(
            f"{name}: tier={results[name]['tier']} "
            f"rms_rel={results[name]['rms_rel']:.6g}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    log(f"wrote {args.output}")
    if not results["gate_compile_vs_eager_2e2"]:
        log("stack gate failed: compiled vs eager exceeds allclose 2e-2")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
