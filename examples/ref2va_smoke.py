# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Ref2VA smoke: animate a reference image with MiniMax-H3."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import torch

from examples.fl2va_smoke import (
    export_outputs,
    parse_vram,
    report_latent_parity,
)
from omni_infinity.runner import RESOLUTIONS, ReferenceRunner

DEFAULT_STORE = "/mnt/raid0nvme0/leyang/h3-store-v2"
DEFAULT_GOLDENS = Path("tests/fixtures/goldens/ref2va_goldens.pt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument(
        "--prompt", default="Animate the referenced subject naturally."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--resolution", default="256p", choices=sorted(RESOLUTIONS)
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--max-vram", default=None)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("OMNI_H3_CHECKPOINT", "MiniMaxAI/MiniMax-H3"),
    )
    parser.add_argument("--store-dir", default=DEFAULT_STORE)
    parser.add_argument("--goldens", type=Path, default=DEFAULT_GOLDENS)
    parser.add_argument("--record-goldens", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--full-resident",
        action="store_true",
        help="record the stock Ref2VA baseline without store/streaming hooks",
    )
    return parser.parse_args(argv)


def runner_options(args: argparse.Namespace) -> dict:
    if args.full_resident:
        return {
            "workflow": "ref2va",
            "offload": True,
            "store_dir": None,
            "store_components": (),
            "adaln_host_cache": False,
            "block_stream_blocks_per_group": 0,
            "stream_text_encoder": False,
            "step_overlap": False,
        }
    return {
        "workflow": "ref2va",
        "offload": True,
        "store_dir": args.store_dir,
        "store_components": ("transformer_ref", "vae", "audio_vae"),
        "adaln_host_cache": True,
        "block_stream_blocks_per_group": 1,
        "stream_text_encoder": True,
        "step_overlap": True,
    }


def build_references(ref_path: Path):
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

    return [MiniMaxH3ImageReference.from_file(ref_path)]


def _reference_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_goldens(result, goldens_dir: Path, args) -> None:
    import diffusers

    goldens_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": args.prompt,
        "reference_sha256": _reference_sha256(args.ref),
        "seed": args.seed,
        "steps": args.steps,
        "resolution": args.resolution,
        "frames": args.frames,
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "latents": (
            result.latents.cpu() if result.latents is not None else None
        ),
        "audio_latents": (
            result.audio_latents.cpu()
            if result.audio_latents is not None
            else None
        ),
    }
    path = goldens_dir / "ref2va_goldens.pt"
    torch.save(payload, path)
    print(f"goldens recorded at {path}")


def _offload_margin(max_vram: str | None) -> str | None:
    if max_vram is None or not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    margin = max(total - parse_vram(max_vram), 0)
    return f"{margin / 1e9:.1f}GB"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.full_resident and args.record_goldens and args.frames < 120:
        raise SystemExit(
            "full-resident golden recording requires frames >= 120"
        )

    options = runner_options(args)
    options["offload_memory_margin"] = _offload_margin(args.max_vram)
    runner = ReferenceRunner.from_pretrained(args.checkpoint, **options)
    references = build_references(args.ref)

    if args.max_vram is not None:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = runner.generate(
        args.prompt,
        references=references,
        seed=args.seed,
        num_inference_steps=args.steps,
        resolution=args.resolution,
        num_frames=args.frames,
    )
    elapsed = time.perf_counter() - start
    print(f"generate wall-clock: {elapsed:.1f}s")

    parity_ok = True
    recording_baseline = args.full_resident and args.record_goldens is not None
    if not recording_baseline and args.goldens is not None:
        parity_ok = report_latent_parity(result, args.goldens)

    vram_ok = True
    if args.max_vram is not None:
        peak = torch.cuda.max_memory_allocated()
        budget = parse_vram(args.max_vram)
        vram_ok = peak < budget
        print(
            "full pipeline-window max_memory_allocated: "
            f"{peak / (1024**3):.2f} GiB "
            f"(budget {budget / (1024**3):.2f} GiB) ok={vram_ok}"
        )

    if args.record_goldens is not None:
        if not args.full_resident:
            raise SystemExit("--record-goldens requires --full-resident")
        record_goldens(result, args.record_goldens, args)

    export_outputs(result, args.output_dir)
    if not (parity_ok and vram_ok):
        raise SystemExit("QA gate failed (see parity / max_memory above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
