#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Phase-0 compile-blocks divergence localization probe (pinned stack).

Measures, on one GPU and one model load, WHERE the ``compile-blocks``
golden-parity divergence (video-latent ``rms_rel ~= 0.0726`` end to end)
enters, using real step-1 activations captured from an eager generation:

1. capture: eager ``resident`` profile generation (adaln-host-cache) with
   forward-pre-hooks on every transformer block; the eager run doubles as
   the bitwise golden control.
2. eager replay sanity: re-running each block on its captured input must
   reproduce the in-situ output bitwise, or replays are not trustworthy.
3. backend bisect: ``torch.compile`` with ``eager`` / ``aot_eager`` /
   ``inductor`` backends on sampled blocks separates Dynamo tracing,
   AOT decompositions, and Inductor codegen as divergence sources.
4. piece bisect: a bitwise-verified reconstruction of the block forward
   swaps exactly one piece (adaln / norm+modulate / attn / residual-gate
   / ff, plus attention internals) to a compiled version to attribute the
   per-block error to named ops.
5. inductor config variants: precision-oriented Inductor configs on the
   worst block, searching for an in-stack mitigation.
6. production path: ``compile_repeated_blocks(fullgraph=True,
   dynamic=True)`` exactly as the runner applies it; per-block and
   chained (within-forward) divergence tables, plus token-refiner blocks.

Single-forward errors here are per-forward contributions; the end-to-end
golden gap additionally amplifies across 50 blocks x 8 steps. End-to-end
verification of any promising config stays with
``benchmarks/profile_denoise_step.py --profile compile-resident``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))

import profile_denoise_step as pds  # noqa: E402  (sibling script reuse)

COMPILE_KWARGS = {"fullgraph": True, "dynamic": True}


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    candidate = candidate.detach().float()
    reference = reference.detach().float()
    rms_rel = (
        (candidate - reference).square().mean().sqrt()
        / reference.square().mean().sqrt()
    ).item()
    max_abs = (candidate - reference).abs().max().item()
    return {
        "rms_rel": rms_rel,
        "max_abs": max_abs,
        "bitwise": bool(torch.equal(candidate, reference)),
    }


class StepOneCapture:
    """Capture every transformer block's inputs on the first forward."""

    def __init__(self, transformer: torch.nn.Module) -> None:
        self.transformer = transformer
        self.blocks = list(transformer.transformer_blocks)
        self.refiner_blocks = list(transformer.token_refiner.refiner_blocks)
        self.block_inputs: list[torch.Tensor | None] = [None] * len(self.blocks)
        self.refiner_inputs: list[torch.Tensor | None] = [None] * len(
            self.refiner_blocks
        )
        self.final_norm_input: torch.Tensor | None = None
        self.norm_out_input: torch.Tensor | None = None
        self.shared: dict[str, Any] = {}
        self._forward_index = 0
        self._handles: list[Any] = []

    def _active(self) -> bool:
        return self._forward_index == 1

    def __enter__(self) -> StepOneCapture:
        def transformer_pre(module, args, kwargs):
            self._forward_index += 1

        self._handles.append(
            self.transformer.register_forward_pre_hook(
                transformer_pre, with_kwargs=True
            )
        )

        def make_block_pre(index: int):
            def block_pre(module, args):
                if not self._active():
                    return
                hidden_states, temb, adaln_indices, rotary_emb = args[:4]
                self.block_inputs[index] = hidden_states.detach().to(
                    "cpu", copy=True
                )
                if not self.shared:
                    self.shared = {
                        "temb": temb.detach().to("cpu", copy=True),
                        "adaln_indices": adaln_indices.detach().to(
                            "cpu", copy=True
                        ),
                        "rotary_emb": tuple(
                            part.detach().to("cpu", copy=True)
                            for part in rotary_emb
                        ),
                    }

            return block_pre

        for index, block in enumerate(self.blocks):
            self._handles.append(
                block.register_forward_pre_hook(make_block_pre(index))
            )

        def make_refiner_pre(index: int):
            def refiner_pre(module, args):
                if self._active():
                    self.refiner_inputs[index] = (
                        args[0].detach().to("cpu", copy=True)
                    )

            return refiner_pre

        for index, block in enumerate(self.refiner_blocks):
            self._handles.append(
                block.register_forward_pre_hook(make_refiner_pre(index))
            )

        def final_norm_pre(module, args):
            if self._active():
                self.final_norm_input = args[0].detach().to("cpu", copy=True)

        self._handles.append(
            self.transformer.token_refiner.final_norm.register_forward_pre_hook(
                final_norm_pre
            )
        )

        def norm_out_pre(module, args):
            if self._active() and self.norm_out_input is None:
                self.norm_out_input = args[0].detach().to("cpu", copy=True)

        self._handles.append(
            self.transformer.norm_out.register_forward_pre_hook(norm_out_pre)
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def block_eager_fns(block, module) -> dict[str, Callable]:
    """Eager pieces mirroring MiniMaxH3TransformerBlock.forward exactly."""

    def norm_mod(norm):
        def fn(hidden_states, scale, shift, adaln_indices):
            normed = norm(hidden_states)
            return normed * (
                1.0 + scale.index_select(0, adaln_indices)
            ) + shift.index_select(0, adaln_indices)

        return fn

    def residual_gate(residual, gate, branch, adaln_indices):
        return residual + gate.index_select(0, adaln_indices) * branch

    return {
        "adaln": lambda temb: block.adaln_proj(temb),
        "norm1mod": norm_mod(block.norm1),
        "attn": lambda x, rotary_emb: block.attn(x, rotary_emb, None),
        "res1": residual_gate,
        "norm2mod": norm_mod(block.norm2),
        "ff": lambda x: block.ff(x),
        "res2": residual_gate,
    }


def attn_eager_fns(attn, module) -> dict[str, Callable]:
    """Eager pieces mirroring MiniMaxH3AttnProcessor.__call__ exactly."""

    def qkv(hidden_states):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        return query, key, value

    def qknorm_rope(query, key, rotary_emb):
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = module._apply_rotary_emb(query, *rotary_emb)
        key = module._apply_rotary_emb(key, *rotary_emb)
        return query, key

    def sdpa(query, key, value):
        return module.dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=None,
            parallel_config=None,
        )

    def out(hidden_states, query):
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)

    return {"qkv": qkv, "qknorm_rope": qknorm_rope, "sdpa": sdpa, "out": out}


def block_recon_forward(fns, attn_fns, hidden_states, shared):
    temb = shared["temb"]
    adaln_indices = shared["adaln_indices"]
    rotary_emb = shared["rotary_emb"]
    (
        shift_msa,
        scale_msa,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
    ) = fns["adaln"](temb)
    residual = hidden_states
    normed = fns["norm1mod"](hidden_states, scale_msa, shift_msa, adaln_indices)
    if attn_fns is None:
        attn_output = fns["attn"](normed, rotary_emb)
    else:
        query, key, value = attn_fns["qkv"](normed)
        query, key = attn_fns["qknorm_rope"](query, key, rotary_emb)
        attn_output = attn_fns["sdpa"](query, key, value)
        attn_output = attn_fns["out"](attn_output, query)
    hidden_states = fns["res1"](residual, gate_msa, attn_output, adaln_indices)
    residual = hidden_states
    normed = fns["norm2mod"](hidden_states, scale_mlp, shift_mlp, adaln_indices)
    ff_output = fns["ff"](normed)
    return fns["res2"](residual, gate_mlp, ff_output, adaln_indices)


def compiled_variant(fns: dict, name: str, **compile_kwargs) -> dict:
    variant = dict(fns)
    torch._dynamo.reset()
    variant[name] = torch.compile(
        fns[name], **{**COMPILE_KWARGS, **compile_kwargs}
    )
    return variant


def run_block_eager(block, hidden_states, shared) -> torch.Tensor:
    return block(
        hidden_states,
        shared["temb"],
        shared["adaln_indices"],
        shared["rotary_emb"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=pds.DEFAULT_CHECKPOINT
    )
    parser.add_argument("--store-dir", type=Path, default=pds.DEFAULT_STORE)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO / "results" / "p2_parity_followup" / "phase0",
    )
    parser.add_argument(
        "--first-frame", type=Path, default=pds.DEFAULT_FIRST_FRAME
    )
    parser.add_argument("--goldens", type=Path, default=pds.DEFAULT_GOLDENS)
    parser.add_argument("--prompt", default="a red ball bouncing")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--resolution", default="256p")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample-blocks",
        type=int,
        nargs="+",
        default=[0, 24, 49],
        help="block indices for backend/piece/config bisects",
    )
    args = parser.parse_args()
    pds._validate_environment()

    from diffusers.models.transformers import transformer_minimax_h3 as m

    from omni_infinity.registry import resolve_profile
    from omni_infinity.runner import _transformer_component

    results: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "phase0 compile-blocks divergence localization",
        "compile_kwargs": {
            key: str(val) for key, val in COMPILE_KWARGS.items()
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
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
            "sample_blocks": args.sample_blocks,
        },
    }

    # ---- stage 1: load the eager resident profile and capture step 1 ----
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
    log(f"loading eager resident profile on {torch.cuda.get_device_name(0)}")
    with pds.patched_transformer_config_lookup(args.checkpoint):
        runner = resolved.runner.from_pretrained(
            resolved.checkpoint,
            device="cuda",
            torch_dtype=torch.bfloat16,
            **runner_kwargs,
        )
    transformer = _transformer_component(runner.pipeline)
    from PIL import Image

    first_frame = Image.open(args.first_frame).convert("RGB")
    log("starting eager control generation with step-1 capture hooks")
    started = time.perf_counter()
    with StepOneCapture(transformer) as capture:
        result = runner.generate(
            args.prompt,
            image=first_frame,
            seed=args.seed,
            num_inference_steps=args.steps,
            resolution=args.resolution,
            num_frames=args.frames,
        )
    torch.cuda.synchronize()
    log(f"eager generation done in {time.perf_counter() - started:.1f}s")

    golden = torch.load(args.goldens, map_location="cpu", weights_only=False)
    control = pds.latent_parity(result.latents, golden["latents"])
    results["eager_control_parity"] = control
    log(f"eager control: tier={control['tier']} bitwise={control['bitwise']}")
    if not control["bitwise"]:
        log("FATAL: eager control is not bitwise; captures are not golden")
    del result, golden
    torch.cuda.empty_cache()

    blocks = capture.blocks
    missing = [i for i, t in enumerate(capture.block_inputs) if t is None]
    if missing or capture.norm_out_input is None:
        raise RuntimeError(f"capture incomplete: missing blocks {missing}")
    shared_gpu = {
        "temb": capture.shared["temb"].cuda(),
        "adaln_indices": capture.shared["adaln_indices"].cuda(),
        "rotary_emb": tuple(
            part.cuda() for part in capture.shared["rotary_emb"]
        ),
    }
    seq_len = capture.block_inputs[0].shape[1]
    results["capture"] = {
        "blocks": len(blocks),
        "refiner_blocks": len(capture.refiner_blocks),
        "seq_len": int(seq_len),
        "hidden_size": int(capture.block_inputs[0].shape[-1]),
        "temb_dtype": str(capture.shared["temb"].dtype),
        "hidden_dtype": str(capture.block_inputs[0].dtype),
    }
    log(f"captured {len(blocks)} block inputs, seq_len={seq_len}")

    def block_input(index: int) -> torch.Tensor:
        return capture.block_inputs[index].cuda()

    def block_reference(index: int) -> torch.Tensor:
        if index + 1 < len(blocks):
            return capture.block_inputs[index + 1].cuda()
        return capture.norm_out_input.cuda()

    # ---- stage 2: eager replay sanity ----
    log("stage 2: eager replay sanity (expect bitwise)")
    replay_rows = []
    with torch.no_grad():
        for index in range(len(blocks)):
            out = run_block_eager(blocks[index], block_input(index), shared_gpu)
            replay_rows.append(metrics(out, block_reference(index)))
            del out
    nonbitwise = [i for i, row in enumerate(replay_rows) if not row["bitwise"]]
    results["eager_replay"] = {
        "all_bitwise": not nonbitwise,
        "nonbitwise_blocks": nonbitwise,
        "max_rms_rel": max(row["rms_rel"] for row in replay_rows),
    }
    log(f"eager replay: all_bitwise={not nonbitwise} nonbitwise={nonbitwise}")

    # ---- stage 3: backend bisect on sampled blocks ----
    log("stage 3: backend bisect (eager / aot_eager / inductor)")
    backend_rows: dict[str, dict[str, dict]] = {}
    with torch.no_grad():
        for index in args.sample_blocks:
            block = blocks[index]
            reference = block_reference(index)
            row = {}
            for backend in ("eager", "aot_eager", "inductor"):
                torch._dynamo.reset()
                compiled = torch.compile(
                    run_block_eager, backend=backend, **COMPILE_KWARGS
                )
                out = compiled(block, block_input(index), shared_gpu)
                row[backend] = metrics(out, reference)
                del out, compiled
                log(
                    f"  block {index} backend={backend}: "
                    f"rms_rel={row[backend]['rms_rel']:.3e} "
                    f"bitwise={row[backend]['bitwise']}"
                )
            backend_rows[str(index)] = row
            del reference
            torch.cuda.empty_cache()
    results["backend_bisect"] = backend_rows

    # ---- stage 4: piece bisect on sampled blocks ----
    log("stage 4: piece bisect (one compiled piece at a time)")
    piece_rows: dict[str, dict[str, Any]] = {}
    piece_names = (
        "adaln",
        "norm1mod",
        "attn",
        "res1",
        "norm2mod",
        "ff",
        "res2",
    )
    attn_piece_names = ("qkv", "qknorm_rope", "sdpa", "out")
    with torch.no_grad():
        for index in args.sample_blocks:
            block = blocks[index]
            fns = block_eager_fns(block, m)
            attn_fns = attn_eager_fns(block.attn, m)
            hidden = block_input(index)
            reference = block_reference(index)
            row: dict[str, Any] = {}
            recon = block_recon_forward(fns, None, hidden, shared_gpu)
            row["recon_eager"] = metrics(recon, reference)
            recon_attn = block_recon_forward(fns, attn_fns, hidden, shared_gpu)
            row["recon_eager_attn_pieces"] = metrics(recon_attn, reference)
            del recon, recon_attn
            for name in piece_names:
                variant = compiled_variant(fns, name)
                out = block_recon_forward(variant, None, hidden, shared_gpu)
                row[name] = metrics(out, reference)
                del out, variant
                log(
                    f"  block {index} piece={name}: "
                    f"rms_rel={row[name]['rms_rel']:.3e} "
                    f"bitwise={row[name]['bitwise']}"
                )
            for name in attn_piece_names:
                variant = compiled_variant(attn_fns, name)
                out = block_recon_forward(fns, variant, hidden, shared_gpu)
                row[f"attn_{name}"] = metrics(out, reference)
                del out, variant
                log(
                    f"  block {index} piece=attn_{name}: "
                    f"rms_rel={row[f'attn_{name}']['rms_rel']:.3e} "
                    f"bitwise={row[f'attn_{name}']['bitwise']}"
                )
            piece_rows[str(index)] = row
            del hidden, reference
            torch.cuda.empty_cache()
    results["piece_bisect"] = piece_rows

    # ---- stage 5: inductor config variants on the worst sampled block ----
    inductor_config = torch._inductor.config
    config_variants: dict[str, dict[str, Any]] = {
        "default": {},
        "force_same_precision": {"force_same_precision": True},
        "emulate_precision_casts": {"emulate_precision_casts": True},
        "force_same+emulate": {
            "force_same_precision": True,
            "emulate_precision_casts": True,
        },
        "pattern_matcher_off": {"pattern_matcher": False},
        "fuse_off": {"max_fusion_size": 1, "epilogue_fusion": False},
    }
    worst = max(
        args.sample_blocks,
        key=lambda i: backend_rows[str(i)]["inductor"]["rms_rel"],
    )
    log(f"stage 5: inductor config variants on worst block {worst}")
    config_rows: dict[str, Any] = {"block": worst}
    with torch.no_grad():
        block = blocks[worst]
        reference = block_reference(worst)
        for variant_name, overrides in config_variants.items():
            valid = {
                key: val
                for key, val in overrides.items()
                if hasattr(inductor_config, key)
            }
            skipped = sorted(set(overrides) - set(valid))
            saved = {key: getattr(inductor_config, key) for key in valid}
            try:
                for key, val in valid.items():
                    setattr(inductor_config, key, val)
                torch._dynamo.reset()
                compiled = torch.compile(run_block_eager, **COMPILE_KWARGS)
                out = compiled(block, block_input(worst), shared_gpu)
                config_rows[variant_name] = metrics(out, reference)
                config_rows[variant_name]["skipped_keys"] = skipped
                del out, compiled
            finally:
                for key, val in saved.items():
                    setattr(inductor_config, key, val)
            log(
                f"  config={variant_name}: "
                f"rms_rel={config_rows[variant_name]['rms_rel']:.3e} "
                f"bitwise={config_rows[variant_name]['bitwise']} "
                f"skipped={skipped}"
            )
        del reference
        torch.cuda.empty_cache()
    results["inductor_config_variants"] = config_rows

    # ---- stage 6: production compile_repeated_blocks path ----
    log("stage 6: production compile_repeated_blocks(fullgraph, dynamic)")
    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()
    transformer.compile_repeated_blocks(**COMPILE_KWARGS)
    per_block = []
    chained = []
    with torch.no_grad():
        hidden = block_input(0)
        chained_hidden = hidden.clone()
        for index in range(len(blocks)):
            reference = block_reference(index)
            out = run_block_eager(blocks[index], block_input(index), shared_gpu)
            per_block.append(metrics(out, reference))
            chained_hidden = run_block_eager(
                blocks[index], chained_hidden, shared_gpu
            )
            chained.append(metrics(chained_hidden, reference))
            del out, reference
        refiner_rows = []
        for index, block in enumerate(capture.refiner_blocks):
            if index + 1 < len(capture.refiner_blocks):
                reference = capture.refiner_inputs[index + 1].cuda()
            else:
                reference = capture.final_norm_input.cuda()
            out = block(capture.refiner_inputs[index].cuda())
            refiner_rows.append(metrics(out, reference))
            del out, reference
    unique_graphs = int(counters["stats"]["unique_graphs"])
    worst_block = max(
        range(len(per_block)), key=lambda i: per_block[i]["rms_rel"]
    )
    results["production_compile"] = {
        "unique_graphs": unique_graphs,
        "per_block": per_block,
        "per_block_rms_rel_max": per_block[worst_block]["rms_rel"],
        "per_block_rms_rel_argmax": worst_block,
        "per_block_rms_rel_median": sorted(row["rms_rel"] for row in per_block)[
            len(per_block) // 2
        ],
        "chained": chained,
        "chained_final_rms_rel": chained[-1]["rms_rel"],
        "refiner_per_block": refiner_rows,
    }
    log(
        f"production: unique_graphs={unique_graphs} "
        f"per-block max rms_rel={per_block[worst_block]['rms_rel']:.3e} "
        f"(block {worst_block}) chained final "
        f"rms_rel={chained[-1]['rms_rel']:.3e}"
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / "probe.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
