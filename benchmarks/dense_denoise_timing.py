#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Measure dense BF16 MiniMax-H3 denoise forwards with CUDA events."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from contextlib import contextmanager
from pathlib import Path

import torch

from omni_infinity.registry import resolve_profile
from omni_infinity.runner import _transformer_component

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    "/mnt/raid0nvme0/leyang/.cache/huggingface/hub/"
    "models--MiniMaxAI--MiniMax-H3/snapshots/"
    "42ed227ee7df40d41602854ae760620d6eb651fe"
)
DEFAULT_STORE = Path("/mnt/raid0nvme0/leyang/h3-store-v2")
DEFAULT_FIRST_FRAME = REPO / "tests" / "fixtures" / "ref.png"
BYTE_FLOOR_SECONDS = 0.11
_MISSING = object()


def transformer_config_root(checkpoint: Path) -> Path:
    """Locate the FL2VA transformer config in a merged H3 snapshot."""
    if (checkpoint / "transformer" / "config.json").is_file():
        return checkpoint
    nested = checkpoint / "FL2VA"
    if (nested / "transformer" / "config.json").is_file():
        return nested
    raise FileNotFoundError(f"no FL2VA transformer config under {checkpoint}")


@contextmanager
def patched_transformer_config_lookup(checkpoint: Path):
    """Adapt the store loader to a release snapshot's nested FL2VA config."""
    from diffusers import MiniMaxH3Transformer3DModel

    config_root = transformer_config_root(checkpoint)
    if config_root == checkpoint:
        yield
        return
    original = MiniMaxH3Transformer3DModel.load_config
    local_descriptor = MiniMaxH3Transformer3DModel.__dict__.get(
        "load_config", _MISSING
    )

    def load_config(cls, pretrained_model_name_or_path, *args, **kwargs):
        requested = Path(pretrained_model_name_or_path)
        if requested == checkpoint and kwargs.get("subfolder") == "transformer":
            pretrained_model_name_or_path = config_root
        return original(pretrained_model_name_or_path, *args, **kwargs)

    MiniMaxH3Transformer3DModel.load_config = classmethod(load_config)
    try:
        print(
            "using nested transformer config from "
            f"{config_root / 'transformer'}",
            flush=True,
        )
        yield
    finally:
        if local_descriptor is _MISSING:
            del MiniMaxH3Transformer3DModel.load_config
        else:
            MiniMaxH3Transformer3DModel.load_config = local_descriptor


class DenoiseTimer:
    """Time dense transformer forwards and the denoise-window memory peak."""

    def __init__(self, transformer: torch.nn.Module, total_steps: int) -> None:
        self.transformer = transformer
        self.total_steps = total_steps
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.peak_bytes = 0
        self._pre_handle = None
        self._post_handle = None

    def __enter__(self) -> "DenoiseTimer":
        self._pre_handle = self.transformer.register_forward_pre_hook(self._pre)
        self._post_handle = self.transformer.register_forward_hook(self._post)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._pre_handle is not None:
            self._pre_handle.remove()
        if self._post_handle is not None:
            self._post_handle.remove()

    def _pre(self, module, args) -> None:
        if not self.events:
            torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.events.append((start, end))
        print(
            f"step {len(self.events)}/{self.total_steps}: "
            "transformer forward started",
            flush=True,
        )

    def _post(self, module, args, output) -> None:
        self.events[-1][1].record()
        self.peak_bytes = max(
            self.peak_bytes, torch.cuda.max_memory_allocated()
        )
        print(
            f"step {len(self.events)}/{self.total_steps}: "
            "transformer forward done",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="256p")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--first-frame", type=Path, default=DEFAULT_FIRST_FRAME)
    parser.add_argument("--prompt", default="a red ball bouncing")
    return parser.parse_args()


def validate_environment() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise SystemExit(
            "set CUDA_VISIBLE_DEVICES to exactly one idle GPU index"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("exactly one visible CUDA device is required")


def main() -> int:
    args = parse_args()
    validate_environment()
    if args.steps < 2:
        raise SystemExit("--steps must be at least 2 (one warmup is discarded)")

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
    card = torch.cuda.get_device_name(0)
    cuda_cc = ".".join(
        str(value) for value in torch.cuda.get_device_capability(0)
    )
    print(
        "loading dense BF16 resident profile "
        "(store-backed, adaln-host-cache, no block-stream) "
        f"on {card}",
        flush=True,
    )
    with patched_transformer_config_lookup(args.checkpoint):
        runner = resolved.runner.from_pretrained(
            resolved.checkpoint,
            device="cuda",
            torch_dtype=torch.bfloat16,
            **runner_kwargs,
        )

    from PIL import Image

    transformer = _transformer_component(runner.pipeline)
    first_frame = Image.open(args.first_frame).convert("RGB")
    with DenoiseTimer(transformer, args.steps) as timer:
        result = runner.generate(
            args.prompt,
            image=first_frame,
            seed=args.seed,
            num_inference_steps=args.steps,
            resolution=args.resolution,
            num_frames=args.frames,
        )
    torch.cuda.synchronize()

    if len(timer.events) < 2:
        raise RuntimeError(
            "expected at least two transformer forwards, "
            f"saw {len(timer.events)}"
        )
    if not bool(torch.isfinite(result.latents).all().item()):
        raise RuntimeError("non-finite video latents")
    print(
        f"requested inference steps={args.steps}; "
        f"actual transformer forwards={len(timer.events)}",
        flush=True,
    )
    step_wall_ms = [start.elapsed_time(end) for start, end in timer.events]
    median_step_wall_ms = statistics.median(step_wall_ms[1:])
    peak_gib = timer.peak_bytes / 1024**3
    print(
        f"median_step_wall_ms={median_step_wall_ms:.3f}; "
        f"peak_gib={peak_gib:.3f}; finite_latents=True",
        flush=True,
    )
    if median_step_wall_ms / 1000 < BYTE_FLOOR_SECONDS:
        raise RuntimeError(
            "median step is below the 0.11 s byte-bound sanity floor: "
            f"{median_step_wall_ms / 1000:.6f} s"
        )
    print(
        "byte-floor gate: PASS "
        f"({median_step_wall_ms / 1000:.3f} s >= {BYTE_FLOOR_SECONDS:.2f} s)",
        flush=True,
    )

    payload = {
        "median_step_wall_ms": median_step_wall_ms,
        "peak_gib": peak_gib,
        "cuda_cc": cuda_cc,
        "card": card,
        "steps": args.steps,
        "actual_transformer_forwards": len(timer.events),
        "warmup_steps_discarded": 1,
        "resolution": args.resolution,
        "frames": args.frames,
        "seed": args.seed,
        "profile": "dense BF16 resident store-backed adaln-host-cache",
        "finite_latents": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
