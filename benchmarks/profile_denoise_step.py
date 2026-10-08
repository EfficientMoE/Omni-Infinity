#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Profile MiniMax-H3 denoise-step launch gaps with torch.profiler.

The profiler records one user annotation per transformer forward. CUDA
annotations provide per-stream step windows; kernel and memcpy intervals are
then clipped to those windows and unioned, preserving overlap. ``launch_gap``
excludes both kernel-busy and copy-busy intervals so block-stream weight
transfers are not mislabeled as CPU launch overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from omni_infinity.registry import resolve_profile
from omni_infinity.runner import _transformer_component

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    "/mnt/raid0nvme0/leyang/.cache/huggingface/hub/"
    "models--MiniMaxAI--MiniMax-H3/snapshots/"
    "42ed227ee7df40d41602854ae760620d6eb651fe"
)
DEFAULT_STORE = Path("/mnt/raid0nvme0/leyang/h3-store-v2")
DEFAULT_RESULTS = REPO / "results" / "p2_phase0"
DEFAULT_FIRST_FRAME = REPO / "tests" / "fixtures" / "ref.png"
PROFILE_OPTIMIZATIONS = {
    "resident": ("adaln-host-cache",),
    "block-stream": (
        "adaln-host-cache",
        "block-stream",
        "text-encoder-stream",
    ),
}
STEP_PREFIX = "denoise_step_"
_MISSING = object()


def union_duration_us(intervals: list[tuple[float, float]]) -> float:
    """Return the duration of the union of half-open intervals."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def clipped_interval(
    start: float,
    end: float,
    window_start: float,
    window_end: float,
) -> tuple[float, float] | None:
    clipped = max(start, window_start), min(end, window_end)
    return clipped if clipped[1] > clipped[0] else None


def transformer_config_root(checkpoint: Path) -> Path:
    """Locate the FL2VA transformer config in merged H3 snapshots."""
    if (checkpoint / "transformer" / "config.json").is_file():
        return checkpoint
    nested = checkpoint / "FL2VA"
    if (nested / "transformer" / "config.json").is_file():
        return nested
    raise FileNotFoundError(f"no FL2VA transformer config under {checkpoint}")


def effective_video_frames(requested_frames: int) -> int:
    """Mirror the H3 video VAE's 17*n+5 frame alignment."""
    effective = requested_frames
    while effective % 17 != 5:
        effective += 1
    return effective


@contextmanager
def patched_transformer_config_lookup(checkpoint: Path):
    """Adapt the store loader to the release snapshot's nested FL2VA config."""
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
        transformer_dir = config_root / "transformer"
        print(
            f"using nested transformer config from {transformer_dir}",
            flush=True,
        )
        yield
    finally:
        if local_descriptor is _MISSING:
            del MiniMaxH3Transformer3DModel.load_config
        else:
            MiniMaxH3Transformer3DModel.load_config = local_descriptor


@dataclass
class StepMarker:
    index: int
    annotation: Any
    cuda_start: torch.cuda.Event
    cuda_end: torch.cuda.Event
    cpu_started: float
    cpu_enqueue_ms: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0


class DenoiseStepProbe:
    """Mark transformer forwards without synchronizing between steps."""

    def __init__(self, transformer: torch.nn.Module, total_steps: int) -> None:
        self.transformer = transformer
        self.total_steps = total_steps
        self.steps: list[StepMarker] = []
        self._pre_handle = None
        self._post_handle = None

    def __enter__(self) -> DenoiseStepProbe:
        self._pre_handle = self.transformer.register_forward_pre_hook(self._pre)
        self._post_handle = self.transformer.register_forward_hook(self._post)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._pre_handle is not None:
            self._pre_handle.remove()
        if self._post_handle is not None:
            self._post_handle.remove()

    def _pre(self, module, args) -> None:
        index = len(self.steps) + 1
        if index == 1:
            torch.cuda.reset_peak_memory_stats()
        annotation = record_function(f"{STEP_PREFIX}{index}")
        annotation.__enter__()
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
        self.steps.append(
            StepMarker(
                index=index,
                annotation=annotation,
                cuda_start=cuda_start,
                cuda_end=cuda_end,
                cpu_started=time.perf_counter(),
            )
        )
        print(
            f"step {index}/{self.total_steps}: transformer forward started",
            flush=True,
        )

    def _post(self, module, args, output) -> None:
        marker = self.steps[-1]
        marker.cuda_end.record()
        marker.annotation.__exit__(None, None, None)
        marker.cpu_enqueue_ms = (time.perf_counter() - marker.cpu_started) * 1e3
        marker.peak_allocated_bytes = torch.cuda.max_memory_allocated()
        marker.peak_reserved_bytes = torch.cuda.max_memory_reserved()
        print(
            f"step {marker.index}/{self.total_steps}: enqueued in "
            f"{marker.cpu_enqueue_ms:.1f} ms, peak allocated "
            f"{marker.peak_allocated_bytes / 1024**3:.2f} GiB",
            flush=True,
        )


def _step_stream_windows(
    events, step_name: str
) -> dict[int, tuple[float, float]]:
    windows: dict[int, tuple[float, float]] = {}
    for event in events:
        if (
            event.device_type != torch.autograd.DeviceType.CUDA
            or event.name != step_name
            or str(event.activity_type) != "gpu_user_annotation"
        ):
            continue
        resource = int(event.device_resource_id)
        interval = event.time_range.start, event.time_range.end
        previous = windows.get(resource)
        windows[resource] = (
            interval
            if previous is None
            else (min(previous[0], interval[0]), max(previous[1], interval[1]))
        )
    return windows


def _events_in_step(events, windows, activity_type: str):
    selected = []
    for event in events:
        if (
            event.device_type != torch.autograd.DeviceType.CUDA
            or str(event.activity_type) != activity_type
        ):
            continue
        window = windows.get(int(event.device_resource_id))
        if window is None:
            continue
        interval = clipped_interval(
            event.time_range.start,
            event.time_range.end,
            window[0],
            window[1],
        )
        if interval is not None:
            selected.append((event, interval))
    return selected


def analyze_steps(events, markers: list[StepMarker]) -> list[dict[str, Any]]:
    rows = []
    for marker in markers:
        name = f"{STEP_PREFIX}{marker.index}"
        windows = _step_stream_windows(events, name)
        if not windows:
            raise RuntimeError(
                f"torch.profiler produced no CUDA window for {name}"
            )
        kernels = _events_in_step(events, windows, "kernel")
        copies = _events_in_step(events, windows, "gpu_memcpy")
        compute_intervals = [interval for _, interval in kernels]
        copy_intervals = [interval for _, interval in copies]
        wall_start = min(start for start, _ in windows.values())
        wall_end = max(end for _, end in windows.values())
        wall_us = wall_end - wall_start
        compute_us = union_duration_us(compute_intervals)
        copy_us = union_duration_us(copy_intervals)
        active_us = union_duration_us(compute_intervals + copy_intervals)
        launch_gap_us = max(wall_us - active_us, 0.0)
        compute_gap_us = max(wall_us - compute_us, 0.0)
        h2d_intervals = [
            interval
            for event, interval in copies
            if "HtoD" in event.name or "Host To Device" in event.name
        ]
        rows.append(
            {
                "step": marker.index,
                "step_wall_ms": wall_us / 1e3,
                "cuda_event_wall_ms": marker.cuda_start.elapsed_time(
                    marker.cuda_end
                ),
                "cpu_enqueue_ms": marker.cpu_enqueue_ms,
                "compute_busy_ms": compute_us / 1e3,
                "copy_busy_ms": copy_us / 1e3,
                "h2d_busy_ms": union_duration_us(h2d_intervals) / 1e3,
                "compute_copy_union_ms": active_us / 1e3,
                "compute_only_gap_ms": compute_gap_us / 1e3,
                "launch_gap_ms": launch_gap_us / 1e3,
                "launch_gap_pct": 100.0 * launch_gap_us / wall_us,
                "kernel_count": len(kernels),
                "copy_count": len(copies),
                "compute_stream_ids": sorted(
                    {int(event.device_resource_id) for event, _ in kernels}
                ),
                "copy_stream_ids": sorted(
                    {int(event.device_resource_id) for event, _ in copies}
                ),
                "peak_allocated_gib": marker.peak_allocated_bytes / 1024**3,
                "peak_reserved_gib": marker.peak_reserved_bytes / 1024**3,
            }
        )
    return rows


def _median(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = rows[1:]
    if not measured:
        raise ValueError("at least two denoise steps are required")
    fields = (
        "step_wall_ms",
        "cuda_event_wall_ms",
        "cpu_enqueue_ms",
        "compute_busy_ms",
        "copy_busy_ms",
        "h2d_busy_ms",
        "compute_copy_union_ms",
        "compute_only_gap_ms",
        "launch_gap_ms",
        "launch_gap_pct",
        "kernel_count",
    )
    summary = {f"median_{field}": _median(measured, field) for field in fields}
    summary["peak_allocated_gib"] = max(
        row["peak_allocated_gib"] for row in rows
    )
    summary["peak_reserved_gib"] = max(row["peak_reserved_gib"] for row in rows)
    summary["warmup_steps_discarded"] = 1
    summary["measured_steps"] = len(measured)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_OPTIMIZATIONS), required=True
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--first-frame", type=Path, default=DEFAULT_FIRST_FRAME)
    parser.add_argument("--prompt", default="a red ball bouncing")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--resolution", default="256p")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _validate_environment() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise SystemExit(
            "set CUDA_VISIBLE_DEVICES to exactly one physical GPU index"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("exactly one visible CUDA device is required")


def main() -> int:
    args = _parse_args()
    _validate_environment()
    if args.steps < 2:
        raise SystemExit("--steps must be at least 2 (step 1 is warmup)")

    optimization_names = PROFILE_OPTIMIZATIONS[args.profile]
    resolved = resolve_profile(
        "h3-dense", optimization_names, checkpoint=str(args.checkpoint)
    )
    runner_kwargs = dict(resolved.runner_kwargs)
    runner_kwargs.update(
        {
            "offload": True,
            "store_dir": str(args.store_dir),
            "store_components": ("transformer", "vae", "audio_vae"),
        }
    )
    print(
        f"loading profile={args.profile} optimizations={optimization_names} "
        f"on {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    with patched_transformer_config_lookup(args.checkpoint):
        runner = resolved.runner.from_pretrained(
            resolved.checkpoint,
            device="cuda",
            torch_dtype=torch.bfloat16,
            **runner_kwargs,
        )
    transformer = _transformer_component(runner.pipeline)
    block_count = len(transformer.transformer_blocks)
    print(
        f"loaded transformer with {block_count} blocks; "
        "starting profiled generation",
        flush=True,
    )

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    from PIL import Image

    first_frame = Image.open(args.first_frame).convert("RGB")
    started = time.perf_counter()
    with DenoiseStepProbe(transformer, args.steps) as probe:
        with profile(activities=activities, acc_events=True) as profiler:
            runner.generate(
                args.prompt,
                image=first_frame,
                seed=args.seed,
                num_inference_steps=args.steps,
                resolution=args.resolution,
                num_frames=args.frames,
            )
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    actual_steps = len(probe.steps)
    if actual_steps < 2:
        raise RuntimeError(
            f"expected at least two transformer forwards, saw {actual_steps}"
        )
    print(
        f"requested inference steps={args.steps}; actual transformer "
        f"forwards={actual_steps}",
        flush=True,
    )

    rows = analyze_steps(profiler.events(), probe.steps)
    summary = summarize(rows)
    median_wall = summary["median_step_wall_ms"]
    if not 500.0 <= median_wall <= 10_000.0:
        raise RuntimeError(
            f"implausible median denoise-step window {median_wall:.1f} ms; "
            "expected roughly 2-4 s and require 0.5-10 s"
        )
    for row in rows:
        print(
            f"step {row['step']}/{args.steps}: "
            f"wall={row['step_wall_ms']:.1f} ms "
            f"compute={row['compute_busy_ms']:.1f} ms "
            f"copy={row['copy_busy_ms']:.1f} ms "
            f"launch_gap={row['launch_gap_ms']:.1f} ms "
            f"({row['launch_gap_pct']:.2f}%) kernels={row['kernel_count']}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "model_arch": resolved.model_arch,
        "optimizations": list(optimization_names),
        "method": {
            "profiler": "torch.profiler CPU+CUDA (CUPTI)",
            "step_boundary": (
                "transformer forward-hook record_function annotation"
            ),
            "step_wall": "CUDA annotation envelope across per-step streams",
            "compute_busy": "union of CUDA kernel intervals",
            "copy_busy": "union of CUDA memcpy intervals",
            "launch_gap": (
                "copy-adjusted idle upper bound: step wall minus union(kernel, "
                "memcpy); copy stalls excluded"
            ),
            "launch_gap_pct_aggregation": "median of per-step percentages",
            "fallback_wall": "per-step CUDA-event elapsed time",
        },
        "parameters": {
            "prompt": args.prompt,
            "first_frame": str(args.first_frame),
            "first_frame_sha256": hashlib.sha256(
                args.first_frame.read_bytes()
            ).hexdigest(),
            "steps": args.steps,
            "actual_transformer_forwards": actual_steps,
            "resolution": args.resolution,
            "frames": args.frames,
            "effective_frames": effective_video_frames(args.frames),
            "seed": args.seed,
            "warmup_steps_discarded": 1,
        },
        "environment": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "checkpoint": str(args.checkpoint),
            "store_dir": str(args.store_dir),
        },
        "full_generation_wall_seconds": elapsed_seconds,
        "steps": rows,
        "summary": summary,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / f"{args.profile}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}", flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
