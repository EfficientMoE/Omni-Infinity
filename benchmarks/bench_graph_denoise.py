#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Measure resident H3 CUDA-graph timing and golden parity.

Timing uses torch.profiler with the same step windows as Phase 0. Warmup and
capture steps are labeled separately; steady-state medians include replay
steps only. Parity mode writes evidence before returning a failed gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.profile_denoise_step import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FIRST_FRAME,
    DEFAULT_GOLDENS,
    DEFAULT_STORE,
    DenoiseStepProbe,
    analyze_steps,
    effective_video_frames,
    golden_provenance,
    latent_parity,
    parity_gate_passes,
    patched_transformer_config_lookup,
    summarize,
)
from omni_infinity.registry import resolve_profile
from omni_infinity.runner import _transformer_component

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "results" / "p2_phase2"
PHASE0_BASELINE = REPO / "results" / "p2_phase0" / "resident.json"
PROFILE_OPTIMIZATIONS = {
    "resident": ("adaln-host-cache",),
    "graph-resident": ("adaln-host-cache", "cuda-graph"),
}


def classify_graph_step(before: dict, after: dict) -> str:
    if after["capture_failures"] > before["capture_failures"]:
        return "capture-failed"
    if after["captures"] > before["captures"]:
        return "capture"
    if after["replays"] > before["replays"]:
        return "replay"
    before_warmup = before["fallback_reasons"].get("warmup_not_done", 0)
    after_warmup = after["fallback_reasons"].get("warmup_not_done", 0)
    if after_warmup > before_warmup:
        return "warmup"
    return "eager-fallback"


def replay_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    replay_rows = [row for row in rows if row["phase"] == "replay"]
    if not replay_rows:
        return {"steps": 0}

    def median(name: str) -> float:
        return statistics.median(float(row[name]) for row in replay_rows)

    return {
        "steps": len(replay_rows),
        "median_step_wall_ms": median("step_wall_ms"),
        "median_compute_busy_ms": median("compute_busy_ms"),
        "median_copy_busy_ms": median("copy_busy_ms"),
        "median_launch_gap_ms": median("launch_gap_ms"),
        "median_kernel_count": median("kernel_count"),
    }


def compare_output_trees(graph_output, eager_output) -> list[float]:
    from torch.utils._pytree import tree_flatten

    graph_values, graph_spec = tree_flatten(graph_output)
    eager_values, eager_spec = tree_flatten(eager_output)
    if graph_spec != eager_spec:
        raise ValueError("graph/eager output structures differ")
    comparisons = []
    for graph_value, eager_value in zip(
        graph_values, eager_values, strict=True
    ):
        if not isinstance(graph_value, torch.Tensor) or not isinstance(
            eager_value, torch.Tensor
        ):
            continue
        graph_float = graph_value.float()
        eager_float = eager_value.float()
        denominator = eager_float.square().mean().sqrt()
        numerator = (graph_float - eager_float).square().mean().sqrt()
        comparisons.append(float((numerator / denominator).item()))
    return comparisons


class GraphDenoiseStepProbe(DenoiseStepProbe):
    def __init__(self, transformer, total_steps: int, manager) -> None:
        super().__init__(transformer, total_steps)
        self.manager = manager
        self._before: list[dict] = []
        self.phases: list[str] = []

    def _snapshot(self) -> dict:
        if self.manager is None:
            return {
                "captures": 0,
                "replays": 0,
                "capture_failures": 0,
                "fallback_reasons": {},
            }
        return self.manager.stats_snapshot()

    def _pre(self, module, args) -> None:
        self._before.append(self._snapshot())
        super()._pre(module, args)

    def _post(self, module, args, output) -> None:
        super()._post(module, args, output)
        if self.manager is None:
            self.phases.append("eager")
        else:
            self.phases.append(
                classify_graph_step(self._before[-1], self._snapshot())
            )


def parse_driver_version(banner: str) -> str:
    matches = re.findall(r"\b\d+\.\d+(?:\.\d+)?\b", banner)
    return matches[-1] if matches else "unknown"


def graph_execution_gate_passes(profile_name: str, stats: dict | None) -> bool:
    if profile_name != "graph-resident":
        return True
    expected_fallbacks = {"warmup_not_done", "shape_bucket_miss"}
    unexpected_fallback = bool(
        stats is not None
        and any(
            count and reason not in expected_fallbacks
            for reason, count in stats["fallback_reasons"].items()
        )
    )
    return bool(
        stats is not None
        and stats["captures"] >= 1
        and stats["replays"] >= 1
        and stats["capture_failures"] == 0
        and stats["graphs"] >= 1
        and not unexpected_fallback
    )


def _driver_version() -> str:
    path = Path("/proc/driver/nvidia/version")
    if not path.is_file():
        return "unknown"
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    return parse_driver_version(first_line)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("timing", "parity"), required=True)
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_OPTIMIZATIONS), required=True
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--first-frame", type=Path, default=DEFAULT_FIRST_FRAME)
    parser.add_argument("--goldens", type=Path, default=DEFAULT_GOLDENS)
    parser.add_argument("--prompt", default="a red ball bouncing")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--resolution", default="256p")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-replays", action="store_true")
    return parser.parse_args()


def _validate_environment() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise SystemExit(
            "set CUDA_VISIBLE_DEVICES to exactly one physical GPU index"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("exactly one visible CUDA device is required")


def _load_runner(args, optimization_names):
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
    print(
        "runner loaded; pinned AdaLN "
        f"{runner.pinned_adaln_bytes / 1024**3:.2f} GiB",
        flush=True,
    )
    return resolved, runner


def _environment(args) -> dict[str, Any]:
    import diffusers

    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "cuda": torch.version.cuda,
        "driver": _driver_version(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "checkpoint": str(args.checkpoint),
        "store_dir": str(args.store_dir),
    }


def _parity_and_provenance(args, result):
    import diffusers

    golden = torch.load(args.goldens, map_location="cpu", weights_only=False)
    parity = latent_parity(result.latents, golden["latents"])
    first_frame_sha256 = hashlib.sha256(
        args.first_frame.read_bytes()
    ).hexdigest()
    provenance = golden_provenance(
        golden,
        torch_version=torch.__version__,
        diffusers_version=diffusers.__version__,
        gpu_name=torch.cuda.get_device_name(0),
        first_frame_sha256=first_frame_sha256,
        prompt=args.prompt,
        seed=args.seed,
        steps=args.steps,
        resolution=args.resolution,
        frames=args.frames,
    )
    return parity, provenance, first_frame_sha256


def main() -> int:
    args = _parse_args()
    _validate_environment()
    if args.steps < 2:
        raise SystemExit("--steps must be at least two")
    optimization_names = PROFILE_OPTIMIZATIONS[args.profile]
    resolved, runner = _load_runner(args, optimization_names)
    transformer = _transformer_component(runner.pipeline)
    manager = runner.cuda_graph_manager
    replay_comparisons = []
    if args.verify_replays:
        if manager is None:
            raise SystemExit("--verify-replays requires the graph profile")
        original_try_execute = manager.try_execute

        def verifying_try_execute(key, function, *call_args, **call_kwargs):
            before = manager.stats_snapshot()
            graph_output = original_try_execute(
                key, function, *call_args, **call_kwargs
            )
            after = manager.stats_snapshot()
            if graph_output is None:
                return None
            torch.cuda.synchronize()
            eager_output = function(*call_args, **call_kwargs)
            torch.cuda.synchronize()
            errors = compare_output_trees(graph_output, eager_output)
            phase_name = classify_graph_step(before, after)
            replay_comparisons.append(
                {"phase": phase_name, "rms_rel_per_tensor": errors}
            )
            print(
                f"graph/eager verification phase={phase_name} "
                f"rms_rel={errors}",
                flush=True,
            )
            return graph_output

        manager.try_execute = verifying_try_execute
    from PIL import Image

    first_frame = Image.open(args.first_frame).convert("RGB")
    started = time.perf_counter()
    probe = None
    profiler = None
    if args.mode == "timing":
        probe = GraphDenoiseStepProbe(transformer, args.steps, manager)
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with probe:
            with profile(activities=activities, acc_events=True) as profiler:
                result = runner.generate(
                    args.prompt,
                    image=first_frame,
                    seed=args.seed,
                    num_inference_steps=args.steps,
                    resolution=args.resolution,
                    num_frames=args.frames,
                )
    else:
        result = runner.generate(
            args.prompt,
            image=first_frame,
            seed=args.seed,
            num_inference_steps=args.steps,
            resolution=args.resolution,
            num_frames=args.frames,
            step_callback=lambda completed, total: print(
                f"denoise progress {completed}/{total}", flush=True
            ),
        )
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    parity, provenance, first_frame_sha256 = _parity_and_provenance(
        args, result
    )
    graph_stats = manager.stats_snapshot() if manager is not None else None
    graph_execution_gate_passed = graph_execution_gate_passes(
        args.profile, graph_stats
    )
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "profile": args.profile,
        "model_arch": resolved.model_arch,
        "optimizations": list(optimization_names),
        "parameters": {
            "prompt": args.prompt,
            "first_frame": str(args.first_frame),
            "first_frame_sha256": first_frame_sha256,
            "steps": args.steps,
            "resolution": args.resolution,
            "frames": args.frames,
            "effective_frames": effective_video_frames(args.frames),
            "seed": args.seed,
            "goldens": str(args.goldens),
        },
        "environment": _environment(args),
        "full_generation_wall_seconds": elapsed_seconds,
        "pinned_adaln_bytes": runner.pinned_adaln_bytes,
        "pinned_adaln_gib": runner.pinned_adaln_bytes / 1024**3,
        "graph_stats": graph_stats,
        "graph_execution_gate_passed": graph_execution_gate_passed,
        "graph_eager_output_comparisons": replay_comparisons,
        "parity": parity,
        "golden_provenance": provenance,
    }
    if args.mode == "timing":
        assert probe is not None and profiler is not None
        rows = analyze_steps(profiler.events(), probe.steps)
        for row, phase_name in zip(rows, probe.phases, strict=True):
            row["phase"] = phase_name
            print(
                f"step {row['step']}: phase={phase_name} "
                f"wall={row['step_wall_ms']:.1f} ms "
                f"compute={row['compute_busy_ms']:.1f} ms "
                f"copy={row['copy_busy_ms']:.1f} ms "
                f"gap={row['launch_gap_ms']:.1f} ms "
                f"kernels={row['kernel_count']}",
                flush=True,
            )
        payload["steps"] = rows
        payload["post_first_step_summary"] = summarize(rows)
        replay_metrics = replay_summary(rows)
        payload["replay_summary"] = replay_metrics
        if PHASE0_BASELINE.is_file():
            baseline = json.loads(PHASE0_BASELINE.read_text(encoding="utf-8"))
            payload["phase0_off_baseline_summary"] = baseline["summary"]
        output = args.results_dir / f"{args.profile}.json"
    else:
        gate_passed = provenance["comparable"] and parity_gate_passes(
            parity, require_bitwise=True
        )
        gate_passed = gate_passed and graph_execution_gate_passed
        payload["golden_gate_passed"] = gate_passed
        output = args.results_dir / "parity" / f"{args.profile}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}", flush=True)
    print(
        f"parity tier={parity['tier']} rms_rel={parity['rms_rel']:.6g}; "
        f"provenance comparable={provenance['comparable']}",
        flush=True,
    )
    if not graph_execution_gate_passed:
        print("graph execution gate failed", flush=True)
        return 1
    if (
        args.mode == "timing"
        and args.profile == "graph-resident"
        and replay_metrics["steps"] == 0
    ):
        print("timing replay gate failed", flush=True)
        return 1
    if args.mode == "parity" and not payload["golden_gate_passed"]:
        print("golden gate failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
