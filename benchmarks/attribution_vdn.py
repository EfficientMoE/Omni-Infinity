# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Run the VDN-H3 speedup-attribution experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "results" / "vdn" / "attribution"
DEFAULT_VDN = REPO / "third_party" / "vdn-minimax-h3"
DEFAULT_CKPTS = DEFAULT_VDN / "ckpts"
SHIM = REPO / "benchmarks" / "vdn_prof_shim"
COMPONENT_RUNS = ("D-prof", "V0-prof", "V1-prof", "V2-prof")


@dataclass(frozen=True)
class Run:
    name: str
    config: str
    checkpoint: str | None
    frames: int
    shim: bool = True


RUNS = (
    Run(
        "V0-prof-overheadcheck-off",
        "8nfe.yaml",
        "stage-dmd-step-250",
        222,
        False,
    ),
    Run("V0-prof-overheadcheck-on", "8nfe.yaml", "stage-dmd-step-250", 222),
    Run("D-prof", "8nfe.yaml", None, 222),
    Run("V0-prof", "8nfe.yaml", "stage-dmd-step-250", 222),
    Run("V1-prof", "8nfe_tuned.yaml", "stage-dmd-step-250", 222),
    Run("V2-prof", "8nfe_tuned_fp8.yaml", "stage-dmd-step-250", 222),
    Run("D-120-prof", "8nfe.yaml", None, 120),
    Run("D-345-prof", "8nfe.yaml", None, 345),
    Run("V0-120-prof", "8nfe.yaml", "stage-dmd-step-250", 120),
    Run("V0-311-prof", "8nfe.yaml", "stage-dmd-step-250", 311),
)
GROUPS = {
    "V0-prof-overheadcheck": RUNS[:2],
    **{run.name: (run,) for run in RUNS[2:]},
}


def build_command(run: Run, results: Path, ckpts: Path) -> list[str]:
    checkpoint = ckpts / run.checkpoint if run.checkpoint else None
    return [
        "python",
        "src/inference/infer.py",
        "--config",
        f"configs/inference/{run.config}",
        f"checkpoint={checkpoint}" if checkpoint else "checkpoint=null",
        "render.prompt_file=prompts/example_0.pt",
        f"render.out={results / (run.name + '.mp4')}",
        f"render.num_frames={run.frames}",
        "render.seed=0",
        "render.warmup_steps=2",
        "render.record=true",
        "render.save_latents=false",
    ]


def _mean_step_seconds(results: Path, run: Run) -> float:
    path = results / f"{run.name}.mp4.inference.json"
    payload = json.loads(path.read_text())
    values = payload["timings"]["step_seconds"]
    return statistics.mean(float(value) for value in values)


def _summarize_steps(
    range_steps: list[dict[str, dict[str, float | int]]],
    step_seconds: list[float],
    warmups: int = 2,
) -> dict[str, Any]:
    measured = range_steps[warmups:]
    if len(measured) != len(step_seconds):
        counts = f"{len(measured)} ranges != {len(step_seconds)} timings"
        raise ValueError(counts)
    range_names = set().union(*(step.keys() for step in measured))
    summary: dict[str, Any] = {}
    for name in range_names:
        summary[f"{name}_ms"] = statistics.mean(
            float(step.get(name, {}).get("total_ms", 0.0)) for step in measured
        )
        summary[f"{name}_calls"] = statistics.mean(
            int(step.get(name, {}).get("calls", 0)) for step in measured
        )
    summary["step_s"] = statistics.mean(step_seconds)
    summary["ratios"] = [
        float(step["step_total"]["total_ms"]) / (seconds * 1000.0)
        for step, seconds in zip(measured, step_seconds)
    ]
    attention_ms = sum(
        float(summary.get(f"{name}_ms", 0.0))
        for name in ("dense_attn", "window_softmax", "linear_branch", "gates")
    )
    summary["other_ms"] = float(summary["step_total_ms"]) - attention_ms
    return summary


def _load_summary(results: Path, name: str) -> dict[str, Any]:
    ranges = json.loads((results / f"{name}.ranges.json").read_text())["nfes"]
    record = json.loads(
        (results / f"{name}.mp4.inference.json").read_text()
    )
    seconds = [float(value) for value in record["timings"]["step_seconds"]]
    return _summarize_steps(ranges, seconds)


def _report_components(results: Path) -> None:
    summaries = {name: _load_summary(results, name) for name in COMPONENT_RUNS}
    expected = {
        "D-prof": {"dense_attn"},
        "V0-prof": {"window_softmax", "linear_branch", "gates"},
        "V1-prof": {"window_softmax", "linear_branch", "gates"},
        "V2-prof": {"window_softmax", "linear_branch", "gates"},
    }
    for name, required in expected.items():
        present = {
            key.removesuffix("_ms")
            for key in summaries[name]
            if key.endswith("_ms") and summaries[name][key]
        }
        forbidden = {"dense_attn", "window_softmax", "linear_branch", "gates"}
        forbidden -= required
        if not required <= present or forbidden & present:
            raise SystemExit(
                f"{name}: expected {sorted(required)}, found {sorted(present)}"
            )
        ratios = summaries[name]["ratios"]
        if any(not 0.95 <= ratio <= 1.05 for ratio in ratios):
            raise SystemExit(f"{name}: envelope ratios outside [0.95, 1.05]")

    output = results / "components.csv"
    fields = (
        "name",
        "step_s",
        "step_total_ms",
        "envelope_ratio_min",
        "envelope_ratio_max",
        "dense_attn_ms",
        "window_softmax_ms",
        "linear_branch_ms",
        "gates_ms",
        "other_ms",
        "linear_calls",
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in COMPONENT_RUNS:
            summary = summaries[name]
            row = {
                "name": name,
                "step_s": summary["step_s"],
                "step_total_ms": summary["step_total_ms"],
                "envelope_ratio_min": min(summary["ratios"]),
                "envelope_ratio_max": max(summary["ratios"]),
                "dense_attn_ms": summary.get("dense_attn_ms", 0.0),
                "window_softmax_ms": summary.get("window_softmax_ms", 0.0),
                "linear_branch_ms": summary.get("linear_branch_ms", 0.0),
                "gates_ms": summary.get("gates_ms", 0.0),
                "other_ms": summary["other_ms"],
                "linear_calls": summary.get("linear_calls_calls", 0.0),
            }
            writer.writerow(row)
            print(
                f"{name}: step_total/step_seconds "
                f"[{min(summary['ratios']):.4f}, {max(summary['ratios']):.4f}] "
                f"components_ms dense={row['dense_attn_ms']:.1f} "
                f"window={row['window_softmax_ms']:.1f} "
                f"linear={row['linear_branch_ms']:.1f} "
                f"gates={row['gates_ms']:.1f} other={row['other_ms']:.1f}"
            )
    dense = summaries["D-prof"]
    share = 100.0 * dense["dense_attn_ms"] / dense["step_total_ms"]
    print(f"dense attention share = {share:.2f}%")
    per_layer = {
        name: summaries[name]["step_total_ms"] / 50.0 for name in COMPONENT_RUNS
    }
    print(
        "per-layer-equivalent ms (50 main blocks): "
        + ", ".join(f"{name}={value:.2f}" for name, value in per_layer.items())
    )
    print(
        "shape ratios: "
        f"dense/hybrid={per_layer['D-prof'] / per_layer['V0-prof']:.2f}x, "
        f"hybrid/tuned={per_layer['V0-prof'] / per_layer['V1-prof']:.2f}x, "
        f"tuned/fp8={per_layer['V1-prof'] / per_layer['V2-prof']:.2f}x; "
        "paper B200 dense/hybrid=1.73x, hybrid/(kernels+fp8)=1.53x"
    )
    print(f"wrote {output}")


def _gpu_is_idle(sample: str) -> bool:
    values = (int(value.strip()) for value in sample.split(","))
    memory_mib, utilization = values
    return memory_mib < 512 and utilization < 10


def _assert_idle_gpu(gpu: int) -> str:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu),
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    if not _gpu_is_idle(query):
        message = f"refusing busy GPU {gpu}: memory/utilization={query}"
        raise RuntimeError(message)
    model = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name",
            "--format=csv,noheader",
            "-i",
            str(gpu),
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    print(f"GPU idle check: {model}; memory/utilization={query}", flush=True)
    return model


def _git_sha() -> str:
    env = os.environ.copy()
    env["GIT_MASTER"] = "1"
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        env=env,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _append_gpu_runlog(
    run: Run,
    command_line: str,
    cwd: Path,
    gpu_model: str,
    elapsed: float,
    artifacts: list[Path],
    returncode: int,
) -> None:
    path = Path(
        os.environ.get("VDN_RUNLOG", REPO / "docs/attribution_vdn_runlog.md")
    )
    artifact_text = ", ".join(str(item) for item in artifacts)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n### {timestamp} — GPU {run.name}\n\n"
            f"- **Command:** `{command_line}`\n"
            f"- **CWD:** `{cwd}`\n"
            f"- **GPU index + model:** {gpu_model}\n"
            f"- **Git SHA:** `{_git_sha()}`\n"
            f"- **Wall time:** {elapsed:.3f} s\n"
            f"- **Artifacts:** {artifact_text}\n"
            f"- **Outcome:** {run.name}: exit {returncode}\n"
        )


def _execute(run: Run, args: argparse.Namespace) -> None:
    results = Path(args.results_dir).resolve()
    vdn = Path(args.vdn_dir).resolve()
    ckpts = Path(args.ckpts).resolve()
    results.mkdir(parents=True, exist_ok=True)
    record = results / f"{run.name}.mp4.inference.json"
    ranges = results / f"{run.name}.ranges.json"
    if record.exists() and (not run.shim or ranges.exists()) and not args.force:
        print(f"reuse {run.name}: {record}")
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("HF_HOME", "/mnt/raid0nvme0/leyang/.cache/huggingface")
    if run.shim:
        old_path = env.get("PYTHONPATH")
        suffix = os.pathsep + old_path if old_path else ""
        env["PYTHONPATH"] = str(SHIM) + suffix
        env["VDN_PROF_OUT"] = str(ranges)
    else:
        env.pop("VDN_PROF_OUT", None)
    command = build_command(run, results, ckpts)
    gpu_model = _assert_idle_gpu(args.gpu)
    env_display = {
        "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
        "PYTHONPATH": env.get("PYTHONPATH", ""),
        "HF_HOME": env["HF_HOME"],
    }
    if run.shim:
        env_display["VDN_PROF_OUT"] = env["VDN_PROF_OUT"]
    command_line = " ".join(
        [f"{key}={shlex.quote(value)}" for key, value in env_display.items()]
        + [shlex.join(command)]
    )
    log_path = results / f"{run.name}.log"
    print(f"run {run.name}: {command_line}", flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=vdn,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        while process.poll() is None:
            print(f"poll {run.name}: pid={process.pid} running", flush=True)
            time.sleep(30)
    elapsed = time.monotonic() - started
    artifacts = [record, log_path, results / f"{run.name}.mp4"]
    if run.shim:
        artifacts.append(ranges)
    _append_gpu_runlog(
        run,
        command_line,
        vdn,
        gpu_model,
        elapsed,
        artifacts,
        process.returncode,
    )
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    print(f"completed {run.name} in {elapsed:.1f}s; log={log_path}", flush=True)


def _check_overhead(results: Path) -> None:
    off, on = RUNS[:2]
    off_seconds = _mean_step_seconds(results, off)
    on_seconds = _mean_step_seconds(results, on)
    delta = 100.0 * (on_seconds / off_seconds - 1.0)
    print(
        "profiler overhead: "
        f"off={off_seconds:.4f}s/NFE on={on_seconds:.4f}s/NFE "
        f"delta={delta:+.2f}% (gate <3%)"
    )
    if abs(delta) >= 3.0:
        raise SystemExit("profiler overhead gate failed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--vdn-dir", default=str(DEFAULT_VDN))
    parser.add_argument("--ckpts", default=str(DEFAULT_CKPTS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--only", choices=tuple(GROUPS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-components", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.report_components:
        _report_components(Path(args.results_dir).resolve())
        return 0
    runs = GROUPS[args.only] if args.only else RUNS
    if args.dry_run:
        results = Path(args.results_dir).resolve()
        ckpts = Path(args.ckpts).resolve()
        for run in runs:
            env = f"CUDA_VISIBLE_DEVICES={args.gpu}"
            if run.shim:
                ranges = results / f"{run.name}.ranges.json"
                env += f" PYTHONPATH={SHIM} VDN_PROF_OUT={ranges}"
            print(
                f"{env} {shlex.join(build_command(run, results, ckpts))} "
                f"# cwd={Path(args.vdn_dir).resolve()}"
            )
        return 0
    for run in runs:
        _execute(run, args)
    if args.only == "V0-prof-overheadcheck":
        _check_overhead(Path(args.results_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
