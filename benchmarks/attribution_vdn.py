# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Run the VDN-H3 speedup-attribution experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    Run("V0-294-prof", "8nfe.yaml", "stage-dmd-step-250", 294),
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
    record = json.loads((results / f"{name}.mp4.inference.json").read_text())
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


def _latent_frames(requested: int) -> tuple[int, int]:
    aligned = requested
    while aligned % 17 != 5:
        aligned += 1
    latent = (aligned - 5) // 17 * 5 + 2
    return aligned, latent


def _window_density(num_frames: int) -> float:
    allowed = 0
    for query in range(num_frames):
        chunk = query // 5
        lo = max((chunk - 1) * 5, 0)
        hi = min((chunk + 2) * 5 - 1, num_frames - 1)
        if query in (0, num_frames - 1):
            keys = set(range(num_frames))
        else:
            keys = set(range(lo, hi + 1))
            keys.update((0, num_frames - 1))
        allowed += len(keys)
    return allowed / (num_frames * num_frames)


def _report_density(results: Path, frames: list[int]) -> None:
    densities: dict[int, float] = {}
    print(
        "analytic video-frame-pair density "
        "(chunk=5, radius=1, anchors=both):"
    )
    for requested in frames:
        aligned, latent = _latent_frames(requested)
        density = _window_density(latent)
        densities[requested] = density
        print(
            f"N={requested}: aligned={aligned}, latent_frames={latent}, "
            f"density={100.0 * density:.4f}%"
        )
        if requested == 345:
            delta = 100.0 * density - 3.57
            print(
                f"N=345 vs paper 3.57%: delta={delta:+.4f} percentage points; "
                "FLAGGED: the exact released frame-pair mask includes three "
                "5-frame chunks plus dense first/last anchor rows and columns, "
                "so the paper uses a different density denominator or geometry."
            )

    dense = _load_summary(results, "D-prof")["dense_attn_ms"]
    density_222 = densities.get(222, _window_density(_latent_frames(222)[1]))
    print("empirical N=222 window/dense CUDA-time proxy:")
    print(
        "run | window_ms | dense_ms | kernel_ratio | density | "
        "efficiency_factor"
    )
    for name in ("V0-prof", "V1-prof", "V2-prof"):
        window = _load_summary(results, name)["window_softmax_ms"]
        ratio = window / dense
        print(
            f"{name} | {window:.1f} | {dense:.1f} | {ratio:.4f} | "
            f"{density_222:.4f} | {ratio / density_222:.3f}"
        )


def _fit_power(
    x_values: list[float],
    y_values: list[float],
) -> tuple[float, float]:
    x_log = [math.log(value) for value in x_values]
    y_log = [math.log(value) for value in y_values]
    x_mean = statistics.mean(x_log)
    y_mean = statistics.mean(y_log)
    denominator = sum((value - x_mean) ** 2 for value in x_log)
    slope = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_log, y_log)
        )
        / denominator
    )
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * value for value in x_log]
    residual = sum(
        (actual - estimate) ** 2 for actual, estimate in zip(y_log, predicted)
    )
    total = sum((actual - y_mean) ** 2 for actual in y_log)
    r_squared = 1.0 - residual / total if total else 1.0
    return slope, r_squared


def _token_count(requested_frames: int) -> tuple[int, int]:
    aligned, latent = _latent_frames(requested_frames)
    video_tokens = latent * 24 * 42
    audio_tokens = 2 * round(aligned / 24 * 40)
    return aligned, 800 + video_tokens + audio_tokens


def _oom_line(results: Path, name: str) -> str:
    log = results / f"{name}.log"
    for line in reversed(log.read_text().splitlines()):
        if "CUDA out of memory" in line:
            return line.strip()
    raise ValueError(f"{name}: OOM log has no allocator line")


def _fit_linear(
    x_values: list[float],
    y_values: list[float],
) -> tuple[float, float]:
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / sum((value - x_mean) ** 2 for value in x_values)
    return slope, y_mean - slope * x_mean


def _report_scaling(results: Path) -> None:
    measured = (
        ("D-120-prof", "D", 120),
        ("D-prof", "D", 222),
        ("D-345-prof", "D", 345),
        ("V0-120-prof", "V0", 120),
        ("V0-prof", "V0", 222),
        ("V2-prof", "V2", 222),
    )
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for name, architecture, frames in measured:
        summary = _load_summary(results, name)
        summaries[name] = summary
        aligned, tokens = _token_count(frames)
        rows.append(
            {
                "run": name,
                "arch": architecture,
                "frames": frames,
                "aligned_frames": aligned,
                "token_count": tokens,
                "status": "OK",
                "step_s": summary["step_s"],
                "dense_attn_ms": summary.get("dense_attn_ms", 0.0),
                "window_softmax_ms": summary.get("window_softmax_ms", 0.0),
                "linear_branch_ms": summary.get("linear_branch_ms", 0.0),
                "gates_ms": summary.get("gates_ms", 0.0),
                "other_ms": summary["other_ms"],
                "notes": "",
            }
        )
    for name, frames in (("V0-294-prof", 294), ("V0-311-prof", 311)):
        aligned, tokens = _token_count(frames)
        rows.append(
            {
                "run": name,
                "arch": "V0",
                "frames": frames,
                "aligned_frames": aligned,
                "token_count": tokens,
                "status": "OOM",
                "step_s": "",
                "dense_attn_ms": "",
                "window_softmax_ms": "",
                "linear_branch_ms": "",
                "gates_ms": "",
                "other_ms": "",
                "notes": _oom_line(results, name),
            }
        )

    csv_path = results / "scaling.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dense_rows = [row for row in rows if row["arch"] == "D"]
    hybrid_rows = [
        row for row in rows if row["arch"] == "V0" and row["status"] == "OK"
    ]
    dense_exponent, dense_r2 = _fit_power(
        [float(row["token_count"]) for row in dense_rows],
        [float(row["step_s"]) for row in dense_rows],
    )
    hybrid_exponent, hybrid_r2 = _fit_power(
        [float(row["token_count"]) for row in hybrid_rows],
        [float(row["step_s"]) for row in hybrid_rows],
    )
    linear_exponent, linear_r2 = _fit_power(
        [float(row["token_count"]) for row in hybrid_rows],
        [float(row["linear_branch_ms"]) for row in hybrid_rows],
    )
    print(f"dense exponent={dense_exponent:.4f}, R²={dense_r2:.4f}")
    print(
        f"hybrid exponent={hybrid_exponent:.4f}, R²={hybrid_r2:.4f} "
        "(two measured points; 294/311 OOM)"
    )
    print(f"linear-branch exponent={linear_exponent:.4f}, R²={linear_r2:.4f}")

    token_x = [float(row["token_count"]) for row in hybrid_rows]
    time_y = [float(row["step_s"]) for row in hybrid_rows]
    slope, intercept = _fit_linear(token_x, time_y)
    _, tokens_345 = _token_count(345)
    predicted_345 = slope * tokens_345 + intercept
    dense_345 = summaries["D-345-prof"]["step_s"]
    table = [
        "| N | dense s/NFE | V0 s/NFE | speedup | status |",
        "|---:|---:|---:|---:|---|",
    ]
    for dense_name, hybrid_name, frames in (
        ("D-120-prof", "V0-120-prof", 120),
        ("D-prof", "V0-prof", 222),
    ):
        dense_time = summaries[dense_name]["step_s"]
        hybrid_time = summaries[hybrid_name]["step_s"]
        table.append(
            f"| {frames} | {dense_time:.4f} | {hybrid_time:.4f} | "
            f"{dense_time / hybrid_time:.3f}× | MEASURED |"
        )
    table.extend(
        [
            "| 294 | — | OOM | — | MEASURED OOM BOUND |",
            "| 311 | — | OOM | — | MEASURED OOM BOUND |",
            f"| 345 | {dense_345:.4f} | {predicted_345:.4f} | "
            f"{dense_345 / predicted_345:.3f}× | "
            "EXTRAPOLATED/OOM-BOUNDED |",
        ]
    )
    markdown = results / "speedup_vs_N.md"
    markdown.write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"wrote {csv_path} and {markdown}")


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
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--frames", nargs="+", type=int, default=[222, 345])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.report_components:
        _report_components(Path(args.results_dir).resolve())
        return 0
    if args.density:
        _report_density(Path(args.results_dir).resolve(), args.frames)
        return 0
    if args.scaling:
        _report_scaling(Path(args.results_dir).resolve())
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
