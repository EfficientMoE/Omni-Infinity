# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Run the VDN-H3 speedup-attribution experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "results" / "vdn" / "attribution"
DEFAULT_VDN = REPO / "third_party" / "vdn-minimax-h3"
DEFAULT_CKPTS = DEFAULT_VDN / "ckpts"
SHIM = REPO / "benchmarks" / "vdn_prof_shim"


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
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
