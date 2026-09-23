# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Run the 16-point VDN-H3 architecture and optimization ablation grid.

The grid compares the upstream native inference stack with Omni-Infinity's
portable diffusers stack.  Importing this module and using ``--dry-run`` are
CPU-only operations; PyTorch is imported lazily only to compare saved latents.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shlex
import statistics
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

FRAMES = 222
SEED = 0
PROMPT = "a red ball bouncing"
CSV_FIELDS = (
    "name",
    "stack",
    "s_per_nfe",
    "peak_gib",
    "rms_rel",
    "cosine",
    "notes",
    "config_json",
)


@dataclass(frozen=True)
class Run:
    """One declarative point in the ablation grid."""

    name: str
    stack: str
    tags: tuple[str, ...]
    params: dict[str, Any]


GRID = [
    Run(
        "dense-8nfe",
        "upstream",
        ("arch",),
        {"config": "8nfe.yaml", "checkpoint": None, "backend": "flex"},
    ),
    Run(
        "hybrid-8nfe",
        "upstream",
        ("arch",),
        {
            "config": "8nfe.yaml",
            "checkpoint": "stage-dmd-step-250",
            "backend": "flex",
        },
    ),
    Run(
        "dense-turbo-lora-8nfe",
        "upstream",
        ("arch", "lora"),
        {
            "config": "8nfe.yaml",
            "checkpoint": None,
            "backend": "flex",
            "external_lora": True,
        },
    ),
    Run(
        "hybrid-50ckpt-50nfe",
        "upstream",
        ("nfe",),
        {"config": "50nfe.yaml", "checkpoint": "stage-b-step-2000"},
    ),
    Run(
        "hybrid-50ckpt-8nfe",
        "upstream",
        ("nfe", "lora"),
        {
            "config": "8nfe.yaml",
            "checkpoint": "stage-b-step-2000",
            "num_steps": 8,
        },
    ),
    Run(
        "hybrid-8nfe-tuned",
        "upstream",
        ("kernels",),
        {"config": "8nfe_tuned.yaml", "checkpoint": "stage-dmd-step-250"},
    ),
    Run(
        "hybrid-8nfe-tuned-fp8",
        "upstream",
        ("precision",),
        {
            "config": "8nfe_tuned_fp8.yaml",
            "checkpoint": "stage-dmd-step-250",
            "fp8": True,
        },
    ),
    Run(
        "hybrid-8nfe-tuned-fp8-skip4",
        "upstream",
        ("precision",),
        {
            "config": "8nfe_tuned_fp8.yaml",
            "checkpoint": "stage-dmd-step-250",
            "fp8": True,
            "fp8_skip": 4,
        },
    ),
    Run(
        "hybrid-8nfe-flex",
        "upstream",
        ("backend",),
        {
            "config": "8nfe_tuned.yaml",
            "checkpoint": "stage-dmd-step-250",
            "backend": "flex",
        },
    ),
    Run(
        "hybrid-8nfe-decomposed",
        "upstream",
        ("backend",),
        {
            "config": "8nfe_tuned.yaml",
            "checkpoint": "stage-dmd-step-250",
            "backend": "decomposed",
        },
    ),
    Run(
        "omni-resident-bf16",
        "omni",
        ("memory",),
        {"offload": True},
    ),
    Run(
        "omni-stream-bf16",
        "omni",
        ("memory",),
        {"offload": True, "block_stream": 1, "stream_text_encoder": True},
    ),
    Run(
        "omni-stream-fp8",
        "omni",
        ("memory", "precision"),
        {
            "offload": True,
            "block_stream": 1,
            "stream_text_encoder": True,
            "fp8": True,
        },
    ),
    Run(
        "omni-stream-groups4",
        "omni",
        ("memory",),
        {"offload": True, "block_stream": 4, "stream_text_encoder": True},
    ),
    Run(
        "omni-50step-bf16",
        "omni",
        ("nfe",),
        {
            "offload": True,
            "block_stream": 1,
            "stream_text_encoder": True,
            "variant": "50-step",
            "evals": 50,
        },
    ),
    Run(
        "omni-notes-textenc",
        "omni",
        ("memory",),
        {"offload": True, "block_stream": 1},
    ),
]


def _path(value: os.PathLike[str] | str) -> pathlib.Path:
    return pathlib.Path(value).expanduser().resolve()


def build_command(
    run: Run,
    results_dir: os.PathLike[str] | str,
    vdn_dir: os.PathLike[str] | str,
    ckpts: os.PathLike[str] | str,
) -> list[str]:
    """Build one subprocess argv without importing GPU libraries."""
    results = _path(results_dir)
    checkpoints = _path(ckpts)
    if run.stack == "upstream":
        out = results / f"{run.name}.mp4"
        checkpoint = run.params.get("checkpoint")
        argv = [
            "python",
            "src/inference/infer.py",
            "--config",
            f"configs/inference/{run.params['config']}",
            f"checkpoint={checkpoints / checkpoint}"
            if checkpoint
            else "checkpoint=null",
        ]
        if run.params.get("external_lora"):
            lora = checkpoints / (
                "external/minimax_h3_turbo_v4_step600_ema.safetensors"
            )
            argv.append(f"external_loras=[{{path: {lora}}}]")
        if backend := run.params.get("backend"):
            argv.append(f"kernels.softmax_backend={backend}")
        if "num_steps" in run.params:
            argv.append(f"render.num_steps={run.params['num_steps']}")
        if "fp8_skip" in run.params:
            argv.append(
                "precision.fp8.skip_end_blocks=" f"{run.params['fp8_skip']}"
            )
        argv.extend(
            [
                "render.prompt_file=prompts/example_0.pt",
                f"render.out={out}",
                f"render.num_frames={FRAMES}",
                f"render.seed={SEED}",
                "render.warmup_steps=2",
                "render.record=true",
                "render.save_latents=true",
                f"render.latents_root={results / 'latents'}",
            ]
        )
        return argv

    if run.stack != "omni":
        raise ValueError(f"unknown stack: {run.stack}")
    argv = [
        "python",
        "examples/vdn_smoke.py",
        "--prompt",
        PROMPT,
        "--checkpoint",
        str(checkpoints),
        "--variant",
        str(run.params.get("variant", "8-step")),
        "--seed",
        str(SEED),
        "--evals",
        str(run.params.get("evals", 8)),
        "--frames",
        str(FRAMES),
    ]
    if run.params.get("fp8"):
        argv.append("--fp8")
    if backend := run.params.get("backend"):
        argv.extend(["--softmax-backend", str(backend)])
    if run.params.get("offload"):
        argv.append("--offload")
    if block_stream := run.params.get("block_stream"):
        argv.extend(["--block-stream", str(block_stream)])
    if run.params.get("stream_text_encoder"):
        argv.append("--stream-text-encoder")
    repo = pathlib.Path(__file__).resolve().parents[1]
    goldens = repo / "tests/fixtures/vdn_goldens/vdn_goldens.pt"
    if not run.params.get("fp8") and goldens.exists():
        argv.extend(["--goldens", str(goldens)])
    argv.extend(["--metrics-out", str(results / f"{run.name}.metrics.json")])
    return argv


def _sample_peak_vram(
    gpu: int, stop: threading.Event, samples: list[float]
) -> None:
    while not stop.is_set():
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            try:
                samples.append(float(result.stdout.strip().splitlines()[0]))
            except (IndexError, ValueError):
                pass
        stop.wait(0.2)


def _run_subprocess(
    argv: list[str],
    cwd: pathlib.Path,
    env: dict[str, str],
    gpu: int,
    sample_vram: bool,
) -> float | None:
    stop = threading.Event()
    samples: list[float] = []
    sampler = None
    if sample_vram:
        sampler = threading.Thread(
            target=_sample_peak_vram,
            args=(gpu, stop, samples),
            daemon=True,
        )
        sampler.start()
    try:
        subprocess.run(argv, cwd=cwd, env=env, check=True)
    finally:
        stop.set()
        if sampler is not None:
            sampler.join()
    return max(samples) / 1024 if samples else None


def _upstream_latents(results_dir: pathlib.Path, name: str) -> pathlib.Path:
    return results_dir / "latents" / f"{name}.mp4.latents.pt"


def compare_latents(
    candidate: pathlib.Path, reference: pathlib.Path
) -> tuple[float, float]:
    """Return relative RMS error and cosine similarity for video latents."""
    import torch

    got = torch.load(candidate, map_location="cpu", weights_only=False)["video"]
    ref = torch.load(reference, map_location="cpu", weights_only=False)["video"]
    got = got.float().flatten()
    ref = ref.float().flatten()
    rms_rel = (((got - ref) ** 2).mean().sqrt() / (ref**2).mean().sqrt()).item()
    cosine = torch.nn.functional.cosine_similarity(
        got.unsqueeze(0), ref.unsqueeze(0)
    ).item()
    return rms_rel, cosine


def _read_upstream_metrics(record_path: pathlib.Path) -> float:
    payload = json.loads(record_path.read_text())
    step_seconds = payload["timings"]["step_seconds"]
    if not step_seconds:
        raise ValueError(f"no evaluation timings in {record_path}")
    return statistics.mean(float(value) for value in step_seconds)


def _read_omni_metrics(path: pathlib.Path) -> tuple[float, float, Any]:
    payload = json.loads(path.read_text())
    seconds = payload.get("s_per_eval")
    if seconds is None:
        seconds = payload["s_per_eval_incl_overhead"]
    return float(seconds), float(payload["peak_gib"]), payload.get("rms_rel")


def execute(
    run: Run,
    results_dir: os.PathLike[str] | str,
    vdn_dir: os.PathLike[str] | str,
    ckpts: os.PathLike[str] | str,
    gpu: int = 0,
) -> dict[str, Any]:
    """Execute one run and return a row suitable for ``results.csv``."""
    results = _path(results_dir)
    upstream = _path(vdn_dir)
    repo = pathlib.Path(__file__).resolve().parents[1]
    argv = build_command(run, results, upstream, ckpts)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cwd = upstream if run.stack == "upstream" else repo
    peak_gib = _run_subprocess(
        argv, cwd, env, gpu, sample_vram=run.stack == "upstream"
    )
    notes: list[str] = []
    rms_rel = None
    cosine = None
    if run.stack == "upstream":
        record = results / f"{run.name}.mp4.inference.json"
        seconds = _read_upstream_metrics(record)
        candidate = _upstream_latents(results, run.name)
        reference = _upstream_latents(results, "hybrid-8nfe")
        if candidate.exists() and reference.exists():
            rms_rel, cosine = compare_latents(candidate, reference)
        if run.params.get("fp8"):
            notes.append("indicative: fp8 changes the sample")
    else:
        metrics = results / f"{run.name}.metrics.json"
        seconds, peak_gib, rms_rel = _read_omni_metrics(metrics)
        notes.append("no warmup flag in vdn_smoke.py")
    return {
        "name": run.name,
        "stack": run.stack,
        "s_per_nfe": seconds,
        "peak_gib": peak_gib,
        "rms_rel": rms_rel,
        "cosine": cosine,
        "notes": "; ".join(notes),
        "config_json": json.dumps(run.params, sort_keys=True),
    }


def _read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _refresh_upstream_quality(
    rows: list[dict[str, Any]], results: pathlib.Path
) -> None:
    reference = _upstream_latents(results, "hybrid-8nfe")
    if not reference.exists():
        return
    run_by_name = {run.name: run for run in GRID}
    for row in rows:
        if row["stack"] != "upstream":
            continue
        candidate = _upstream_latents(results, str(row["name"]))
        if not candidate.exists():
            continue
        row["rms_rel"], row["cosine"] = compare_latents(candidate, reference)
        run = run_by_name[str(row["name"])]
        if run.params.get("fp8") and "indicative" not in str(row["notes"]):
            row["notes"] = "indicative: fp8 changes the sample"


def _dry_run_line(
    run: Run, argv: list[str], vdn_dir: pathlib.Path, gpu: int
) -> str:
    repo = pathlib.Path(__file__).resolve().parents[1]
    cwd = vdn_dir if run.stack == "upstream" else repo
    note = ""
    if run.stack == "omni":
        note = "; note=no warmup flag in vdn_smoke.py"
    return (
        f"{shlex.join(argv)} # cwd={cwd}; "
        f"env=CUDA_VISIBLE_DEVICES={gpu}{note}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/vdn/ablation")
    parser.add_argument("--vdn-dir", default="third_party/vdn-minimax-h3")
    parser.add_argument("--ckpts", default="ckpts/vdn")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", choices=[run.name for run in GRID])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = _path(args.results_dir)
    vdn_dir = _path(args.vdn_dir)
    runs = [run for run in GRID if args.only in (None, run.name)]
    if args.dry_run:
        for run in runs:
            argv = build_command(run, results, vdn_dir, args.ckpts)
            print(_dry_run_line(run, argv, vdn_dir, args.gpu))
        return 0

    results.mkdir(parents=True, exist_ok=True)
    csv_path = results / "results.csv"
    rows: list[dict[str, Any]] = _read_rows(csv_path)
    completed = {row["name"] for row in rows}
    for run in runs:
        if run.name in completed and not args.force:
            continue
        row = execute(run, results, vdn_dir, args.ckpts, args.gpu)
        rows = [old for old in rows if old["name"] != run.name]
        rows.append(row)
        _refresh_upstream_quality(rows, results)
        _write_rows(csv_path, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
