# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""One-factor cache-contribution ablation (issue #24 C1/C3/C5).

Orchestrator mode (default): iterate the GRID, spawn one subprocess per
cell for CUDA-state isolation (the ``benchmarks/ablation_vdn.py``
pattern), poll ``nvidia-smi`` for peak VRAM, aggregate ``results.csv``
in the frozen contract field order.

Cell mode (``--cell NAME``): build the runner through the registry,
run a cold generate then a warm generate of the identical prompt/seed,
diff the latents in-process, print one JSON document to stdout.

Cells needing weights emit ``verdict=SKIP, notes=weights-absent`` rows
when ``OMNI_CHECKPOINT`` is unset. C5 cells emit
``notes=c5-uncalibrated`` unless ``--c5-coefficients``/
``--c5-threshold`` are given (docs/caches.md: no default calibration
exists for MiniMax-H3, so none is invented here).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.caches.contract import (
    CACHE_CONFIGS,
    FIELDS,
    FRAMES,
    PROMPT,
    RESOLUTION,
    SEED,
    STEPS,
    CacheConfig,
    verdict_for,
)


@dataclass(frozen=True)
class Cell:
    """One point in the ablation grid."""

    name: str
    arch: str
    config: CacheConfig


GRID: tuple[Cell, ...] = tuple(
    Cell(f"{arch}-{config.name}", arch, config)
    for arch in ("h3-dense",)
    for config in CACHE_CONFIGS
)


def build_cell_command(
    cell: Cell, *, results_dir: str, extra: tuple[str, ...] = ()
) -> list[str]:
    return [
        "python",
        "-m",
        "benchmarks.caches.ablation",
        "--cell",
        cell.name,
        "--results-dir",
        str(results_dir),
        *extra,
    ]


def rows_from_cell_output(cell: Cell, stdout: str) -> list[dict]:
    """Contract rows (all FIELDS present) from one cell's JSON stdout."""
    payload = json.loads(stdout)
    base = {field: "" for field in FIELDS}
    base.update(
        suite="ablation",
        arch=cell.arch,
        cache_config=cell.config.name,
        dataset="fixture",
        trace_len=1,
        repeat_ratio="",
        rep=0,
    )
    if payload.get("skip"):
        row = dict(base, phase="", notes=payload["skip"])
        row["verdict"] = verdict_for(row)
        return [row]
    rows = []
    cold_ms = None
    for phase in payload["phases"]:
        row = dict(base)
        row["phase"] = phase["phase"]
        row["e2e_ms"] = phase.get("e2e_ms", "")
        row["vram_peak_gib"] = phase.get("vram_peak_gib", "")
        row["rms_rel"] = phase.get("rms_rel", "")
        row["rms_rel_max"] = phase.get("rms_rel_max", "")
        row["c5_computed"] = phase.get("c5_computed", "")
        row["c5_skipped"] = phase.get("c5_skipped", "")
        row["notes"] = phase.get("notes", "")
        stats = phase.get("stats") or {}
        for prefix in ("c1", "c3"):
            for counter in ("hits", "misses"):
                value = (stats.get(prefix) or {}).get(counter, "")
                row[f"{prefix}_{counter}"] = value
        if phase["phase"] == "cold":
            cold_ms = phase.get("e2e_ms")
        elif cold_ms and phase.get("e2e_ms"):
            row["speedup_vs_baseline"] = cold_ms / phase["e2e_ms"]
        row["verdict"] = verdict_for(row)
        rows.append(row)
    return rows


# --------------------------------------------------------------------
# Cell mode (runs inside the per-cell subprocess; imports torch lazily)
# --------------------------------------------------------------------


def _rms_rel(a, b) -> float:
    import torch

    if torch.equal(a, b):
        return 0.0
    denominator = b.float().pow(2).mean().sqrt()
    if float(denominator) == 0.0:
        return float("inf")
    return float(
        (a.float() - b.float()).pow(2).mean().sqrt() / denominator
    )


def _run_cell(cell: Cell, args) -> dict:
    if not os.environ.get("OMNI_CHECKPOINT"):
        return {"cell": cell.name, "skip": "weights-absent"}
    config = cell.config
    denoise_config = None
    if config.denoise_cache:
        if not args.c5_coefficients:
            return {"cell": cell.name, "skip": "c5-uncalibrated"}
        from omni_infinity.caches.denoise import DenoiseCacheConfig

        denoise_config = DenoiseCacheConfig(
            coefficients=tuple(args.c5_coefficients),
            threshold=args.c5_threshold,
            mode="output",
        )
    from omni_infinity.runner import ReferenceRunner, _transformer_component

    runner = ReferenceRunner.from_pretrained(
        os.environ["OMNI_CHECKPOINT"],
        workflow="fl2va",
        offload=True,
        store_dir=os.environ.get("OMNI_STORE_DIR"),
        store_components=tuple(
            filter(
                None,
                os.environ.get("OMNI_STORE_COMPONENTS", "").split(","),
            )
        )
        or None,
        adaln_host_cache=bool(os.environ.get("OMNI_STORE_DIR")),
        block_stream_blocks_per_group=1,
        stream_text_encoder=True,
        condition_cache=config.condition_cache,
        condition_cache_dir=(
            str(Path(args.results_dir) / f"{cell.name}-c1store")
            if config.condition_cache
            else None
        ),
        vision_cache=config.vision_cache,
    )
    phases = []
    reference = None
    for phase in ("cold", "warm"):
        # C5 is per-generation, not cross-request: the cold phase runs
        # dense so warm-vs-cold measures the cache's quality/speed
        # trade. The runner's own `denoise_cache=` kwarg discards the
        # yielded DenoiseCacheStats, so the cell wraps
        # `denoise_step_cache` directly to report computed/skipped.
        c5_stats = None
        wrap_c5 = denoise_config is not None and phase == "warm"
        started = time.perf_counter()
        if wrap_c5:
            from omni_infinity.caches.denoise import denoise_step_cache

            transformer = _transformer_component(
                runner.pipeline, runner.transformer_component
            )
            with denoise_step_cache(
                transformer, denoise_config, STEPS
            ) as c5_stats:
                result = runner.generate(
                    PROMPT,
                    seed=SEED,
                    num_inference_steps=STEPS,
                    resolution=RESOLUTION,
                    num_frames=FRAMES,
                )
        else:
            result = runner.generate(
                PROMPT,
                seed=SEED,
                num_inference_steps=STEPS,
                resolution=RESOLUTION,
                num_frames=FRAMES,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        entry: dict = {"phase": phase, "e2e_ms": elapsed_ms, "stats": {}}
        if runner.condition_cache is not None:
            entry["stats"]["c1"] = runner.condition_cache.stats()
        if runner.vision_cache_controller is not None:
            entry["stats"]["c3"] = runner.vision_cache_controller.cache.stats()
        if c5_stats is not None:
            entry["c5_computed"] = c5_stats.computed
            entry["c5_skipped"] = c5_stats.skipped
        if phase == "cold":
            reference = result
        else:
            entry["rms_rel"] = _rms_rel(result.latents, reference.latents)
            if denoise_config is not None:
                entry["rms_rel_max"] = args.c5_rms_rel_max
        phases.append(entry)
    return {"cell": cell.name, "phases": phases}


# --------------------------------------------------------------------
# Orchestrator mode
# --------------------------------------------------------------------


def _poll_vram(stop: threading.Event, peaks: list[float]) -> None:
    while not stop.wait(0.5):
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            mib = max(float(v) for v in output.split() if v.strip())
            peaks.append(mib / 1024.0)
        except Exception:
            return


def _orchestrate(args) -> int:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    extra: tuple[str, ...] = ()
    if args.c5_coefficients:
        extra = (
            "--c5-coefficients",
            ",".join(str(c) for c in args.c5_coefficients),
            "--c5-threshold",
            str(args.c5_threshold),
            "--c5-rms-rel-max",
            str(args.c5_rms_rel_max),
        )
    all_rows: list[dict] = []
    for cell in GRID:
        if args.only and cell.name != args.only:
            continue
        argv = build_cell_command(
            cell, results_dir=str(results_dir), extra=extra
        )
        print(f"[{cell.name}] {shlex.join(argv)}")
        if args.dry_run:
            continue
        stop = threading.Event()
        peaks: list[float] = []
        poller = threading.Thread(
            target=_poll_vram, args=(stop, peaks), daemon=True
        )
        poller.start()
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=3600
            )
        finally:
            stop.set()
            poller.join(timeout=2)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            rows = [
                dict(
                    {f: "" for f in FIELDS},
                    suite="ablation",
                    arch=cell.arch,
                    cache_config=cell.config.name,
                    verdict="FAIL",
                    notes="cell-crashed",
                )
            ]
        else:
            rows = rows_from_cell_output(cell, proc.stdout)
            if peaks:
                for row in rows:
                    row["vram_peak_gib"] = max(peaks)
        (results_dir / f"{cell.name}.json").write_text(
            proc.stdout
            if not args.dry_run and proc.returncode == 0
            else json.dumps({"stderr": proc.stderr})
        )
        all_rows.extend(rows)
    if not args.dry_run:
        with open(results_dir / "results.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {results_dir / 'results.csv'}")
    return 0


def _parse_coeffs(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/cache-bench")
    parser.add_argument("--cell", default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--c5-coefficients", type=_parse_coeffs, default=()
    )
    parser.add_argument("--c5-threshold", type=float, default=0.1)
    parser.add_argument("--c5-rms-rel-max", type=float, default=0.1)
    args = parser.parse_args()
    if args.cell:
        cell = next(c for c in GRID if c.name == args.cell)
        print(json.dumps(_run_cell(cell, args)))
        return 0
    return _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
