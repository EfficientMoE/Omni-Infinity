# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""One-factor ablation of the streaming workload.

Each non-reference row changes exactly one knob from the baseline profile.
``--dry-run`` prints the grid and does not import torch or write a CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmarks.streaming.contract import (
    BASELINE_ARCH,
    BASELINE_CHUNK_FRAMES,
    BASELINE_OPTS,
    BASELINE_TRANSPORT,
    FIELDS,
    FPS,
    PEAK_GIB_MAX,
    PROMPT,
    REQUESTED_FRAMES,
    SEED,
    SHORT_EDGE,
    WARMUP_DISCARD,
    verdict_for,
)

REFERENCE_NAMES = (
    "chunk-24",
    "transport-ws",
    "arch-dense",
    "chunker-clip",
)
CHUNK_FRAMES = (24, 48, 72, 120)
VARIANCE_LIMIT = 0.03


@dataclass(frozen=True)
class AblationRun:
    name: str
    chunk_frames: int = BASELINE_CHUNK_FRAMES
    transport: str = BASELINE_TRANSPORT
    arch: str = BASELINE_ARCH
    opts: tuple[str, ...] = BASELINE_OPTS
    chunker: str = "clip"


def baseline_profile() -> AblationRun:
    return AblationRun("baseline")


def _without(opt: str) -> tuple[str, ...]:
    return tuple(item for item in BASELINE_OPTS if item != opt)


GRID: tuple[AblationRun, ...] = (
    AblationRun("chunk-24"),
    AblationRun("chunk-48", chunk_frames=48),
    AblationRun("chunk-72", chunk_frames=72),
    AblationRun("chunk-120", chunk_frames=120),
    AblationRun("transport-ws"),
    AblationRun("transport-hls", transport="hls"),
    AblationRun("arch-dense"),
    AblationRun("arch-vdn", arch="vdn-hybrid"),
    AblationRun("opt-no-block-stream", opts=_without("block-stream")),
    AblationRun("opt-fp8", opts=BASELINE_OPTS + ("fp8",)),
    AblationRun("opt-no-text-stream", opts=_without("text-encoder-stream")),
    AblationRun("chunker-clip"),
    AblationRun("chunker-native", chunker="native"),
)


def diff_keys(run: AblationRun, base: AblationRun | None = None) -> list[str]:
    pinned = baseline_profile() if base is None else base
    changed = []
    if run.chunk_frames != pinned.chunk_frames:
        changed.append("chunk_frames")
    if run.transport != pinned.transport:
        changed.append("transport")
    if run.arch != pinned.arch:
        changed.append("arch")
    if run.opts != pinned.opts:
        changed.append("opts")
    if run.chunker != pinned.chunker:
        changed.append("chunker")
    return changed


def measured_reps(first_e2e: float, second_e2e: float) -> int:
    """One planned rep, a second check, and a third when they disagree."""
    if first_e2e == 0:
        return 3
    delta = abs(first_e2e - second_e2e) / abs(first_e2e)
    if delta > VARIANCE_LIMIT:
        return 3
    return 2


def should_retry(notes: str) -> bool:
    return "OOM" not in notes


def _by_name(rows: list[dict]) -> dict[str, dict]:
    return {str(row["name"]): row for row in rows}


def _unmeasured(row: dict) -> bool:
    if row.get("verdict") == "SKIP":
        return True
    return row.get("ttff_ms") in ("", None)


def choose_chunk_frames(rows: list[dict]) -> tuple[int, str]:
    found = _by_name(rows)
    names = [f"chunk-{frames}" for frames in CHUNK_FRAMES]
    if any(name not in found or _unmeasured(found[name]) for name in names):
        return 24, "REPORT"
    ref = float(found["chunk-120"]["ttff_ms"])
    faster: list[int] = []
    for frames in (24, 48, 72):
        row = found[f"chunk-{frames}"]
        if int(row["stall_count"]) != 0:
            continue
        if row.get("quality_vs_job") != "bitwise":
            continue
        if float(row["ttff_ms"]) < ref:
            faster.append(frames)
    if not faster:
        return 24, "REPORT"
    return min(faster), "PASS"


def choose_transport(rows: list[dict]) -> tuple[str, str]:
    found = _by_name(rows)
    ws = found.get("transport-ws")
    hls = found.get("transport-hls")
    if ws is None or hls is None or _unmeasured(ws) or _unmeasured(hls):
        return "ws", "REPORT"
    if float(ws["ttff_ms"]) <= float(hls["ttff_ms"]):
        return "ws", "PASS"
    return "ws", "REPORT"


def choose_arch(rows: list[dict]) -> tuple[str, str]:
    found = _by_name(rows)
    dense = found.get("arch-dense")
    hybrid = found.get("arch-vdn")
    if dense is None or hybrid is None:
        return "h3-dense", "REPORT"
    if _unmeasured(dense) or _unmeasured(hybrid):
        return "h3-dense", "REPORT"
    if "OOM" in str(dense.get("notes", "")) or "OOM" in str(
        hybrid.get("notes", "")
    ):
        return "h3-dense", "REPORT"
    peaks_ok = (
        float(dense["peak_gib"]) <= PEAK_GIB_MAX
        and float(hybrid["peak_gib"]) <= PEAK_GIB_MAX
    )
    faster = float(hybrid["e2e_ms"]) <= float(dense["e2e_ms"])
    if peaks_ok and faster:
        return "vdn-hybrid", "PASS"
    return "h3-dense", "REPORT"


def score_text_stream(rows: list[dict]) -> str:
    found = _by_name(rows)
    base = found.get("chunk-24")
    dropped = found.get("opt-no-text-stream")
    if base is None or dropped is None:
        return "REPORT"
    if _unmeasured(base) or _unmeasured(dropped):
        return "REPORT"
    if "OOM" in str(dropped.get("notes", "")):
        return "REPORT"
    base_step = float(base["s_per_eval"])
    if base_step == 0:
        return "REPORT"
    delta = abs(float(dropped["s_per_eval"]) - base_step) / base_step
    if delta < 0.01 and float(dropped["peak_gib"]) <= PEAK_GIB_MAX:
        return "PASS"
    return "REPORT"


def score_fp8(rows: list[dict]) -> str:
    found = _by_name(rows)
    base = found.get("chunk-24")
    fp8 = found.get("opt-fp8")
    if base is None or fp8 is None or _unmeasured(base) or _unmeasured(fp8):
        return "REPORT"
    if float(fp8["e2e_ms"]) <= float(base["e2e_ms"]):
        return "PASS"
    return "REPORT"


def score_block_stream(rows: list[dict]) -> str:
    found = _by_name(rows)
    row = found.get("opt-no-block-stream")
    if row is None:
        return "REPORT"
    if "OOM" in str(row.get("notes", "")):
        return "REPORT"
    if _unmeasured(row):
        return "REPORT"
    return "PASS"


def dry_run_text() -> str:
    lines = [
        "streaming ablation dry-run",
        f"shape={SHORT_EDGE}p/{REQUESTED_FRAMES}f",
        f"seed={SEED}",
        f"prompt={PROMPT}",
        f"warmup-discard={WARMUP_DISCARD}",
        "reps=1",
    ]
    lines.extend(f"run={run.name}" for run in GRID)
    return "\n".join(lines) + "\n"


def stream_api_present(repo_root: Path) -> bool:
    serve = repo_root / "omni_infinity" / "serve"
    if not serve.is_dir():
        return False
    for path in sorted(serve.glob("*.py")):
        if "/v1/streams" in path.read_text(encoding="utf-8"):
            return True
    return False


ABLATION_FIELDS: tuple[str, ...] = FIELDS + (
    "name",
    "opts",
    "chunker",
    "s_per_eval",
)


def measure_rows(repo_root: Path) -> list[dict]:
    if stream_api_present(repo_root):
        notes = "server-down"
    else:
        notes = "issue-14-absent"
    rows = []
    for run in GRID:
        if run.chunker == "native":
            row_notes = notes + "; native-runner-absent"
            native = False
        else:
            row_notes = notes
            native = None
        row = {field: "" for field in FIELDS}
        row.update(
            {
                "track": "ablation",
                "stack": (
                    "omni-native"
                    if run.chunker == "native"
                    else "omni-clip"
                ),
                "arch": run.arch,
                "chunk_frames": run.chunk_frames,
                "transport": run.transport,
                "rep": 0,
                "quality_vs_job": "skipped",
                "notes": row_notes,
                "video_duration_s": REQUESTED_FRAMES / FPS,
                "name": run.name,
                "opts": ",".join(run.opts),
                "chunker": run.chunker,
                "s_per_eval": "",
            }
        )
        if native is not None:
            row["native_runner"] = native
        row["verdict"] = verdict_for(row)
        rows.append(row)
    return rows


def format_summary(rows: list[dict]) -> str:
    chunk, chunk_verdict = choose_chunk_frames(rows)
    transport, transport_verdict = choose_transport(rows)
    arch, arch_verdict = choose_arch(rows)
    lines = [
        f"chunk_frames={chunk} verdict={chunk_verdict}",
        f"transport={transport} verdict={transport_verdict}",
        f"arch={arch} verdict={arch_verdict}",
        f"text-encoder-stream verdict={score_text_stream(rows)}",
        f"fp8 verdict={score_fp8(rows)}",
        f"block-stream verdict={score_block_stream(rows)}",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=ABLATION_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ABLATION_FIELDS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="results/streaming/ablation.csv")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        sys.stdout.write(dry_run_text())
        return 0
    rows = measure_rows(Path(args.repo_root))
    write_csv(Path(args.out), rows)
    sys.stdout.write(format_summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
