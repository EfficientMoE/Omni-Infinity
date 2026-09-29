# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""One-factor ablation of the streaming workload.

Each non-reference row changes exactly one knob from the baseline profile.
``--dry-run`` prints the grid and does not import torch or write a CSV.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmarks.streaming.client import build_request, detect_native_chunker
from benchmarks.streaming.contract import (
    ARCH_COMPARISON_RESOLUTION,
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
)

REFERENCE_NAMES = (
    "chunk-24",
    "transport-ws",
    "arch-dense",
    "chunker-clip",
)
CHUNK_FRAMES = (24, 48, 72, 120)
VARIANCE_LIMIT = 0.03
_ARCH_COMMON_OPTS = tuple(
    opt for opt in BASELINE_OPTS if opt != "adaln-host-cache"
)


@dataclass(frozen=True)
class ServerProfile:
    """Settings fixed when the model-serving process starts."""

    model_arch: str
    optimizations: tuple[str, ...]
    chunk_frames: int
    stream_enabled: bool
    fallback_hls: bool

    def environment(self) -> tuple[tuple[str, str], ...]:
        return (
            ("OMNI_MODEL_ARCH", self.model_arch),
            ("OMNI_OPTIMIZATIONS", ",".join(self.optimizations)),
            ("OMNI_STREAM_ENABLED", _bool_text(self.stream_enabled)),
            ("OMNI_STREAM_CHUNK_FRAMES", str(self.chunk_frames)),
            ("OMNI_STREAM_FALLBACK_HLS", _bool_text(self.fallback_hls)),
        )

    def restart_command(self) -> tuple[str, ...]:
        assignments = tuple(
            f"{key}={value}" for key, value in self.environment()
        )
        return (
            "env",
            *assignments,
            "python",
            "-m",
            "omni_infinity.serve",
        )


@dataclass(frozen=True)
class AblationRun:
    name: str
    axis: str
    chunk_frames: int = BASELINE_CHUNK_FRAMES
    transport: str = BASELINE_TRANSPORT
    arch: str = BASELINE_ARCH
    opts: tuple[str, ...] = BASELINE_OPTS
    chunker: str = "clip"
    resolution: str = f"{SHORT_EDGE}p"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def baseline_profile() -> AblationRun:
    return axis_baseline("chunk")


def _without(opt: str) -> tuple[str, ...]:
    return tuple(item for item in BASELINE_OPTS if item != opt)


GRID: tuple[AblationRun, ...] = (
    AblationRun("chunk-24", axis="chunk"),
    AblationRun("chunk-48", axis="chunk", chunk_frames=48),
    AblationRun("chunk-72", axis="chunk", chunk_frames=72),
    AblationRun("chunk-120", axis="chunk", chunk_frames=120),
    AblationRun("transport-ws", axis="transport"),
    AblationRun("transport-hls", axis="transport", transport="hls"),
    AblationRun(
        "arch-dense",
        axis="arch",
        opts=_ARCH_COMMON_OPTS,
        resolution=ARCH_COMPARISON_RESOLUTION,
    ),
    AblationRun(
        "arch-vdn",
        axis="arch",
        arch="vdn-hybrid",
        opts=_ARCH_COMMON_OPTS,
        resolution=ARCH_COMPARISON_RESOLUTION,
    ),
    AblationRun(
        "opt-no-block-stream",
        axis="optimizations",
        opts=_without("block-stream"),
    ),
    AblationRun(
        "opt-fp8",
        axis="optimizations",
        opts=BASELINE_OPTS + ("fp8",),
    ),
    AblationRun(
        "opt-no-text-stream",
        axis="optimizations",
        opts=_without("text-encoder-stream"),
    ),
    AblationRun("chunker-clip", axis="chunker"),
    AblationRun("chunker-native", axis="chunker", chunker="native"),
)


def axis_baseline(axis: str) -> AblationRun:
    if axis == "arch":
        return AblationRun(
            "arch-dense",
            axis=axis,
            opts=_ARCH_COMMON_OPTS,
            resolution=ARCH_COMPARISON_RESOLUTION,
        )
    names = {
        "chunk": "chunk-24",
        "transport": "transport-ws",
        "optimizations": "opt-baseline",
        "chunker": "chunker-clip",
    }
    try:
        return AblationRun(names[axis], axis=axis)
    except KeyError:
        raise ValueError(f"unknown ablation axis: {axis!r}") from None


def startup_profile(run: AblationRun) -> ServerProfile:
    return ServerProfile(
        model_arch=run.arch,
        optimizations=run.opts,
        chunk_frames=run.chunk_frames,
        stream_enabled=True,
        fallback_hls=run.transport == "hls",
    )


def restart_command(run: AblationRun) -> str:
    """Return the shell command for the required fresh server process."""

    return shlex.join(startup_profile(run).restart_command())


def request_for(run: AblationRun) -> dict:
    """Build the matching wire request without encoding startup settings."""

    return build_request(
        model_arch=run.arch,
        resolution=run.resolution,
        prompt=PROMPT,
        source=run.chunker,
        optimizations=list(run.opts),
        seed=SEED,
        num_frames=REQUESTED_FRAMES,
    )


def diff_keys(run: AblationRun, base: AblationRun | None = None) -> list[str]:
    pinned = axis_baseline(run.axis) if base is None else base
    profile = startup_profile(run)
    pinned_profile = startup_profile(pinned)
    changed = []
    if profile.chunk_frames != pinned_profile.chunk_frames:
        changed.append("chunk_frames")
    if profile.fallback_hls != pinned_profile.fallback_hls:
        changed.append("fallback_hls")
    if profile.stream_enabled != pinned_profile.stream_enabled:
        changed.append("stream_enabled")
    if profile.model_arch != pinned_profile.model_arch:
        changed.append("arch")
    if profile.optimizations != pinned_profile.optimizations:
        changed.append("opts")
    if run.chunker != pinned.chunker:
        changed.append("chunker")
    if run.resolution != pinned.resolution:
        changed.append("resolution")
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
    lowered = notes.casefold()
    return "oom" not in lowered and "out of memory" not in lowered


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
    # HLS media is populated by the primary WebSocket generation. Its first
    # retrieval therefore happens after generation and is not a live TTFF
    # sample comparable with WebSocket playback.
    del rows
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
        f"dense-baseline-shape={SHORT_EDGE}p/{REQUESTED_FRAMES}f",
        (
            "architecture-comparison-shape="
            f"{ARCH_COMPARISON_RESOLUTION}/{REQUESTED_FRAMES}f"
        ),
        f"seed={SEED}",
        f"prompt={PROMPT}",
        f"warmup-discard={WARMUP_DISCARD}",
        "measured-reps=2; third-if-e2e-variability>3%",
        (
            "transport-comparison=unsupported; "
            "hls-retrieval=post-generation-only"
        ),
    ]
    for run in GRID:
        lines.append(
            f"run={run.name} axis={run.axis} "
            f"restart={restart_command(run)}"
        )
    return "\n".join(lines) + "\n"


ABLATION_FIELDS: tuple[str, ...] = FIELDS + (
    "name",
    "opts",
    "chunker",
    "s_per_eval",
    "restart_command",
    "transport_comparison_support",
)


def measure_rows(repo_root: Path) -> list[dict]:
    """Create a restart manifest; GPU collection is run one profile at a time.

    The API rejects profile mismatches, so a single live process cannot execute
    this grid. The rows intentionally remain skipped until an external runner
    restarts the server with each row's ``restart_command`` and measures the
    matching request from :func:`request_for`.
    """

    del repo_root
    rows = []
    for run in GRID:
        if run.chunker == "native":
            native_support = detect_native_chunker()
            row_notes = native_support.reason
            native = native_support.supported
        else:
            row_notes = "restart-required"
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
                "resolution": run.resolution,
                "restart_command": restart_command(run),
                "transport_comparison_support": (
                    "unsupported"
                    if run.axis == "transport"
                    else "not-applicable"
                ),
            }
        )
        if native is not None:
            row["native_runner"] = native
        row["verdict"] = "SKIP"
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
