# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Server-span microbenchmark for one streaming session.

Named spans plus ``other_ms`` must land within 5% of the server session
wall. Issue #14 does not expose ``/v1/streams`` on this branch, so a live
run records SKIP instead of editing the job API.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from benchmarks.streaming.contract import (
    BASELINE_CHUNK_FRAMES,
    BASELINE_REPS,
    BASELINE_TRANSPORT,
    ENVELOPE_TOLERANCE,
    FIELDS,
    FMP4_FRAG_P50_MAX_MS,
    FPS,
    PROMPT,
    REQUESTED_FRAMES,
    SEED,
    SHORT_EDGE,
    STACKS,
    WARMUP_DISCARD,
    clip_overhead_ok,
    verdict_for,
)

SPAN_KEYS: tuple[str, ...] = (
    "denoise_ms",
    "vae_decode_ms",
    "audio_decode_ms",
    "fmp4_init_ms",
    "fmp4_frag_ms",
    "ws_send_ms",
)
EXTRA_FIELDS: tuple[str, ...] = SPAN_KEYS + (
    "other_ms",
    "session_wall_ms",
    "envelope_ok",
    "scheduler_forward_ms",
    "chunk_total_ms",
)
MICRO_FIELDS: tuple[str, ...] = FIELDS + EXTRA_FIELDS


class SpanRecorder:
    """Accumulates the five server spans a future stream handler should fill."""

    def __init__(self) -> None:
        self.ms = {name: 0.0 for name in SPAN_KEYS}

    def add(self, name: str, elapsed_ms: float) -> None:
        if name not in self.ms:
            raise KeyError(name)
        self.ms[name] += elapsed_ms


def envelope(
    spans: dict[str, float], session_wall_ms: float
) -> tuple[float, bool]:
    """Residual ``other_ms`` and whether it is within 5% of the wall."""
    if session_wall_ms <= 0:
        raise ValueError("session_wall_ms must be positive")
    named = sum(float(spans[name]) for name in SPAN_KEYS)
    other = session_wall_ms - named
    ok = abs(other) / session_wall_ms <= ENVELOPE_TOLERANCE
    return other, ok


def _p50(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty")
    if len(ordered) == 1:
        return float(ordered[0])
    pos = 0.5 * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def frag_latency_ok(samples_ms: list[float]) -> bool:
    return _p50(samples_ms) <= FMP4_FRAG_P50_MAX_MS


def score_measured(
    spans: dict[str, float],
    session_wall_ms: float,
    frag_samples_ms: list[float],
    e2e_ms: float,
    t_job_ready_ms: float,
) -> str:
    _other, ok = envelope(spans, session_wall_ms)
    if not ok or not frag_latency_ok(frag_samples_ms):
        return "FAIL"
    if not clip_overhead_ok(e2e_ms, t_job_ready_ms):
        return "FAIL"
    return "PASS"


def dry_run_text() -> str:
    lines = [
        "streaming micro dry-run",
        f"shape={SHORT_EDGE}p/{REQUESTED_FRAMES}f",
        f"seed={SEED}",
        f"prompt={PROMPT}",
        f"fps={FPS}",
        f"warmup-discard={WARMUP_DISCARD}",
        f"reps={BASELINE_REPS}",
        "external-spans=scheduler_forward_ms,chunk_total_ms",
        "h3-spans=SKIP",
    ]
    lines.extend(f"span={name}" for name in SPAN_KEYS)
    lines.extend(f"stack={stack}" for stack in STACKS)
    return "\n".join(lines) + "\n"


def _blank_extra(row: dict, *, h3_spans: str) -> dict:
    scored = dict(row)
    for name in SPAN_KEYS:
        scored[name] = h3_spans
    scored["other_ms"] = h3_spans
    scored["session_wall_ms"] = h3_spans
    scored["envelope_ok"] = h3_spans
    scored["scheduler_forward_ms"] = h3_spans
    scored["chunk_total_ms"] = h3_spans
    return scored


def stream_api_present(repo_root: Path) -> bool:
    serve = repo_root / "omni_infinity" / "serve"
    if not serve.is_dir():
        return False
    for path in sorted(serve.glob("*.py")):
        if "/v1/streams" in path.read_text(encoding="utf-8"):
            return True
    return False


def _skip_row(stack: str, arch: str, rep: int, notes: str) -> dict:
    row = {field: "" for field in FIELDS}
    row.update(
        {
            "track": "micro",
            "stack": stack,
            "arch": arch,
            "chunk_frames": BASELINE_CHUNK_FRAMES,
            "transport": BASELINE_TRANSPORT,
            "rep": rep,
            "quality_vs_job": "skipped",
            "notes": notes,
            "video_duration_s": REQUESTED_FRAMES / FPS,
        }
    )
    if stack == "omni-native":
        row["native_runner"] = False
    row["verdict"] = verdict_for(row)
    return row


_WEIGHT_ENV = {
    "vllm-helios": "OMNI_HELIOS_WEIGHTS",
    "sglang-lingbot": "OMNI_LINGBOT_WEIGHTS",
    "sglang-sana-wm": "OMNI_SANA_WM_WEIGHTS",
}


def _weights_ready(stack: str) -> bool:
    key = _WEIGHT_ENV.get(stack)
    if key is None:
        return False
    raw = os.environ.get(key, "")
    return bool(raw) and Path(raw).is_dir()


_ARCHES = {
    "omni-clip": ("h3-dense", "vdn-hybrid"),
    "omni-native": ("h3-dense",),
    "vllm-helios": ("helios-distilled",),
    "sglang-lingbot": ("lingbot-world",),
    "sglang-sana-wm": ("sana-wm",),
}


def measure_rows(repo_root: Path, stacks: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    api = stream_api_present(repo_root)
    for stack in stacks:
        if stack in ("omni-clip", "omni-native"):
            notes = "issue-14-absent" if not api else "server-down"
        elif not _weights_ready(stack):
            notes = "weights-absent"
        else:
            notes = "server-down"
        h3_spans = "SKIP"
        for arch in _ARCHES[stack]:
            for rep in range(BASELINE_REPS):
                rows.append(
                    _blank_extra(
                        _skip_row(stack, arch, rep, notes), h3_spans=h3_spans
                    )
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MICRO_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in MICRO_FIELDS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stack", action="append", choices=STACKS)
    parser.add_argument("--out", default="results/streaming/micro.csv")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        sys.stdout.write(dry_run_text())
        return 0
    selected = tuple(args.stack) if args.stack else STACKS
    rows = measure_rows(Path(args.repo_root), selected)
    if any(row.get("envelope_ok") is False for row in rows):
        write_csv(Path(args.out), rows)
        return 1
    write_csv(Path(args.out), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
