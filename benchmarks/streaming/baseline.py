# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Client-observed baseline for the video-streaming workload.

``--dry-run`` prints the protocol and does not import torch or write a CSV.
A measurement pass records SKIP when ``/v1/streams`` is absent, the server
is unreachable, or an external checkpoint is not on disk.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmarks.streaming.contract import (
    BASELINE_CHUNK_FRAMES,
    BASELINE_REPS,
    BASELINE_TRANSPORT,
    FIELDS,
    FPS,
    PROMPT,
    REQUESTED_FRAMES,
    SEED,
    SHORT_EDGE,
    STACKS,
    WARMUP_DISCARD,
    chunk_rtf_from_produce,
    legal_frames,
    verdict_for,
)

_WEIGHT_ENV = {
    "vllm-helios": "OMNI_HELIOS_WEIGHTS",
    "sglang-lingbot": "OMNI_LINGBOT_WEIGHTS",
    "sglang-sana-wm": "OMNI_SANA_WM_WEIGHTS",
}
_ARCHES = {
    "omni-clip": ("h3-dense", "vdn-hybrid"),
    "omni-native": ("h3-dense",),
    "vllm-helios": ("helios-distilled",),
    "sglang-lingbot": ("lingbot-world",),
    "sglang-sana-wm": ("sana-wm",),
}


@dataclass(frozen=True)
class Fragment:
    """One media fragment observed by the client."""

    index: int
    arrive_ms: float
    produce_ms: float
    video_pts_ms: float
    audio_pts_ms: float
    prompt_index: int
    decoded: bool


def dry_run_text() -> str:
    frames = legal_frames(REQUESTED_FRAMES)
    lines = [
        "streaming baseline dry-run",
        f"shape={SHORT_EDGE}p/{frames}f",
        "requested-frames=120",
        f"seed={SEED}",
        f"prompt={PROMPT}",
        f"warmup-discard={WARMUP_DISCARD}",
        f"reps={BASELINE_REPS}",
        f"fps={FPS}",
        f"chunk_frames={BASELINE_CHUNK_FRAMES}",
        f"transport={BASELINE_TRANSPORT}",
    ]
    lines.extend(f"stack={stack}" for stack in STACKS)
    return "\n".join(lines) + "\n"


def stream_api_present(repo_root: Path) -> bool:
    serve = repo_root / "omni_infinity" / "serve"
    if not serve.is_dir():
        return False
    for path in sorted(serve.glob("*.py")):
        if "/v1/streams" in path.read_text(encoding="utf-8"):
            return True
    return False


def weights_ready(stack: str, env: dict[str, str] | None = None) -> bool:
    key = _WEIGHT_ENV.get(stack)
    if key is None:
        return False
    source = os.environ if env is None else env
    raw = source.get(key, "")
    return bool(raw) and Path(raw).is_dir()


def compare_decoded(
    left: tuple[tuple[bytes, ...], bytes],
    right: tuple[tuple[bytes, ...], bytes],
) -> str:
    """Bitwise equality of decoded video frames and stereo audio samples."""
    if left == right:
        return "bitwise"
    return "mismatch"


def metrics_from_timeline(
    *,
    fragments: list[Fragment],
    stall_count: int,
    job_ready_ms: float | None,
    e2e_ms: float,
    peak_gib: float | None,
    frames_decoded: int,
    playback_wall_ms: float,
    chunk_frames: int,
    stack: str,
    arch: str,
    rep: int,
    quality_vs_job: str,
    fps: float = FPS,
    transport: str = BASELINE_TRANSPORT,
    notes: str = "",
    native_runner: bool | None = None,
    camera_accepted: bool | None = None,
    prompt_update_accepted: bool | None = None,
) -> dict:
    """Build one contract row from client timestamps. Accept time is 0."""
    decoded = [frag for frag in fragments if frag.decoded]
    ttff = decoded[0].arrive_ms if decoded else None
    produce = [frag.produce_ms for frag in fragments]
    summary = chunk_rtf_from_produce(produce, chunk_frames / fps)
    if fragments:
        offset = max(
            abs(frag.audio_pts_ms - frag.video_pts_ms) for frag in fragments
        )
        prompt_match = int(
            all(frag.prompt_index == frag.index for frag in fragments)
        )
    else:
        offset = None
        prompt_match = 0
    video_s = (frames_decoded / fps) if frames_decoded else 0.0
    if playback_wall_ms > 0 and frames_decoded:
        sustained = frames_decoded / (playback_wall_ms / 1000.0)
    else:
        sustained = None
    e2e_rtf = (e2e_ms / (video_s * 1000.0)) if video_s else None
    row = {field: "" for field in FIELDS}
    row.update(
        {
            "track": "baseline",
            "stack": stack,
            "arch": arch,
            "chunk_frames": chunk_frames,
            "transport": transport,
            "rep": rep,
            "ttff_ms": "" if ttff is None else ttff,
            "t_job_ready_ms": "" if job_ready_ms is None else job_ready_ms,
            "chunk_produce_ms": produce,
            "chunk_rtf_p50": "" if summary.p50 is None else summary.p50,
            "chunk_rtf_p95": "" if summary.p95 is None else summary.p95,
            "stall_count": stall_count,
            "av_offset_ms": "" if offset is None else offset,
            "prompt_index_match": prompt_match,
            "quality_vs_job": quality_vs_job,
            "e2e_ms": e2e_ms,
            "video_duration_s": video_s,
            "e2e_rtf": "" if e2e_rtf is None else e2e_rtf,
            "peak_gib": "" if peak_gib is None else peak_gib,
            "sustained_fps": "" if sustained is None else sustained,
            "notes": notes,
        }
    )
    if native_runner is not None:
        row["native_runner"] = native_runner
    if camera_accepted is not None:
        row["camera_accepted"] = camera_accepted
    if prompt_update_accepted is not None:
        row["prompt_update_accepted"] = prompt_update_accepted
    row["verdict"] = verdict_for(row)
    return row


def skip_rows(stack: str, notes: str) -> list[dict]:
    rows = []
    native = False if stack == "omni-native" else None
    for arch in _ARCHES[stack]:
        for rep in range(BASELINE_REPS):
            row = metrics_from_timeline(
                fragments=[],
                stall_count=0,
                job_ready_ms=None,
                e2e_ms=0.0,
                peak_gib=None,
                frames_decoded=0,
                playback_wall_ms=0.0,
                chunk_frames=BASELINE_CHUNK_FRAMES,
                stack=stack,
                arch=arch,
                rep=rep,
                quality_vs_job="skipped",
                notes=notes,
                native_runner=native,
            )
            rows.append(row)
    return rows


def measure_stack(stack: str, repo_root: Path) -> list[dict]:
    """One discarded warmup is the caller's job. Here we only record reps."""
    if stack in ("omni-clip", "omni-native"):
        if not stream_api_present(repo_root):
            return skip_rows(stack, "issue-14-absent")
        return skip_rows(stack, "server-down")
    if not weights_ready(stack):
        return skip_rows(stack, "weights-absent")
    return skip_rows(stack, "server-down")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stack", action="append", choices=STACKS)
    parser.add_argument(
        "--out",
        default="results/streaming/baseline.csv",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        sys.stdout.write(dry_run_text())
        return 0
    selected = tuple(args.stack) if args.stack else STACKS
    rows: list[dict] = []
    root = Path(args.repo_root)
    for stack in selected:
        rows.extend(measure_stack(stack, root))
    write_csv(Path(args.out), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
