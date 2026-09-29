# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Frozen metrics and pass targets for the video-streaming workload.

Tracks append columns. They do not rename ``FIELDS``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

FIELDS: tuple[str, ...] = (
    "track",
    "stack",
    "arch",
    "chunk_frames",
    "transport",
    "rep",
    "ttff_ms",
    "t_job_ready_ms",
    "chunk_produce_ms",
    "chunk_rtf_p50",
    "chunk_rtf_p95",
    "stall_count",
    "av_offset_ms",
    "prompt_index_match",
    "quality_vs_job",
    "e2e_ms",
    "video_duration_s",
    "e2e_rtf",
    "peak_gib",
    "sustained_fps",
    "verdict",
    "notes",
)

PROMPT = "a red ball bouncing"
SEED = 0
FPS = 24
SHORT_EDGE = 256
REQUESTED_FRAMES = 120
WARMUP_DISCARD = 1
BASELINE_REPS = 3
BASELINE_CHUNK_FRAMES = 24
BASELINE_TRANSPORT = "ws"
BASELINE_ARCH = "h3-dense"
BASELINE_OPTS: tuple[str, ...] = (
    "adaln-host-cache",
    "block-stream",
    "text-encoder-stream",
)
STACKS: tuple[str, ...] = (
    "omni-clip",
    "omni-native",
    "vllm-helios",
    "sglang-lingbot",
    "sglang-sana-wm",
)
PEAK_GIB_MAX = 24.0
AV_OFFSET_MS_MAX = 40.0
HELIOS_FPS_MIN = 16.0
NATIVE_RTF_P50_MAX = 1.0
NATIVE_RTF_P95_MAX = 1.2
NATIVE_TTFF_CHUNK_FACTOR = 1.5
INPUT_SLACK_MS = 200.0
CLIP_OVERHEAD_FLOOR_MS = 2000.0
CLIP_OVERHEAD_FRACTION = 0.05
ENVELOPE_TOLERANCE = 0.05
FMP4_FRAG_P50_MAX_MS = 50.0

_H3_STRIDE = 17
_H3_OFFSET = 5


@dataclass(frozen=True)
class ChunkRtf:
    """RTF of fragments after the startup fragment."""

    rtfs: list[float]
    p50: float | None
    p95: float | None


def chunk_rtf_from_produce(
    produce_ms: list[float], chunk_duration_s: float
) -> ChunkRtf:
    """RTF for fragments ``i >= 1``. Index 0 is startup and is dropped."""
    if chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be positive")
    scale = chunk_duration_s * 1000.0
    rtfs = [ms / scale for ms in produce_ms[1:]]
    if not rtfs:
        return ChunkRtf(rtfs=[], p50=None, p95=None)
    return ChunkRtf(
        rtfs=rtfs,
        p50=_percentile(rtfs, 50),
        p95=_percentile(rtfs, 95),
    )


def legal_frames(
    requested: int, *, allowed: Callable[[int], bool] | None = None
) -> int:
    """Return ``requested`` when it is allowed, else the next ``17*n+5``."""
    if requested < 1:
        raise ValueError("requested frames must be positive")
    if allowed is None or allowed(requested):
        return requested
    n = 0
    while n < 10_000:
        candidate = _H3_STRIDE * n + _H3_OFFSET
        if candidate >= requested and allowed(candidate):
            return candidate
        n += 1
    raise ValueError(f"no legal frame count >= {requested}")


def clip_overhead_ok(e2e_ms: float, t_job_ready_ms: float) -> bool:
    limit = max(CLIP_OVERHEAD_FLOOR_MS, CLIP_OVERHEAD_FRACTION * t_job_ready_ms)
    return (e2e_ms - t_job_ready_ms) <= limit


def verdict_for(row: dict) -> str:
    """Score one session. Rules that do not apply are not failures."""
    notes = str(row.get("notes") or "")
    if _unmeasured(notes):
        return "SKIP"
    stack = row.get("stack")
    if stack == "omni-native" and row.get("native_runner") is False:
        return "SKIP"
    if stack == "omni-job":
        return "REPORT"
    if stack == "omni-clip":
        return _clip_verdict(row)
    if stack == "omni-native":
        return _native_verdict(row)
    if stack == "vllm-helios":
        return _helios_verdict(row)
    if stack in ("sglang-lingbot", "sglang-sana-wm"):
        return _sglang_verdict(row)
    return "FAIL"


def _unmeasured(notes: str) -> bool:
    return (
        "weights-absent" in notes
        or "issue-14-absent" in notes
        or notes == "api-absent"
    )


def _present(row: dict, key: str):
    if key not in row or row[key] is None or row[key] == "":
        return None
    return row[key]


def _clip_verdict(row: dict) -> str:
    ttff = _present(row, "ttff_ms")
    job = _present(row, "t_job_ready_ms")
    quality = _present(row, "quality_vs_job")
    if ttff is None or job is None or quality is None:
        return "FAIL"
    if float(ttff) > float(job) or quality != "bitwise":
        return "FAIL"
    prompt = _present(row, "prompt_index_match")
    if prompt is not None and int(prompt) != 1:
        return "FAIL"
    offset = _present(row, "av_offset_ms")
    if offset is not None and float(offset) > AV_OFFSET_MS_MAX:
        return "FAIL"
    stall = _present(row, "stall_count")
    if stall is not None and int(stall) != 0:
        return "FAIL"
    peak = _present(row, "peak_gib")
    if peak is not None and float(peak) > PEAK_GIB_MAX:
        return "FAIL"
    return "PASS"


def _native_verdict(row: dict) -> str:
    frames = _present(row, "chunk_frames")
    ttff = _present(row, "ttff_ms")
    rtf50 = _present(row, "chunk_rtf_p50")
    rtf95 = _present(row, "chunk_rtf_p95")
    stall = _present(row, "stall_count")
    if None in (frames, ttff, rtf50, rtf95, stall):
        return "FAIL"
    chunk_ms = int(frames) / FPS * 1000.0
    if float(ttff) > NATIVE_TTFF_CHUNK_FACTOR * chunk_ms:
        return "FAIL"
    if float(rtf50) > NATIVE_RTF_P50_MAX or float(rtf95) > NATIVE_RTF_P95_MAX:
        return "FAIL"
    if int(stall) != 0:
        return "FAIL"
    reaction = _present(row, "input_to_chunk_ms")
    if reaction is not None and float(reaction) > chunk_ms + INPUT_SLACK_MS:
        return "FAIL"
    return "PASS"


def _helios_verdict(row: dict) -> str:
    fps = _present(row, "sustained_fps")
    if fps is None or float(fps) < HELIOS_FPS_MIN:
        return "FAIL"
    stall = _present(row, "stall_count")
    if stall is not None and int(stall) != 0:
        return "FAIL"
    rtf50 = _present(row, "chunk_rtf_p50")
    if rtf50 is not None and float(rtf50) > NATIVE_RTF_P50_MAX:
        return "FAIL"
    return "PASS"


def _sglang_verdict(row: dict) -> str:
    stall = _present(row, "stall_count")
    if stall is not None and int(stall) != 0:
        return "FAIL"
    rtf50 = _present(row, "chunk_rtf_p50")
    if rtf50 is not None and float(rtf50) > NATIVE_RTF_P50_MAX:
        return "FAIL"
    camera = _present(row, "camera_accepted")
    if camera is False:
        return "FAIL"
    if row.get("stack") == "sglang-lingbot":
        prompt = _present(row, "prompt_update_accepted")
        if prompt is False:
            return "FAIL"
    return "PASS"


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (pct / 100.0) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight
