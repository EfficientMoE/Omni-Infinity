# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Metrics and pass targets for the issue #24 cache benchmarks.

The field order is frozen (the ``benchmarks/streaming/contract.py``
rule): shared integrations may append fields but must not rename or
reorder existing ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FIELDS: tuple[str, ...] = (
    "suite",
    "arch",
    "cache_config",
    "dataset",
    "trace_len",
    "repeat_ratio",
    "rep",
    "phase",
    "e2e_ms",
    "vram_peak_gib",
    "c1_hits",
    "c1_misses",
    "c3_hits",
    "c3_misses",
    "c5_computed",
    "c5_skipped",
    "expected_hits",
    "rms_rel",
    "rms_rel_max",
    "speedup_vs_baseline",
    "verdict",
    "notes",
    "c2_hits",
    "c2_misses",
)

PROMPT = "a red ball bouncing"
SEED = 0
STEPS = 8
RESOLUTION = "256p"
FRAMES = 120
WARMUP = 2
MICRO_REPS = 20
ABLATION_REPS = 1
PEAK_GIB_MAX = 24.0

PROMPT_FIXTURE = Path(__file__).parent / "fixtures" / "prompts.json"


@dataclass(frozen=True)
class CacheConfig:
    """One ablation cell: which opt-in caches are enabled."""

    name: str
    condition_cache: bool = False
    encoder_cache: bool = False
    vision_cache: bool = False
    denoise_cache: bool = False
    denoise_indicator: str = "raw"
    denoise_accumulate: bool = False
    denoise_approximator: str = "reuse"


CACHE_CONFIGS: tuple[CacheConfig, ...] = (
    CacheConfig("baseline"),
    CacheConfig("c1", condition_cache=True),
    CacheConfig("c2", encoder_cache=True),
    CacheConfig("c3", vision_cache=True),
    CacheConfig("c5", denoise_cache=True),
    CacheConfig(
        "c5-teacache",
        denoise_cache=True,
        denoise_indicator="teacache",
        denoise_accumulate=True,
    ),
    CacheConfig(
        "c5-taylor1",
        denoise_cache=True,
        denoise_indicator="teacache",
        denoise_accumulate=True,
        denoise_approximator="taylor1",
    ),
    CacheConfig(
        "c5-fbcache",
        denoise_cache=True,
        denoise_indicator="fbcache",
    ),
    CacheConfig("all-exact", condition_cache=True, vision_cache=True),
)

_SKIP_NOTES = (
    "weights-absent",
    "c5-uncalibrated",
    "c5-signal-mismatch",
    "server-down",
)


def _present(row: dict, key: str):
    if key not in row or row[key] is None or row[key] == "":
        return None
    return row[key]


def verdict_for(row: dict) -> str:
    """Score one row. Rules that do not apply are not failures."""
    notes = str(row.get("notes") or "")
    if any(marker in notes for marker in _SKIP_NOTES):
        return "SKIP"
    suite = row.get("suite")
    config = row.get("cache_config")
    if suite in ("micro", "serve-trace") or config == "baseline":
        return "REPORT"
    if config in ("c1", "c2", "c3", "all-exact"):
        return _exact_verdict(row, config)
    if config in ("c5", "c5-teacache", "c5-taylor1", "c5-fbcache"):
        return _c5_verdict(row)
    return "REPORT"


def _exact_verdict(row: dict, config: str) -> str:
    """Require exact caches to replay bitwise and hit when warm."""
    if row.get("phase") != "warm":
        return "REPORT"
    rms = _present(row, "rms_rel")
    if rms is None or float(rms) != 0.0:
        return "FAIL"
    hit_keys = {
        "c1": ("c1_hits",),
        "c2": ("c2_hits",),
        "c3": ("c3_hits",),
        "all-exact": ("c1_hits",),
    }[config]
    for key in hit_keys:
        hits = _present(row, key)
        if hits is None or int(hits) < 1:
            return "FAIL"
    return "PASS"


def _c5_verdict(row: dict) -> str:
    """Require C5 to skip steps, accelerate, and meet its quality bound."""
    if row.get("phase") != "warm":
        return "REPORT"
    skipped = _present(row, "c5_skipped")
    speedup = _present(row, "speedup_vs_baseline")
    rms = _present(row, "rms_rel")
    bound = _present(row, "rms_rel_max")
    if None in (skipped, speedup, rms, bound):
        return "FAIL"
    if int(skipped) < 1:
        return "FAIL"
    if float(speedup) <= 1.0:
        return "FAIL"
    if float(rms) > float(bound):
        return "FAIL"
    return "PASS"
