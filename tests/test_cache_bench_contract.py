# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests freezing the cache-benchmark metric contract."""

from benchmarks.caches.contract import (
    CACHE_CONFIGS,
    FIELDS,
    PROMPT_FIXTURE,
    verdict_for,
)


def test_field_order_is_frozen():
    assert FIELDS[:8] == (
        "suite",
        "arch",
        "cache_config",
        "dataset",
        "trace_len",
        "repeat_ratio",
        "rep",
        "phase",
    )
    assert "verdict" in FIELDS and "notes" in FIELDS
    assert len(FIELDS) == len(set(FIELDS))


def test_cache_configs_cover_each_level_and_baseline():
    names = [c.name for c in CACHE_CONFIGS]
    assert names == ["baseline", "c1", "c3", "c5", "all-exact"]


def test_weights_absent_is_skip_not_fail():
    row = {"suite": "ablation", "cache_config": "c1", "notes": "weights-absent"}
    assert verdict_for(row) == "SKIP"


def test_uncalibrated_c5_is_skip():
    row = {
        "suite": "ablation",
        "cache_config": "c5",
        "notes": "c5-uncalibrated",
    }
    assert verdict_for(row) == "SKIP"


def test_exact_cache_warm_must_be_bitwise_and_hit():
    row = {
        "suite": "ablation",
        "cache_config": "c1",
        "phase": "warm",
        "rms_rel": 0.0,
        "c1_hits": 1,
        "notes": "",
    }
    assert verdict_for(row) == "PASS"
    assert verdict_for({**row, "rms_rel": 1e-3}) == "FAIL"
    assert verdict_for({**row, "c1_hits": 0}) == "FAIL"


def test_c5_needs_skips_and_speedup_never_bitwise_claim():
    row = {
        "suite": "ablation",
        "cache_config": "c5",
        "phase": "warm",
        "c5_skipped": 2,
        "speedup_vs_baseline": 1.3,
        "rms_rel": 0.04,
        "rms_rel_max": 0.1,
        "notes": "",
    }
    assert verdict_for(row) == "PASS"
    assert verdict_for({**row, "rms_rel": 0.2}) == "FAIL"
    assert verdict_for({**row, "c5_skipped": 0}) == "FAIL"
    assert verdict_for({**row, "speedup_vs_baseline": 0.9}) == "FAIL"


def test_baseline_and_micro_rows_report():
    assert (
        verdict_for(
            {"suite": "ablation", "cache_config": "baseline", "notes": ""}
        )
        == "REPORT"
    )
    assert (
        verdict_for({"suite": "micro", "cache_config": "c1", "notes": ""})
        == "REPORT"
    )


def test_prompt_fixture_path_exists():
    assert PROMPT_FIXTURE.is_file()
