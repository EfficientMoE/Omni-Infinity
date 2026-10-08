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
    original_fields = (
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
    )
    assert FIELDS[: len(original_fields)] == original_fields
    assert FIELDS[-2:] == ("c2_hits", "c2_misses")
    assert "verdict" in FIELDS and "notes" in FIELDS
    assert len(FIELDS) == len(set(FIELDS))


def test_cache_configs_cover_each_level_and_baseline():
    names = [c.name for c in CACHE_CONFIGS]
    assert names == [
        "baseline",
        "c1",
        "c2",
        "c3",
        "c5",
        "c5-teacache",
        "c5-taylor1",
        "c5-fbcache",
        "all-exact",
    ]

    configs = {config.name: config for config in CACHE_CONFIGS}
    assert configs["c2"].encoder_cache is True
    assert configs["c5"].denoise_indicator == "raw"
    assert configs["c5-teacache"].denoise_indicator == "teacache"
    assert configs["c5-teacache"].denoise_accumulate is True
    assert configs["c5-taylor1"].denoise_approximator == "taylor1"
    assert configs["c5-fbcache"].denoise_indicator == "fbcache"


def test_weights_absent_is_skip_not_fail():
    row = {
        "suite": "ablation",
        "cache_config": "c1",
        "notes": "weights-absent",
    }
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


def test_c2_warm_must_be_bitwise_and_hit():
    row = {
        "suite": "ablation",
        "cache_config": "c2",
        "phase": "warm",
        "rms_rel": 0.0,
        "c2_hits": 1,
        "notes": "",
    }
    assert verdict_for(row) == "PASS"
    assert verdict_for({**row, "rms_rel": 1e-3}) == "FAIL"
    assert verdict_for({**row, "c2_hits": 0}) == "FAIL"


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

    for config in ("c5-teacache", "c5-taylor1", "c5-fbcache"):
        assert verdict_for({**row, "cache_config": config}) == "PASS"
        assert (
            verdict_for({**row, "cache_config": config, "c5_skipped": 0})
            == "FAIL"
        )


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
