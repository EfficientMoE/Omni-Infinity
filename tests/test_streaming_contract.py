# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Frozen streaming-workload metric contract. CPU only."""

from benchmarks.streaming.contract import (
    FIELDS,
    chunk_rtf_from_produce,
    legal_frames,
    verdict_for,
)


def test_fields_keep_the_frozen_names():
    assert FIELDS[:6] == (
        "track",
        "stack",
        "arch",
        "chunk_frames",
        "transport",
        "rep",
    )
    assert "verdict" in FIELDS
    assert "notes" in FIELDS
    assert "chunk_rtf_p50" in FIELDS
    assert FIELDS.index("resolution") > FIELDS.index("notes")
    for metric in (
        "production_latency",
        "detailed_spans",
        "native_performance",
        "hls_ttff",
    ):
        assert f"{metric}_support" in FIELDS
        assert f"{metric}_reason" in FIELDS


def test_index_zero_produce_time_is_excluded_from_p50():
    # Index 0 is startup. Including it would pull p50 up to 3.0.
    summary = chunk_rtf_from_produce([99_999.0, 1000.0, 3000.0], 1.0)
    assert summary.rtfs == [1.0, 3.0]
    assert summary.p50 == 2.0
    assert summary.p95 > summary.p50


def test_omni_clip_ignores_a_slow_chunk_rtf_when_the_job_gate_passes():
    row = {
        "stack": "omni-clip",
        "chunk_rtf_p50": 3,
        "ttff_ms": 10,
        "t_job_ready_ms": 10,
        "quality_vs_job": "bitwise",
        "prompt_index_match": 1,
        "av_offset_ms": 40,
        "stall_count": 0,
        "peak_gib": 24,
    }
    assert verdict_for(row) == "PASS"


def test_omni_clip_fails_closed_on_the_other_targets():
    base = {
        "stack": "omni-clip",
        "chunk_rtf_p50": 0.1,
        "ttff_ms": 10,
        "t_job_ready_ms": 10,
        "quality_vs_job": "bitwise",
        "prompt_index_match": 1,
        "av_offset_ms": 40,
        "stall_count": 0,
        "peak_gib": 24,
    }
    assert verdict_for({**base, "ttff_ms": 11}) == "FAIL"
    assert verdict_for({**base, "quality_vs_job": "mismatch"}) == "FAIL"
    assert verdict_for({**base, "prompt_index_match": 0}) == "FAIL"
    assert verdict_for({**base, "av_offset_ms": 40.1}) == "FAIL"
    assert verdict_for({**base, "stall_count": 1}) == "FAIL"
    assert verdict_for({**base, "peak_gib": 24.1}) == "FAIL"


def test_omni_native_rtf_over_one_fails():
    row = {
        "stack": "omni-native",
        "native_runner": True,
        "chunk_frames": 24,
        "ttff_ms": 1000,
        "chunk_rtf_p50": 1.3,
        "chunk_rtf_p95": 1.2,
        "stall_count": 0,
        "quality_vs_job": "skipped",
    }
    assert verdict_for(row) == "FAIL"


def test_missing_native_runner_is_skip():
    assert (
        verdict_for({"stack": "omni-native", "native_runner": False}) == "SKIP"
    )


def test_legal_frames_keeps_120_or_substitutes_124():
    assert legal_frames(120) == 120
    assert legal_frames(120, allowed=lambda n: (n - 5) % 17 == 0) == 124


def test_helios_fps_gate_is_sixteen():
    passing = {
        "stack": "vllm-helios",
        "sustained_fps": 16.0,
        "stall_count": 0,
        "chunk_rtf_p50": 1.0,
    }
    failing = {**passing, "sustained_fps": 15.9}
    assert verdict_for(passing) == "PASS"
    assert verdict_for(failing) == "FAIL"


def test_unmeasured_notes_skip_before_numeric_gates():
    assert (
        verdict_for(
            {
                "stack": "omni-clip",
                "notes": "issue-14-absent",
                "quality_vs_job": "mismatch",
            }
        )
        == "SKIP"
    )
    assert (
        verdict_for({"stack": "vllm-helios", "notes": "weights-absent"})
        == "SKIP"
    )
    assert verdict_for({"stack": "omni-clip", "notes": "server-down"}) == "SKIP"


def test_sglang_rejects_only_cookbook_controls():
    lingbot = {
        "stack": "sglang-lingbot",
        "stall_count": 0,
        "chunk_rtf_p50": 1.0,
        "camera_accepted": True,
        "prompt_update_accepted": False,
    }
    sana = {
        "stack": "sglang-sana-wm",
        "stall_count": 0,
        "chunk_rtf_p50": 1.0,
        "camera_accepted": True,
        "prompt_update_accepted": False,
    }
    assert verdict_for(lingbot) == "FAIL"
    assert verdict_for(sana) == "PASS"
