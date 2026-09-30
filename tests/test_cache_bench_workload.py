# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the seeded prompt-trace builder (no network)."""

import json

import pytest

from benchmarks.caches.contract import PROMPT_FIXTURE
from benchmarks.caches.workload import build_trace, load_prompts


def test_fixture_has_both_pools():
    payload = json.loads(PROMPT_FIXTURE.read_text())
    assert payload["schema"] == 1
    assert len(payload["vidprom"]) >= 16
    assert len(payload["vbench"]) >= 8
    assert all(isinstance(p, str) and p for p in payload["vidprom"])


def test_load_prompts_defaults_to_fixture():
    prompts = load_prompts("fixture")
    assert len(prompts) >= 16


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        load_prompts("nope")


def test_trace_is_deterministic():
    a = build_trace(seed=0, length=32, repeat_ratio=0.5)
    b = build_trace(seed=0, length=32, repeat_ratio=0.5)
    assert a == b
    assert build_trace(seed=1, length=32, repeat_ratio=0.5) != a


def test_trace_repeat_ratio_yields_expected_hits():
    trace = build_trace(seed=0, length=32, repeat_ratio=0.5)
    assert len(trace) == 32
    repeats = sum(1 for item in trace if item.expected_hit)
    assert repeats == 16
    # A repeated item's prompt appeared earlier in the trace.
    seen = set()
    for item in trace:
        if item.expected_hit:
            assert item.prompt in seen
        seen.add(item.prompt)


def test_zero_repeat_ratio_means_all_unique():
    trace = build_trace(seed=0, length=16, repeat_ratio=0.0)
    prompts = [item.prompt for item in trace]
    assert len(set(prompts)) == 16
    assert not any(item.expected_hit for item in trace)


def test_duplicate_source_prompts_do_not_count_as_unique(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.caches.workload.load_prompts",
        lambda *_args, **_kwargs: ["x", "x"],
    )
    with pytest.raises(ValueError, match="pool has 1"):
        build_trace(seed=0, length=2, repeat_ratio=0.0)
