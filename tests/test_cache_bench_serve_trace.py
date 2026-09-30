# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for trace replay bookkeeping (HTTP mocked)."""

from benchmarks.caches.contract import FIELDS
from benchmarks.caches.serve_trace import replay, summarize
from benchmarks.caches.workload import build_trace


class _FakeClient:
    """Job API stub: instant success, latency keyed by repeat."""

    def __init__(self):
        self.calls = []

    def submit_and_wait(self, prompt: str) -> float:
        self.calls.append(prompt)
        return 10.0 if self.calls.count(prompt) > 1 else 100.0


def test_replay_emits_one_row_per_request():
    trace = build_trace(seed=0, length=8, repeat_ratio=0.25)
    rows = replay(trace, _FakeClient(), arch="h3-dense",
                  cache_config="c1", repeat_ratio=0.25)
    assert len(rows) == 8
    assert all(set(row) == set(FIELDS) for row in rows)
    hits = [row for row in rows if row["expected_hits"] == 1]
    assert len(hits) == 2
    assert all(row["e2e_ms"] == 10.0 for row in hits)


def test_summarize_reports_latency_split():
    trace = build_trace(seed=0, length=8, repeat_ratio=0.25)
    rows = replay(trace, _FakeClient(), arch="h3-dense",
                  cache_config="c1", repeat_ratio=0.25)
    summary = summarize(rows)
    assert summary["requests"] == 8
    assert summary["expected_hits"] == 2
    assert summary["repeat_p50_ms"] < summary["unique_p50_ms"]
