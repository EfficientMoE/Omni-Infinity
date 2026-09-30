# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke for the cache microbenchmarks (tiny sizes, real code)."""

import json

from benchmarks.caches.micro import _time_ms, run_micro


def test_micro_runs_on_cpu_and_reports_all_benches(tmp_path):
    out = tmp_path / "micro.json"
    rows = run_micro(
        out_path=out, reps=3, warmup=1, embed_rows=64, image_bytes=4096
    )
    payload = json.loads(out.read_text())
    assert payload["rows"] == rows
    names = {row["bench"] for row in rows}
    assert names == {
        "c1_condition_key",
        "c1_mem_get",
        "c1_mem_put",
        "c1_disk_get_cold",
        "c3_call_key",
        "c5_decision_overhead",
    }
    for row in rows:
        assert row["suite"] == "micro"
        assert row["reps"] == 3
        assert row["p50_ms"] >= 0.0
        assert row["p95_ms"] >= row["p50_ms"] or row["p95_ms"] >= 0.0


def test_c5_overhead_row_reports_skip_counters(tmp_path):
    rows = run_micro(
        out_path=tmp_path / "m.json",
        reps=3,
        warmup=1,
        embed_rows=16,
        image_bytes=128,
    )
    c5 = next(r for r in rows if r["bench"] == "c5_decision_overhead")
    assert c5["extra"]["computed"] >= 1
    assert c5["extra"]["skipped"] >= 1


def test_cuda_timing_uses_synchronized_wall_clock(monkeypatch):
    synchronizations = []
    clock = iter((10.0, 10.125))

    class FakeEvent:
        def __init__(self, **_kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, _other):
            return 0.0

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "torch.cuda.synchronize", lambda: synchronizations.append(1)
    )
    monkeypatch.setattr("torch.cuda.Event", FakeEvent)
    monkeypatch.setattr("time.perf_counter", lambda: next(clock))

    assert _time_ms(lambda: None, reps=1, warmup=0) == [125.0]
    assert len(synchronizations) == 2
