# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests: ablation grid construction and cell-command building."""

import json
from types import SimpleNamespace

import torch

from benchmarks.caches.ablation import (
    GRID,
    _run_cell,
    build_cell_command,
    rows_from_cell_output,
)
from benchmarks.caches.contract import FIELDS


def test_grid_covers_every_cache_config_for_h3():
    names = [(cell.arch, cell.config.name) for cell in GRID]
    for config in ("baseline", "c1", "c3", "c5", "all-exact"):
        assert ("h3-dense", config) in names


def test_cell_command_is_a_module_invocation():
    cell = GRID[0]
    argv = build_cell_command(cell, results_dir="results/x")
    assert argv[:3] == ["python", "-m", "benchmarks.caches.ablation"]
    assert "--cell" in argv
    assert cell.name in argv


def test_c5_cell_without_calibration_is_marked_skip():
    cell = next(c for c in GRID if c.config.name == "c5")
    argv = build_cell_command(cell, results_dir="r")
    # No --c5-coefficients passed: the cell must be able to emit a SKIP
    # row rather than crash; asserted via rows_from_cell_output below.
    payload = {"cell": cell.name, "skip": "c5-uncalibrated"}
    rows = rows_from_cell_output(cell, json.dumps(payload))
    assert rows[0]["verdict"] == "SKIP"
    assert rows[0]["notes"] == "c5-uncalibrated"
    assert argv  # command still constructible


def test_c5_cell_rows_carry_step_counters_and_score():
    cell = next(c for c in GRID if c.config.name == "c5")
    payload = {
        "cell": cell.name,
        "phases": [
            {"phase": "cold", "e2e_ms": 100.0, "stats": {}},
            {
                "phase": "warm",
                "e2e_ms": 60.0,
                "rms_rel": 0.04,
                "rms_rel_max": 0.1,
                "c5_computed": 5,
                "c5_skipped": 3,
                "stats": {},
            },
        ],
    }
    rows = rows_from_cell_output(cell, json.dumps(payload))
    warm = rows[1]
    assert warm["c5_computed"] == 5 and warm["c5_skipped"] == 3
    assert warm["speedup_vs_baseline"] == 100.0 / 60.0
    assert warm["verdict"] == "PASS"


def test_rows_from_cell_output_orders_fields_and_scores():
    cell = next(c for c in GRID if c.config.name == "c1")
    payload = {
        "cell": cell.name,
        "phases": [
            {"phase": "cold", "e2e_ms": 100.0, "stats": {}},
            {
                "phase": "warm",
                "e2e_ms": 50.0,
                "rms_rel": 0.0,
                "stats": {"c1": {"hits": 1, "misses": 1}},
            },
        ],
    }
    rows = rows_from_cell_output(cell, json.dumps(payload))
    assert [row["phase"] for row in rows] == ["cold", "warm"]
    warm = rows[1]
    assert warm["c1_hits"] == 1
    assert warm["verdict"] == "PASS"
    assert set(warm) == set(FIELDS)


class _Stats:
    def stats(self):
        return {"hits": 0, "misses": 0, "entries": 0, "bytes": 0}


class _FakeRunner:
    def __init__(self, *, condition=False, vision=False):
        self.condition_cache = _Stats() if condition else None
        self.vision_cache_controller = (
            SimpleNamespace(cache=_Stats()) if vision else None
        )
        self.calls = []

    def generate(self, _prompt, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(latents=torch.zeros(1))


def _cell_args(tmp_path):
    return SimpleNamespace(
        results_dir=str(tmp_path),
        c5_coefficients=(),
        c5_threshold=0.1,
        c5_rms_rel_max=0.1,
    )


def test_c1_cell_clears_stale_disk_store_before_runner_creation(
    tmp_path, monkeypatch
):
    cell = next(c for c in GRID if c.config.name == "c1")
    store = tmp_path / f"{cell.name}-c1store"
    store.mkdir()
    stale = store / "stale.pt"
    stale.write_bytes(b"old benchmark run")
    runner = _FakeRunner(condition=True)

    def from_pretrained(_checkpoint, **kwargs):
        assert kwargs["condition_cache_dir"] == str(store)
        assert not stale.exists()
        return runner

    monkeypatch.setenv("OMNI_CHECKPOINT", "checkpoint")
    monkeypatch.setattr(
        "omni_infinity.runner.ReferenceRunner.from_pretrained", from_pretrained
    )
    _run_cell(cell, _cell_args(tmp_path))


def test_c3_cell_uses_same_image_for_cold_and_warm(tmp_path, monkeypatch):
    cell = next(c for c in GRID if c.config.name == "c3")
    runner = _FakeRunner(vision=True)
    monkeypatch.setenv("OMNI_CHECKPOINT", "checkpoint")
    monkeypatch.setattr(
        "omni_infinity.runner.ReferenceRunner.from_pretrained",
        lambda *_args, **_kwargs: runner,
    )

    _run_cell(cell, _cell_args(tmp_path))

    assert len(runner.calls) == 2
    assert runner.calls[0]["image"] is runner.calls[1]["image"]
