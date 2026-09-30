# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests: ablation grid construction and cell-command building."""

import json
import sys

from benchmarks.caches.ablation import (
    GRID,
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
    assert argv[:3] == [sys.executable, "-m", "benchmarks.caches.ablation"]
    assert "--cell" in argv
    assert cell.name in argv


def test_c5_cell_without_calibration_is_marked_skip():
    cell = next(
        candidate for candidate in GRID if candidate.config.name == "c5"
    )
    argv = build_cell_command(cell, results_dir="r")
    payload = {"cell": cell.name, "skip": "c5-uncalibrated"}
    rows = rows_from_cell_output(cell, json.dumps(payload))
    assert rows[0]["verdict"] == "SKIP"
    assert rows[0]["notes"] == "c5-uncalibrated"
    assert argv


def test_c5_cell_rows_carry_step_counters_and_score():
    cell = next(
        candidate for candidate in GRID if candidate.config.name == "c5"
    )
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
    cell = next(
        candidate for candidate in GRID if candidate.config.name == "c1"
    )
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


def test_bad_cell_output_becomes_a_complete_failure_row():
    cell = GRID[0]
    rows = rows_from_cell_output(cell, "stray output before JSON")
    assert len(rows) == 1
    assert set(rows[0]) == set(FIELDS)
    assert rows[0]["verdict"] == "FAIL"
    assert rows[0]["notes"] == "cell-badoutput"


def test_noisy_stdout_still_yields_the_payload_rows():
    cell = GRID[0]
    payload = {
        "cell": cell.name,
        "phases": [{"phase": "cold", "e2e_ms": 100.0, "stats": {}}],
    }
    stdout = "\n".join(
        (
            "`num_frames` has to be of the form 17 * n + 5; rounding.",
            "100%|##########| 7/7 [00:20<00:00,  2.94s/it]",
            '{"not": "the payload"}',
            json.dumps(payload),
            "",
        )
    )
    rows = rows_from_cell_output(cell, stdout)
    assert rows[0]["phase"] == "cold"
    assert rows[0]["e2e_ms"] == 100.0
