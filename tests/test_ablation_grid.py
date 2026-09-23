# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""The ablation grid is well-formed without touching a GPU."""

from benchmarks.ablation_vdn import GRID, build_command


def test_grid_names_unique():
    names = [run.name for run in GRID]
    assert len(names) == len(set(names))


def test_every_run_builds_a_command():
    for run in GRID:
        argv = build_command(
            run, results_dir="/tmp/x", vdn_dir="/tmp/v", ckpts="/tmp/c"
        )
        assert argv[0] == "python"
        assert any("infer.py" in a or "vdn_smoke.py" in a for a in argv)


def test_axes_covered():
    tags = {tag for run in GRID for tag in run.tags}
    assert {
        "arch",
        "nfe",
        "kernels",
        "backend",
        "precision",
        "memory",
        "lora",
    } <= tags
