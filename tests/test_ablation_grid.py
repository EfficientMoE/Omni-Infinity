# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""The ablation grids are well-formed without touching a GPU."""

import json
import os
import pathlib
import subprocess
import sys

import benchmarks.ablation_vdn as ablation
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


def test_p2_grid_preserves_vdn_grid_and_maps_optimizations_exactly():
    assert len(GRID) == 16
    p2_grid = ablation.P2_GRID
    assert len(p2_grid) == 8
    assert {run.params["nfe"] for run in p2_grid} == {8, 16}
    expected = {
        "baseline": ["adaln-host-cache"],
        "compile": ["adaln-host-cache", "compile-blocks"],
        "graph": ["adaln-host-cache", "cuda-graph"],
        "compile+graph": [
            "adaln-host-cache",
            "compile-blocks",
            "cuda-graph",
        ],
    }
    for run in p2_grid:
        assert run.stack == "p2"
        assert run.tags == ("p2",)
        assert run.params["optimizations"] == expected[run.params["config"]]


def test_every_p2_run_builds_timing_and_nfe8_parity_commands():
    for run in ablation.P2_GRID:
        timing = ablation.build_p2_cell_command(
            run,
            mode="timing",
            results_dir="/tmp/p2",
        )
        assert timing[:2] == ["python", "benchmarks/ablation_vdn.py"]
        assert timing[timing.index("--p2-cell") + 1] == "timing"
        assert timing[timing.index("--steps") + 1] == str(run.params["nfe"])
        if run.params["nfe"] == 8:
            parity = ablation.build_p2_cell_command(
                run,
                mode="parity",
                results_dir="/tmp/p2",
            )
            assert parity[parity.index("--p2-cell") + 1] == "parity"


def test_p2_import_and_dry_run_are_cpu_only(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parents[1])
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.ablation_vdn; "
            "assert 'torch' not in sys.modules",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
    dry_run = subprocess.run(
        [
            sys.executable,
            "benchmarks/ablation_vdn.py",
            "--suite",
            "p2",
            "--results-dir",
            str(tmp_path),
            "--dry-run",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "CUDA_VISIBLE_DEVICES=4" in dry_run.stdout
    assert "CUDA_VISIBLE_DEVICES=0" in dry_run.stdout


def test_nvidia_driver_version_parser_uses_kernel_module_version():
    banner = (
        "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  "
        "590.48.01  Release Build"
    )
    assert ablation._parse_nvidia_driver_version(banner) == "590.48.01"


def test_collect_p2_row_reuses_existing_cell_artifacts(tmp_path):
    run = next(
        run for run in ablation.P2_GRID if run.name == "p2-baseline-8nfe"
    )
    timing = {
        "s_per_eval": 1.25,
        "peak_gib": 71.89,
        "rms_rel": None,
        "summary": {"measurement_scope": "post-first-forward median"},
        "graph_note": "not requested",
    }
    parity = {
        "rms_rel": 0.0,
        "cosine": 1.0,
        "parity": {"tier": "bitwise"},
        "graph_note": "not requested",
    }
    (tmp_path / "baseline-nfe8-timing.json").write_text(json.dumps(timing))
    (tmp_path / "baseline-nfe8-parity.json").write_text(json.dumps(parity))
    row = ablation.collect_p2_row(run, tmp_path)
    assert row["s_per_nfe"] == 1.25
    assert row["rms_rel"] == 0.0
    assert row["cosine"] == 1.0
    assert "parity=bitwise" in row["notes"]
