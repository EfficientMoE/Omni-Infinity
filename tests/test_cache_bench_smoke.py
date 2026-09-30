# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Smokes for the cache benchmark harness.

The CPU half asserts the orchestrator's dry-run works end to end. The
GPU half runs the real c1 cell once and asserts a warm bitwise hit —
same env contract as the parity gates (OMNI_CHECKPOINT + CUDA).
"""

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_dry_run_builds_the_full_grid(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.caches.ablation",
         "--dry-run", "--results-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for name in ("baseline", "c1", "c3", "c5", "all-exact"):
        assert f"h3-dense-{name}" in proc.stdout


@pytest.mark.gpu
@pytest.mark.weights
def test_c1_cell_hits_bitwise_on_warm(tmp_path):
    if not os.environ.get("OMNI_CHECKPOINT"):
        pytest.skip("OMNI_CHECKPOINT unset")
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.caches.ablation",
         "--cell", "h3-dense-c1", "--results-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO, timeout=3600,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    warm = payload["phases"][1]
    assert warm["rms_rel"] == 0.0
    assert warm["stats"]["c1"]["hits"] >= 1
