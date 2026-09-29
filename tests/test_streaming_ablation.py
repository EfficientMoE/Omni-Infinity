# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the one-factor streaming ablation grid."""

import subprocess
import sys
from pathlib import Path

from benchmarks.streaming.ablation import (
    GRID,
    REFERENCE_NAMES,
    choose_arch,
    choose_chunk_frames,
    choose_transport,
    diff_keys,
    dry_run_text,
    main,
    measured_reps,
    score_block_stream,
    score_fp8,
    score_text_stream,
    should_retry,
)

_REPO = Path(__file__).resolve().parents[1]
_NAMES = (
    "chunk-24",
    "chunk-48",
    "chunk-72",
    "chunk-120",
    "transport-ws",
    "transport-hls",
    "arch-dense",
    "arch-vdn",
    "opt-no-block-stream",
    "opt-fp8",
    "opt-no-text-stream",
    "chunker-clip",
    "chunker-native",
)


def _row(name: str, **overrides) -> dict:
    base = {
        "name": name,
        "verdict": "PASS",
        "ttff_ms": 100,
        "stall_count": 0,
        "quality_vs_job": "bitwise",
        "peak_gib": 20,
        "e2e_ms": 100,
        "s_per_eval": 20,
        "notes": "",
    }
    base.update(overrides)
    return base


def test_dry_run_lists_every_name(tmp_path: Path):
    text = dry_run_text()
    for name in _NAMES:
        assert f"run={name}" in text
    out = tmp_path / "ablation.csv"
    assert main(["--dry-run", "--out", str(out)]) == 0
    assert not out.exists()


def test_dry_run_does_not_import_torch():
    code = (
        "import sys\n"
        "from benchmarks.streaming.ablation import main\n"
        "assert 'torch' not in sys.modules\n"
        "raise SystemExit(main(['--dry-run']))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_each_variant_changes_one_factor():
    assert [run.name for run in GRID] == list(_NAMES)
    for run in GRID:
        count = len(diff_keys(run))
        if run.name in REFERENCE_NAMES:
            assert count == 0, run.name
        else:
            assert count == 1, run.name


def test_chunk_choice_is_the_smallest_faster_than_120():
    rows = [
        _row("chunk-24", ttff_ms=30),
        _row("chunk-48", ttff_ms=10),
        _row("chunk-72", ttff_ms=20),
        _row("chunk-120", ttff_ms=40),
    ]
    assert choose_chunk_frames(rows) == (24, "PASS")
    slower = [
        _row("chunk-24", ttff_ms=50),
        _row("chunk-48", ttff_ms=50),
        _row("chunk-72", ttff_ms=50),
        _row("chunk-120", ttff_ms=40),
    ]
    assert choose_chunk_frames(slower) == (24, "REPORT")


def test_transport_and_arch_targets():
    ok = [
        _row("transport-ws", ttff_ms=10),
        _row("transport-hls", ttff_ms=12),
        _row("arch-dense", e2e_ms=100, peak_gib=20),
        _row("arch-vdn", e2e_ms=80, peak_gib=20),
    ]
    assert choose_transport(ok) == ("ws", "PASS")
    assert choose_arch(ok) == ("vdn-hybrid", "PASS")
    missed = [
        _row("transport-ws", ttff_ms=12),
        _row("transport-hls", ttff_ms=10),
    ]
    assert choose_transport(missed) == ("ws", "REPORT")


def test_opt_targets_and_oom_is_not_retried():
    rows = [
        _row("chunk-24", e2e_ms=100, s_per_eval=20),
        _row("opt-fp8", e2e_ms=90),
        _row("opt-no-text-stream", s_per_eval=20.1, peak_gib=20),
        _row("opt-no-block-stream", notes="OOM", verdict="SKIP"),
    ]
    assert score_fp8(rows) == "PASS"
    assert score_text_stream(rows) == "PASS"
    assert score_block_stream(rows) == "REPORT"
    assert should_retry("OOM") is False
    assert measured_reps(100.0, 102.0) == 2
    assert measured_reps(100.0, 104.0) == 3


def test_unmeasured_grid_stays_on_the_baseline_choice():
    rows = [
        _row(name, verdict="SKIP", ttff_ms="", notes="issue-14-absent")
        for name in _NAMES
    ]
    assert choose_chunk_frames(rows) == (24, "REPORT")
    assert choose_transport(rows) == ("ws", "REPORT")
    assert choose_arch(rows) == ("h3-dense", "REPORT")
