# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the streaming baseline client."""

import subprocess
import sys
from pathlib import Path

from benchmarks.streaming.baseline import (
    Fragment,
    compare_decoded,
    dry_run_text,
    main,
    metrics_from_timeline,
    stream_api_present,
)
from benchmarks.streaming.contract import STACKS

_REPO = Path(__file__).resolve().parents[1]


def test_dry_run_prints_protocol_and_writes_no_csv(tmp_path: Path):
    out = tmp_path / "baseline.csv"
    text = dry_run_text()
    for stack in STACKS:
        assert f"stack={stack}" in text
    assert "shape=256p/120f" in text
    assert "seed=0" in text
    assert "warmup-discard=1" in text
    assert "reps=3" in text
    assert main(["--dry-run", "--out", str(out)]) == 0
    assert not out.exists()


def test_dry_run_does_not_import_torch():
    code = (
        "import sys\n"
        "from benchmarks.streaming.baseline import main\n"
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
    assert "torch" not in proc.stdout


def test_issue_14_absent_skips_both_omni_arches(tmp_path: Path):
    out = tmp_path / "baseline.csv"
    assert stream_api_present(_REPO) is False
    assert main(["--stack", "omni-clip", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "h3-dense" in text
    assert "vdn-hybrid" in text
    assert "issue-14-absent" in text
    assert "SKIP" in text


def test_timeline_ttff_is_the_first_decoded_fragment():
    fragments = [
        Fragment(0, 50.0, 8000.0, 0.0, 0.0, 0, True),
        Fragment(1, 90.0, 400.0, 1000.0, 1005.0, 1, True),
    ]
    row = metrics_from_timeline(
        fragments=fragments,
        stall_count=0,
        job_ready_ms=100.0,
        e2e_ms=120.0,
        peak_gib=19.0,
        frames_decoded=48,
        playback_wall_ms=2000.0,
        chunk_frames=24,
        stack="omni-clip",
        arch="h3-dense",
        rep=0,
        quality_vs_job="bitwise",
    )
    assert row["ttff_ms"] == 50.0
    assert row["chunk_rtf_p50"] == 400.0 / 1000.0
    assert row["av_offset_ms"] == 5.0
    assert row["prompt_index_match"] == 1
    assert row["sustained_fps"] == 24.0
    assert row["verdict"] == "PASS"


def test_compare_decoded_is_bitwise_or_mismatch():
    frame = (b"rgb",)
    audio = b"pcm"
    assert compare_decoded((frame, audio), (frame, audio)) == "bitwise"
    assert compare_decoded((frame, audio), ((b"other",), audio)) == "mismatch"
