# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for streaming stage spans."""

import subprocess
import sys
from pathlib import Path

from benchmarks.streaming.micro import (
    SPAN_KEYS,
    envelope,
    frag_latency_ok,
    main,
    score_measured,
)

_REPO = Path(__file__).resolve().parents[1]


def _spans(total: float) -> dict[str, float]:
    spans = {name: 0.0 for name in SPAN_KEYS}
    spans["denoise_ms"] = total
    return spans


def test_envelope_exact_sum_passes():
    other, ok = envelope(_spans(1000.0), 1000.0)
    assert other == 0
    assert ok


def test_envelope_1060_fails_five_percent_gate():
    other, ok = envelope(_spans(1060.0), 1000.0)
    assert other == -60
    assert ok is False


def test_fragment_and_overhead_targets():
    assert frag_latency_ok([40.0, 50.0, 60.0]) is True
    assert frag_latency_ok([40.0, 80.0]) is False
    assert (
        score_measured(_spans(1000.0), 1000.0, [40.0, 50.0], 1000.0, 1000.0)
        == "PASS"
    )
    assert (
        score_measured(_spans(1060.0), 1000.0, [10.0], 1000.0, 1000.0)
        == "FAIL"
    )
    assert (
        score_measured(_spans(1000.0), 1000.0, [10.0], 5000.0, 1000.0)
        == "FAIL"
    )


def test_dry_run_prints_spans_without_torch_or_csv(tmp_path: Path):
    out = tmp_path / "micro.csv"
    code = (
        "import sys\n"
        "from benchmarks.streaming.micro import SPAN_KEYS, main\n"
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
    for name in SPAN_KEYS:
        assert f"span={name}" in proc.stdout
    assert "256p/120f" in proc.stdout
    assert "seed=0" in proc.stdout
    assert main(["--dry-run", "--out", str(out)]) == 0
    assert not out.exists()
