# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the streaming baseline measurement CLI."""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from benchmarks.streaming.baseline import (
    dry_run_text,
    main,
    measure_run,
    measure_stack,
)
from benchmarks.streaming.client import TestClientTransport
from benchmarks.streaming.contract import BASELINE_REPS, FIELDS, STACKS

_REPO = Path(__file__).resolve().parents[1]


def test_dry_run_prints_protocol_and_writes_no_csv(tmp_path: Path):
    out = tmp_path / "baseline.csv"
    text = dry_run_text()
    for stack in STACKS:
        assert f"stack={stack}" in text
    assert "shape=256p/120f" in text
    assert "vdn-shape=768p/120f" in text
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
    direct = subprocess.run(
        [
            sys.executable,
            "benchmarks/streaming/baseline.py",
            "--dry-run",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode == 0, direct.stderr
    assert "shape=256p/120f" in direct.stdout


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def test_measure_run_uses_typed_client_for_post_websocket_and_metrics():
    app = FastAPI()
    requests: list[dict] = []

    @app.post("/v1/streams", status_code=202)
    def create_stream(payload: dict):
        requests.append(payload)
        return {"stream_id": "a" * 32}

    @app.websocket("/v1/streams/{stream_id}/ws")
    async def stream(websocket: WebSocket, stream_id: str):
        assert stream_id == "a" * 32
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "init",
                "codec": 'video/mp4; codecs="avc1.42E01E,mp4a.40.2"',
                "init_b64": _b64(b"init"),
                "action_script": [
                    {"t": 0.25, "action": "forward", "instruction": "go"}
                ],
            }
        )
        for index in range(2):
            await websocket.send_json(
                {
                    "type": "chunk",
                    "index": index,
                    "pts": float(index),
                    "duration": 1.0,
                    "keyframe": True,
                    "video_b64": _b64(f"fragment-{index}".encode()),
                    "audio_b64": None,
                    "prompt": "a red ball bouncing",
                    "instruction": "go" if index == 0 else None,
                    "action": "forward" if index == 0 else None,
                    "done": index == 1,
                }
            )
        await websocket.send_json(
            {
                "type": "end",
                "artifact_url": f"/v1/jobs/{stream_id}/artifacts",
            }
        )

    fetched: list[str] = []
    with TestClient(app) as test_client:
        row = measure_run(
            transport=TestClientTransport(test_client),
            stack="omni-clip",
            arch="h3-dense",
            rep=0,
            fragment_probe=lambda _init, _fragment: (0.0, 0.020),
            artifact_fetcher=lambda url: fetched.append(url) or b"artifact",
            parity_checker=lambda init, chunks, artifact: (
                init == b"init" and len(chunks) == 2 and artifact == b"artifact"
            ),
        )

    assert requests[0]["resolution"] == "256p"
    assert requests[0]["num_frames"] == 120
    assert fetched == ["/v1/jobs/" + "a" * 32 + "/artifacts"]
    assert row["ttff_ms"] >= 0
    assert row["arrival_gap_p50"] >= 0
    assert row["arrival_gap_p95"] >= row["arrival_gap_p50"]
    assert row["chunk_produce_ms"] == ""
    assert row["chunk_rtf_p50"] == ""
    assert row["production_latency_support"] == "unsupported"
    assert (
        row["production_latency_reason"]
        == "wire protocol has no server production timestamps"
    )
    assert row["detailed_spans_support"] == "unsupported"
    assert row["av_offset_ms"] == 20.0
    assert row["cue_alignment"] == 1
    assert row["prompt_index_match"] == 1
    assert row["decoded_parity"] == "bitwise"
    assert row["quality_vs_job"] == "bitwise"
    assert row["resolution"] == "256p"
    assert row["verdict"] == "PASS"
    assert tuple(row) == FIELDS


def test_vdn_measurement_is_labelled_and_requested_at_768p():
    class RejectingTransport:
        def create_stream(self, request):
            assert request["model_arch"] == "vdn-hybrid"
            assert request["resolution"] == "768p"
            raise ConnectionError("stop after inspecting request")

    rows = measure_stack(
        "omni-clip",
        transport=RejectingTransport(),
        arches=("vdn-hybrid",),
    )
    assert len(rows) == BASELINE_REPS
    assert {row["resolution"] for row in rows} == {"768p"}
    assert {row["verdict"] for row in rows} == {"SKIP"}
    assert all("server-down" in row["notes"] for row in rows)


def test_native_stub_and_absent_external_weights_are_skip_rows():
    native = measure_stack("omni-native", transport=None)
    assert len(native) == BASELINE_REPS
    assert {row["verdict"] for row in native} == {"SKIP"}
    assert all(
        row["native_performance_support"] == "unsupported" for row in native
    )
    assert all("synthetic 16x16 stub" in row["notes"] for row in native)

    external = measure_stack("vllm-helios", transport=None, env={})
    assert len(external) == BASELINE_REPS
    assert {row["verdict"] for row in external} == {"SKIP"}
    assert all(row["notes"] == "weights-absent" for row in external)


def test_cli_server_failure_writes_compatible_skip_csv(tmp_path: Path):
    out = tmp_path / "baseline.csv"
    assert (
        main(
            [
                "--stack",
                "omni-clip",
                "--base-url",
                "http://127.0.0.1:1",
                "--request-timeout",
                "0.01",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(FIELDS)
    assert len(lines) == 1 + 2 * BASELINE_REPS
    assert all("server-down" in line and "SKIP" in line for line in lines[1:])
