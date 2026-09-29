# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the streaming microbenchmark timing contract."""

import base64
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from benchmarks.streaming.client import (
    AcceptedStream,
    TestClientTransport,
    TimedEvent,
    parse_event,
)
from benchmarks.streaming.micro import (
    PHASE_KEYS,
    SPAN_KEYS,
    envelope,
    main,
    measure_session,
)

_REPO = Path(__file__).resolve().parents[1]


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _init() -> dict:
    return {
        "type": "init",
        "codec": 'video/mp4; codecs="avc1.42E01E,mp4a.40.2"',
        "init_b64": _b64(b"init"),
        "action_script": [],
    }


def _chunk(index: int, *, done: bool) -> dict:
    return {
        "type": "chunk",
        "index": index,
        "pts": float(index),
        "duration": 1.0,
        "keyframe": True,
        "video_b64": _b64(f"fragment-{index}".encode()),
        "audio_b64": None,
        "prompt": "a red ball bouncing",
        "instruction": None,
        "action": None,
        "done": done,
    }


class _FakeTransport:
    def create_stream(self, request):
        assert request["source"] == "clip"
        return AcceptedStream("a" * 32, 1_000_000_000)

    def events(self, stream_id):
        assert stream_id == "a" * 32
        payloads = (
            (_init(), 1_100_000_000),
            (_chunk(0, done=False), 1_300_000_000),
            (_chunk(1, done=True), 2_300_000_000),
            (
                {
                    "type": "end",
                    "artifact_url": f"/v1/jobs/{stream_id}/artifacts",
                },
                2_600_000_000,
            ),
        )
        for payload, received_ns in payloads:
            yield TimedEvent(parse_event(payload), received_ns)


def test_phase_envelope_uses_only_instrumented_top_level_spans():
    exact = {name: 250.0 for name in PHASE_KEYS}
    other, ok = envelope(exact, 1000.0)
    assert other == 0
    assert ok is True

    other, ok = envelope({**exact, PHASE_KEYS[0]: 310.0}, 1000.0)
    assert other == -60
    assert ok is False


def test_fake_transport_labels_arrivals_and_unsupported_production_spans():
    clock = iter((2_700_000_000, 2_725_000_000))
    fetched = []

    row = measure_session(
        _FakeTransport(),
        stack="omni-clip",
        arch="h3-dense",
        rep=0,
        artifact_fetch=lambda url: fetched.append(url) or b"artifact",
        fragment_probe=lambda _init_bytes, _fragment: (0.0, 0.0),
        clock_ns=lambda: next(clock),
    )

    assert row["accept_to_init_ms"] == 100.0
    assert row["init_to_first_chunk_ms"] == 200.0
    assert row["first_to_last_chunk_arrival_ms"] == 1000.0
    assert row["last_chunk_to_end_ms"] == 300.0
    assert row["session_wall_ms"] == 1600.0
    assert row["artifact_retrieval_ms"] == 25.0
    assert row["arrival_gap_ms"] == "[1000.0]"
    assert row["chunk_produce_ms"] == ""
    assert row["production_latency_support"] == "unsupported"
    assert row["hls_ttff_support"] == "unsupported"
    assert row["envelope_ok"] is True
    assert row["other_ms"] == 0.0
    assert all(row[name] == "UNSUPPORTED" for name in SPAN_KEYS)
    assert fetched == ["/v1/jobs/" + "a" * 32 + "/artifacts"]


def test_testclient_transport_measures_real_routes_without_telemetry():
    app = FastAPI()
    artifact_requests = []

    @app.post("/v1/streams", status_code=202)
    def create_stream(payload: dict):
        assert payload["prompt"] == "a red ball bouncing"
        return {"stream_id": "b" * 32}

    @app.websocket("/v1/streams/{stream_id}/ws")
    async def stream(websocket: WebSocket, stream_id: str):
        await websocket.accept()
        await websocket.send_json(_init())
        await websocket.send_json(_chunk(0, done=True))
        await websocket.send_json(
            {
                "type": "end",
                "artifact_url": f"/v1/jobs/{stream_id}/artifacts",
            }
        )
        await websocket.close()

    @app.get("/v1/jobs/{stream_id}/artifacts")
    def artifact(stream_id: str):
        artifact_requests.append(stream_id)
        return b"artifact"

    with TestClient(app) as client:
        row = measure_session(
            TestClientTransport(client),
            stack="omni-clip",
            arch="h3-dense",
            rep=0,
            artifact_fetch=lambda url: client.get(url).content,
            fragment_probe=lambda _init_bytes, _fragment: (0.0, None),
        )

    assert row["session_wall_ms"] >= 0
    assert row["accept_to_init_ms"] >= 0
    assert row["artifact_retrieval_ms"] >= 0
    assert row["ttff_ms"] >= 0
    assert row["detailed_spans_support"] == "unsupported"
    assert artifact_requests == ["b" * 32]


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
        assert f"span={name}:UNSUPPORTED" in proc.stdout
    for name in PHASE_KEYS:
        assert f"phase={name}" in proc.stdout
    assert "hls-ttff=UNSUPPORTED:post-generation-only" in proc.stdout
    assert "256p/120f" in proc.stdout
    assert "seed=0" in proc.stdout
    assert main(["--dry-run", "--out", str(out)]) == 0
    assert not out.exists()
