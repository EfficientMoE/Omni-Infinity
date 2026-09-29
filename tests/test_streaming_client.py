# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the shared streaming benchmark client."""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from benchmarks.streaming.client import (
    BenchmarkClient,
    ChunkEvent,
    InitEvent,
    LocalhostConfig,
    MetricSupport,
    ProtocolError,
    TestClientTransport,
    TimedEvent,
    build_request,
    detect_native_chunker,
    observe_events,
    parse_event,
)


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _init() -> dict:
    return {
        "type": "init",
        "codec": 'video/mp4; codecs="avc1.42E01E,mp4a.40.2"',
        "init_b64": _b64(b"init"),
        "action_script": [
            {"t": 0.25, "action": "forward", "instruction": "go"}
        ],
    }


def _chunk(index: int, *, pts: float, done: bool) -> dict:
    return {
        "type": "chunk",
        "index": index,
        "pts": pts,
        "duration": 1.0,
        "keyframe": True,
        "video_b64": _b64(f"fragment-{index}".encode()),
        "audio_b64": None,
        "prompt": "a red ball bouncing",
        "instruction": "go" if index == 0 else None,
        "action": "forward" if index == 0 else None,
        "done": done,
    }


def test_actual_wire_messages_are_strictly_validated():
    init = parse_event(_init())
    assert isinstance(init, InitEvent)
    assert init.init_bytes == b"init"

    chunk = parse_event(_chunk(0, pts=0.0, done=True))
    assert isinstance(chunk, ChunkEvent)
    assert chunk.video_bytes == b"fragment-0"

    assert parse_event(
        {"type": "end", "artifact_url": "/v1/jobs/id/artifacts"}
    ).artifact_url.endswith("/artifacts")
    assert parse_event({"type": "error", "detail": "failed"}).detail == "failed"

    with pytest.raises(ProtocolError, match="unexpected fields"):
        parse_event({**_init(), "server_produce_ms": 1.0})
    with pytest.raises(ProtocolError, match="index"):
        parse_event({**_chunk(0, pts=0.0, done=True), "index": True})
    with pytest.raises(ProtocolError, match="base64"):
        parse_event({**_chunk(0, pts=0.0, done=True), "video_b64": "***"})


def test_testclient_transport_posts_and_consumes_websocket():
    app = FastAPI()

    @app.post("/v1/streams", status_code=202)
    def create_stream(payload: dict):
        assert payload["resolution"] == "256p"
        return {"stream_id": "a" * 32}

    @app.websocket("/v1/streams/{stream_id}/ws")
    async def stream(websocket: WebSocket, stream_id: str):
        assert stream_id == "a" * 32
        await websocket.accept()
        await websocket.send_json(_init())
        await websocket.send_json(_chunk(0, pts=0.0, done=True))
        await websocket.send_json(
            {
                "type": "end",
                "artifact_url": f"/v1/jobs/{stream_id}/artifacts",
            }
        )
        await websocket.close()

    with TestClient(app) as test_client:
        result = BenchmarkClient(
            TestClientTransport(test_client),
            fragment_probe=lambda _init_bytes, _fragment: (0.0, 0.0),
        ).run(build_request())

    assert result.stream_id == "a" * 32
    assert result.metrics.ttff_ms >= 0
    assert result.metrics.arrival_gaps_ms == ()
    assert result.metrics.stall_count == 0
    assert result.metrics.cue_alignment is True
    assert result.events[-1].event.artifact_url.endswith("/artifacts")


def test_observed_metrics_are_arrival_based_not_production_latency():
    events = (
        TimedEvent(parse_event(_init()), 1_000_000_000),
        TimedEvent(parse_event(_chunk(0, pts=0.0, done=False)), 1_100_000_000),
        TimedEvent(parse_event(_chunk(1, pts=1.0, done=True)), 2_300_000_000),
        TimedEvent(
            parse_event(
                {"type": "end", "artifact_url": "/v1/jobs/id/artifacts"}
            ),
            2_400_000_000,
        ),
    )
    metrics = observe_events(
        accepted_ns=1_000_000_000,
        events=events,
        request_prompt="a red ball bouncing",
        fragment_probe=lambda _init_bytes, _fragment: (0.0, 0.020),
    )
    assert metrics.ttff_ms == 100.0
    assert metrics.arrival_gaps_ms == (1200.0,)
    assert metrics.stall_count == 1
    assert metrics.av_pts_boundary_offset_ms == 20.0
    assert metrics.cue_alignment is True
    assert metrics.production_latency == MetricSupport(
        False, "wire protocol has no server production timestamps"
    )
    fields = metrics.contract_fields()
    assert fields["arrival_gap_ms"] == "[1200.0]"
    assert fields["arrival_gap_p50"] == 1200.0
    assert fields["production_latency_support"] == "unsupported"
    assert "chunk_produce_ms" not in fields


def test_request_defaults_allow_dense_256p_and_vdn_768p():
    assert build_request(model_arch="h3-dense")["resolution"] == "256p"
    assert (
        build_request(model_arch="h3-dense", resolution="768p")["resolution"]
        == "768p"
    )
    assert build_request(model_arch="vdn-hybrid")["resolution"] == "768p"
    with pytest.raises(ValueError, match="768p"):
        build_request(model_arch="vdn-hybrid", resolution="256p")


def test_localhost_config_is_explicit_and_import_does_not_load_torch():
    config = LocalhostConfig("http://127.0.0.1:8000")
    assert config.websocket_url("abc") == (
        "ws://127.0.0.1:8000/v1/streams/abc/ws"
    )
    with pytest.raises(ValueError, match="localhost"):
        LocalhostConfig("https://example.com")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['torch'] = None; "
                "import benchmarks.streaming.client"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_synthetic_native_chunker_is_reported_unsupported():
    support = detect_native_chunker(
        Path(__file__).parents[1] / "omni_infinity/streaming/native.py"
    )
    assert support.supported is False
    assert support.reason == "NativeChunker is the synthetic 16x16 stub"
