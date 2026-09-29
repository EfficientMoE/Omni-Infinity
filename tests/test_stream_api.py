# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import io
import json
import socket
import threading
import time

import av
import httpx
import numpy as np
import pytest
import torch
import uvicorn
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from omni_infinity.runner import GenerationResult
from omni_infinity.serve import stream as stream_module
from omni_infinity.serve.app import ServerSettings, create_app
from omni_infinity.serve.stream import ChunkPipe
from omni_infinity.streaming import CODEC, MediaChunk


def _body(**overrides):
    body = {
        "type": "fl2va",
        "prompt": "a red ball bouncing",
        "optimizations": [],
    }
    body.update(overrides)
    return body


@pytest.fixture
def artifact_result():
    frames = np.zeros((6, 16, 16, 3), dtype=np.float32)
    frames[:, :, :, 0] = np.linspace(0.0, 1.0, 6)[:, None, None]
    return GenerationResult(
        videos=[frames],
        audio=torch.zeros(1, 2, 8000, dtype=torch.float32),
        sampling_rate=48000,
        latents=None,
        audio_latents=None,
    )


class _FakeRunner:
    """Counts overlapping generations and can block inside ``generate``."""

    def __init__(self, result, *, release: threading.Event | None = None):
        self.result = result
        self.release = release
        self.entered = threading.Event()
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def generate(self, prompt, *, step_callback, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            if self.release is not None:
                assert self.release.wait(timeout=10)
            step_callback(1, kwargs["num_inference_steps"])
            return self.result
        finally:
            with self.lock:
                self.active -= 1


def _stream_settings(tmp_path, **overrides):
    values = {
        "jobs_dir": tmp_path,
        "optimizations": (),
        "stream_enabled": True,
        "stream_chunk_frames": 4,
    }
    values.update(overrides)
    return ServerSettings(**values)


def _open_stream(client, **overrides):
    response = client.post("/v1/streams", json=_body(**overrides))
    assert response.status_code == 202, response.text
    return response.json()["stream_id"]


def _read_until_end(websocket, messages=None):
    collected = messages if messages is not None else []
    while True:
        message = websocket.receive_json()
        collected.append(message)
        if message["type"] in {"end", "error"}:
            return collected


def _probe(path):
    with av.open(str(path)) as container:
        video = container.streams.video
        audio = container.streams.audio
        return {
            "video_streams": len(video),
            "audio_streams": len(audio),
            "video_codec": video[0].codec_context.name,
            "audio_codec": audio[0].codec_context.name,
            "audio_channels": audio[0].layout.name,
            "sample_rate": audio[0].sample_rate,
            "frames": sum(1 for _ in container.decode(video=0)),
        }


def test_streams_are_absent_unless_enabled(tmp_path):
    settings = ServerSettings(jobs_dir=tmp_path, optimizations=())
    with TestClient(create_app(settings, lambda: object())) as client:
        assert client.post("/v1/streams", json=_body()).status_code == 404


def test_open_session_caps_and_reuses_job_errors(tmp_path):
    settings = ServerSettings(
        jobs_dir=tmp_path, optimizations=(), stream_enabled=True
    )
    with TestClient(create_app(settings, lambda: object())) as client:
        first = client.post("/v1/streams", json=_body())
        assert first.status_code == 202
        assert len(first.json()["stream_id"]) == 32

        second = client.post("/v1/streams", json=_body())
        assert second.status_code == 409
        assert second.json()["detail"] == "stream session limit reached"

        ref = client.post("/v1/streams", json=_body(type="ref2va"))
        assert ref.status_code == 501

        other = client.post("/v1/streams", json=_body(model_arch="vdn-hybrid"))
        assert other.status_code == 409
        assert other.json()["detail"] == (
            "request profile does not match the loaded server profile"
        )
        assert len(list(tmp_path.iterdir())) == 1


def test_finish_releases_stream_session_slot(tmp_path):
    settings = ServerSettings(
        jobs_dir=tmp_path, optimizations=(), stream_enabled=True
    )
    with TestClient(create_app(settings, lambda: object())) as client:
        first = client.post("/v1/streams", json=_body())
        stream_id = first.json()["stream_id"]
        client.app.state.stream_service.finish(stream_id)

        second = client.post("/v1/streams", json=_body())
        assert second.status_code == 202


def test_stream_persists_only_generation_request_fields(tmp_path):
    settings = ServerSettings(
        jobs_dir=tmp_path, optimizations=(), stream_enabled=True
    )
    with TestClient(create_app(settings, lambda: object())) as client:
        response = client.post(
            "/v1/streams",
            json=_body(
                source="native",
                action_script=[
                    {"t": 0, "action": "forward", "instruction": "go"}
                ],
            ),
        )
        assert response.status_code == 202
        stream_id = response.json()["stream_id"]

        request = client.get(f"/v1/jobs/{stream_id}").json()["request"]
        assert request["prompt"] == "a red ball bouncing"
        assert "action_script" not in request
        assert "source" not in request


def test_stream_validates_typed_action_script(tmp_path):
    settings = ServerSettings(
        jobs_dir=tmp_path, optimizations=(), stream_enabled=True
    )
    with TestClient(create_app(settings, lambda: object())) as client:
        response = client.post(
            "/v1/streams",
            json=_body(
                action_script=[
                    {"t": 0, "action": "forward", "instruction": "go"}
                ]
                * 65
            ),
        )
        assert response.status_code == 422


def test_stream_enabled_env_parses_bools(monkeypatch):
    monkeypatch.delenv("OMNI_STREAM_ENABLED", raising=False)
    monkeypatch.delenv("OMNI_STREAM_CHUNK_FRAMES", raising=False)
    assert ServerSettings.from_env().stream_enabled is False

    monkeypatch.setenv("OMNI_STREAM_ENABLED", "true")
    monkeypatch.setenv("OMNI_STREAM_CHUNK_FRAMES", "12")
    assert ServerSettings.from_env().stream_enabled is True
    assert ServerSettings.from_env().stream_chunk_frames == 12

    monkeypatch.setenv("OMNI_STREAM_ENABLED", "0")
    assert ServerSettings.from_env().stream_enabled is False


def _pipe_chunk(index, *, keyframe=False, done=False):
    return MediaChunk(
        index=index,
        pts=index / 24,
        duration=1 / 24,
        keyframe=keyframe,
        video_bytes=b"\x00",
        audio_bytes=None,
        prompt="prompt",
        done=done,
    )


def test_chunk_pipe_protects_the_first_chunk_and_resyncs_on_keyframes():
    async def scenario():
        pipe = ChunkPipe(asyncio.get_running_loop(), 1)
        pipe.put(_pipe_chunk(0, keyframe=True))
        pipe.put(_pipe_chunk(1))
        pipe.put(_pipe_chunk(2, keyframe=True))
        first, closed, error = await pipe.drain()

        pipe.put(_pipe_chunk(3))
        pipe.put(_pipe_chunk(4, keyframe=True))
        second, _, _ = await pipe.drain()

        pipe.put(_pipe_chunk(5, done=True))
        pipe.close()
        last, done, _ = await pipe.drain()
        return first, closed, error, second, last, done, pipe.dropped

    first, closed, error, second, last, done, dropped = asyncio.run(scenario())

    assert [chunk.index for chunk in first] == [0]
    assert (closed, error) == (False, None)
    assert [chunk.index for chunk in second] == [4]
    assert [chunk.index for chunk in last] == [5]
    assert done is True
    assert dropped == 3


def test_socket_route_is_absent_unless_streaming_is_enabled(tmp_path):
    settings = ServerSettings(jobs_dir=tmp_path, optimizations=())
    with TestClient(create_app(settings, lambda: object())) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/v1/streams/{'a' * 32}/ws"):
                pass


def test_socket_rejects_an_unknown_stream_session(tmp_path):
    settings = _stream_settings(tmp_path)
    with TestClient(create_app(settings, lambda: object())) as client:
        with client.websocket_connect(f"/v1/streams/{'a' * 32}/ws") as socket:
            message = socket.receive_json()
    assert message["type"] == "error"
    assert "unknown" in message["detail"]


def test_socket_rejects_a_second_player_for_one_session(
    tmp_path, artifact_result
):
    release = threading.Event()
    runner = _FakeRunner(artifact_result, release=release)
    settings = _stream_settings(tmp_path)
    with TestClient(create_app(settings, lambda: runner)) as client:
        stream_id = _open_stream(client)
        messages: list[dict] = []

        def reader():
            with client.websocket_connect(
                f"/v1/streams/{stream_id}/ws"
            ) as socket:
                _read_until_end(socket, messages)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            assert runner.entered.wait(timeout=10)
            with client.websocket_connect(
                f"/v1/streams/{stream_id}/ws"
            ) as second:
                rejected = second.receive_json()
            assert rejected == {
                "type": "error",
                "detail": "stream session already has a player",
            }
        finally:
            release.set()
            thread.join(timeout=15)

        assert not thread.is_alive()
        assert messages[-1]["type"] == "end"


def test_socket_sends_the_first_chunk_before_the_artifact(
    tmp_path, artifact_result, monkeypatch
):
    real_write = stream_module.write_artifacts
    entered = threading.Event()
    release = threading.Event()

    def gated_write(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=10)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(stream_module, "write_artifacts", gated_write)

    runner = _FakeRunner(artifact_result)
    settings = _stream_settings(tmp_path)
    with TestClient(create_app(settings, lambda: runner)) as client:
        stream_id = _open_stream(
            client,
            action_script=[{"t": 0, "action": "wait", "instruction": "hold"}],
        )
        video_path = tmp_path / stream_id / "output.mp4"
        with client.websocket_connect(f"/v1/streams/{stream_id}/ws") as socket:
            init = socket.receive_json()
            assert init["type"] == "init"
            assert init["codec"] == CODEC
            assert base64.b64decode(init["init_b64"])[4:8] == b"ftyp"
            assert init["action_script"] == [
                {"t": 0.0, "action": "wait", "instruction": "hold"}
            ]

            first = socket.receive_json()
            assert first["type"] == "chunk"
            assert first["index"] == 0
            assert first["keyframe"] is True
            assert first["prompt"] == "a red ball bouncing"
            assert first["audio_b64"] is None
            assert base64.b64decode(first["video_b64"])[4:8] == b"moof"
            assert first["done"] is False
            assert not video_path.exists()

            release.set()
            messages = _read_until_end(socket, [init, first])

        assert entered.is_set()
        chunks = [item for item in messages if item["type"] == "chunk"]
        assert [item["index"] for item in chunks] == [0, 1]
        assert chunks[-1]["done"] is True
        assert messages[-1] == {
            "type": "end",
            "artifact_url": f"/v1/jobs/{stream_id}/artifacts",
        }

        record = client.get(f"/v1/jobs/{stream_id}").json()
        assert record["status"] == "succeeded"
        assert record["artifacts"]["video_url"] == (
            f"/v1/jobs/{stream_id}/artifacts"
        )

        artifact = client.get(f"/v1/jobs/{stream_id}/artifacts")
        assert artifact.status_code == 200
        assert artifact.content == video_path.read_bytes()
        downloaded = tmp_path / "stream.mp4"
        downloaded.write_bytes(artifact.content)
        probed = _probe(downloaded)
        assert probed["video_streams"] == 1
        assert probed["audio_streams"] == 1


def test_stream_artifact_matches_the_job_artifact(tmp_path, artifact_result):
    runner = _FakeRunner(artifact_result)
    settings = _stream_settings(tmp_path)
    with TestClient(create_app(settings, lambda: runner)) as client:
        stream_id = _open_stream(client)
        with client.websocket_connect(f"/v1/streams/{stream_id}/ws") as socket:
            _read_until_end(socket)

        job = client.post("/v1/jobs", json=_body())
        job_id = job.json()["id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            body = client.get(f"/v1/jobs/{job_id}").json()
            if body["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        assert body["status"] == "succeeded"

    assert _probe(tmp_path / stream_id / "output.mp4") == _probe(
        tmp_path / job_id / "output.mp4"
    )


def test_socket_leaves_the_job_worker_free_during_slow_sends(
    tmp_path, artifact_result, monkeypatch
):
    stuck = threading.Event()
    released = threading.Event()
    original_send = WebSocket.send_json

    async def slow_send(self, data, mode="text"):
        if data.get("type") == "chunk":
            stuck.set()
            while not released.is_set():
                await asyncio.sleep(0.01)
        await original_send(self, data, mode)

    monkeypatch.setattr(WebSocket, "send_json", slow_send)

    runner = _FakeRunner(artifact_result)
    settings = _stream_settings(tmp_path)
    with TestClient(create_app(settings, lambda: runner)) as client:
        stream_id = _open_stream(client)
        messages: list[dict] = []

        def reader():
            with client.websocket_connect(
                f"/v1/streams/{stream_id}/ws"
            ) as socket:
                _read_until_end(socket, messages)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            assert stuck.wait(timeout=10)

            job = client.post("/v1/jobs", json=_body())
            job_id = job.json()["id"]
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                body = client.get(f"/v1/jobs/{job_id}").json()
                if body["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(0.01)
            assert body["status"] == "succeeded"
            assert not released.is_set()
        finally:
            released.set()
            thread.join(timeout=15)

        assert not thread.is_alive()
        assert messages[-1]["type"] == "end"
        assert runner.max_active == 1


def test_stream_waits_for_the_job_worker(tmp_path, artifact_result):
    release = threading.Event()
    runner = _FakeRunner(artifact_result, release=release)
    settings = _stream_settings(tmp_path)
    with TestClient(create_app(settings, lambda: runner)) as client:
        job = client.post("/v1/jobs", json=_body())
        assert job.status_code == 202
        assert runner.entered.wait(timeout=10)

        stream_id = _open_stream(client)
        messages: list[dict] = []

        def reader():
            with client.websocket_connect(
                f"/v1/streams/{stream_id}/ws"
            ) as socket:
                _read_until_end(socket, messages)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            time.sleep(0.3)
            assert runner.max_active == 1
            assert messages == []
            assert client.get(f"/v1/jobs/{stream_id}").json()["status"] == (
                "queued"
            )
        finally:
            release.set()
            thread.join(timeout=15)

        assert not thread.is_alive()
        assert runner.max_active == 1
        assert messages[-1]["type"] == "end"


def test_socket_reports_generation_failure(tmp_path):
    class FailingRunner:
        def generate(self, *args, **kwargs):
            raise RuntimeError("GPU exploded")

    settings = _stream_settings(tmp_path)
    with TestClient(create_app(settings, lambda: FailingRunner())) as client:
        stream_id = _open_stream(client)
        with client.websocket_connect(f"/v1/streams/{stream_id}/ws") as socket:
            message = socket.receive_json()

        assert message["type"] == "error"
        assert message["detail"] == "RuntimeError: GPU exploded"
        assert not (tmp_path / stream_id / "output.mp4").exists()

        record = client.get(f"/v1/jobs/{stream_id}").json()
        assert record["status"] == "failed"
        assert record["error"] == "RuntimeError: GPU exploded"
        assert client.get(f"/v1/jobs/{stream_id}/artifacts").status_code == 409

        assert client.post("/v1/streams", json=_body()).status_code == 202


def _free_local_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.mark.timeout(120)
def test_live_uvicorn_socket_upgrade_streams_chunks(tmp_path, artifact_result):
    websockets_sync = pytest.importorskip("websockets.sync.client")

    runner = _FakeRunner(artifact_result)
    port = _free_local_port()
    app = create_app(_stream_settings(tmp_path), lambda: runner)
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning", ws="auto"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not server.started:
            assert thread.is_alive(), "uvicorn exited during startup"
            time.sleep(0.05)
        assert server.started, "uvicorn did not start"

        created = httpx.post(f"{base_url}/v1/streams", json=_body(), timeout=10)
        assert created.status_code == 202, created.text
        stream_id = created.json()["stream_id"]

        messages = []
        with websockets_sync.connect(
            f"ws://127.0.0.1:{port}/v1/streams/{stream_id}/ws",
            open_timeout=10,
        ) as socket:
            while True:
                message = json.loads(socket.recv(timeout=30))
                messages.append(message)
                if message["type"] in {"end", "error"}:
                    break

        assert messages[0]["type"] == "init"
        assert messages[0]["codec"] == CODEC
        chunks = [item for item in messages if item["type"] == "chunk"]
        assert [item["index"] for item in chunks] == [0, 1]
        assert chunks[-1]["done"] is True
        assert messages[-1] == {
            "type": "end",
            "artifact_url": f"/v1/jobs/{stream_id}/artifacts",
        }

        artifact = httpx.get(
            f"{base_url}/v1/jobs/{stream_id}/artifacts", timeout=30
        )
        assert artifact.status_code == 200
        with av.open(io.BytesIO(artifact.content)) as container:
            assert len(container.streams.video) == 1
            assert len(container.streams.audio) == 1
    finally:
        server.should_exit = True
        thread.join(timeout=30)
