# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import socket
import threading
import time
from pathlib import Path

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
from omni_infinity.serve.models import JobStatus
from omni_infinity.serve.store import JobStoreError
from omni_infinity.serve.stream import StreamSession
from omni_infinity.streaming import CODEC


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
    return _result_with_frames(6)


def _result_with_frames(frame_count):
    frames = np.zeros((frame_count, 16, 16, 3), dtype=np.float32)
    frames[:, :, :, 0] = np.linspace(0.0, 1.0, frame_count)[:, None, None]
    return GenerationResult(
        videos=[frames],
        audio=torch.zeros(
            1, 2, frame_count * 48_000 // 24, dtype=torch.float32
        ),
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
    monkeypatch.delenv("OMNI_STREAM_QUEUE_CHUNKS", raising=False)
    assert ServerSettings.from_env().stream_enabled is False

    monkeypatch.setenv("OMNI_STREAM_ENABLED", "true")
    monkeypatch.setenv("OMNI_STREAM_CHUNK_FRAMES", "12")
    monkeypatch.setenv("OMNI_STREAM_QUEUE_CHUNKS", "3")
    assert ServerSettings.from_env().stream_enabled is True
    assert ServerSettings.from_env().stream_chunk_frames == 12
    assert ServerSettings.from_env().stream_queue_chunks == 3

    monkeypatch.setenv("OMNI_STREAM_ENABLED", "0")
    assert ServerSettings.from_env().stream_enabled is False


def test_terminal_latch_is_set_only_after_store_transition_succeeds():
    request = stream_module.StreamRequest.model_validate(_body())
    session = StreamSession("a" * 32, request)

    class FlakyStore:
        def __init__(self):
            self.calls = 0

        def transition(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise JobStoreError("disk unavailable")

    store = FlakyStore()
    assert (
        session.mark_terminal(store, JobStatus.FAILED, error="first") is False
    )
    assert session.mark_terminal(store, JobStatus.FAILED, error="retry") is True
    assert store.calls == 2


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


def test_slow_socket_with_one_queue_slot_sends_every_fragment(
    tmp_path, monkeypatch
):
    original_send = WebSocket.send_json

    async def slow_send(self, data, mode="text"):
        if data.get("type") == "chunk":
            await asyncio.sleep(0.01)
        await original_send(self, data, mode)

    monkeypatch.setattr(WebSocket, "send_json", slow_send)
    runner = _FakeRunner(_result_with_frames(40))
    settings = _stream_settings(tmp_path, stream_queue_chunks=1)
    with TestClient(create_app(settings, lambda: runner)) as client:
        stream_id = _open_stream(client)
        with client.websocket_connect(f"/v1/streams/{stream_id}/ws") as socket:
            messages = _read_until_end(socket)

    chunks = [message for message in messages if message["type"] == "chunk"]
    assert [chunk["index"] for chunk in chunks] == list(range(10))
    assert chunks[-1]["done"] is True


def test_input_changes_the_next_native_chunk_only(tmp_path, artifact_result):
    runner = _FakeRunner(artifact_result)
    settings = _stream_settings(tmp_path, stream_chunk_frames=2)
    incoming = {
        "type": "input",
        "action": "forward",
        "prompt": "run",
        "key": "ArrowUp",
        "down": True,
    }

    with TestClient(create_app(settings, lambda: runner)) as client:
        native_id = _open_stream(client, source="native", prompt="idle")
        with client.websocket_connect(f"/v1/streams/{native_id}/ws") as socket:
            assert socket.receive_json()["type"] == "init"
            first = socket.receive_json()
            socket.send_json(incoming)
            messages = _read_until_end(socket, [first])

        native_chunks = [
            message for message in messages if message["type"] == "chunk"
        ]
        assert native_chunks[0]["prompt"] == "idle"
        assert native_chunks[0]["instruction"] is None
        assert native_chunks[1]["prompt"] == "run"
        assert native_chunks[1]["action"] == "forward"
        assert native_chunks[1]["instruction"] == "forward"

        clip_id = _open_stream(client, source="clip", prompt="idle")
        with client.websocket_connect(f"/v1/streams/{clip_id}/ws") as socket:
            assert socket.receive_json()["type"] == "init"
            first = socket.receive_json()
            socket.send_json(incoming)
            messages = _read_until_end(socket, [first])

    clip_chunks = [
        message for message in messages if message["type"] == "chunk"
    ]
    assert all(chunk["prompt"] == "idle" for chunk in clip_chunks)
    assert all(chunk["instruction"] is None for chunk in clip_chunks)


def test_malformed_native_input_reports_error_and_continues(
    tmp_path, artifact_result
):
    runner = _FakeRunner(artifact_result)
    settings = _stream_settings(tmp_path, stream_chunk_frames=2)

    with TestClient(create_app(settings, lambda: runner)) as client:
        stream_id = _open_stream(client, source="native", prompt="idle")
        with client.websocket_connect(f"/v1/streams/{stream_id}/ws") as socket:
            assert socket.receive_json()["type"] == "init"
            first = socket.receive_json()
            socket.send_json({"type": "not-input"})
            error = socket.receive_json()
            second = socket.receive_json()
            end = socket.receive_json()

    assert first["index"] == 0
    assert error["type"] == "error"
    assert second["type"] == "chunk"
    assert second["index"] == 1
    assert second["prompt"] == "idle"
    assert end["type"] == "end"


def test_hls_stays_404_until_the_fallback_flag(tmp_path, artifact_result):
    runner = _FakeRunner(artifact_result)
    disabled = _stream_settings(tmp_path, stream_fallback_hls=False)
    with TestClient(create_app(disabled, lambda: runner)) as client:
        stream_id = _open_stream(client)
        assert (
            client.get(f"/v1/streams/{stream_id}/playlist.m3u8").status_code
            == 404
        )

    enabled = _stream_settings(tmp_path, stream_fallback_hls=True)
    with TestClient(create_app(enabled, lambda: runner)) as client:
        stream_id = _open_stream(client)
        playlist_url = f"/v1/streams/{stream_id}/playlist.m3u8"
        before = client.get(playlist_url)
        assert before.status_code == 409
        assert before.json()["detail"] == "stream media is not ready"

        with client.websocket_connect(f"/v1/streams/{stream_id}/ws") as socket:
            messages = _read_until_end(socket)

        playlist = client.get(playlist_url)
        assert playlist.status_code == 200
        assert '#EXT-X-MAP:URI="init.mp4"' in playlist.text
        assert "seg/0.m4s" in playlist.text
        assert "#EXT-X-ENDLIST" in playlist.text

        init = client.get(f"/v1/streams/{stream_id}/init.mp4")
        segment = client.get(f"/v1/streams/{stream_id}/seg/0.m4s")
        prompts = client.get(f"/v1/streams/{stream_id}/prompts.json")

    assert init.status_code == 200
    assert b"moov" in init.content
    first_chunk = next(
        message for message in messages if message["type"] == "chunk"
    )
    assert segment.content == base64.b64decode(first_chunk["video_b64"])
    assert prompts.json()["cues"][0]["prompt"] == "a red ball bouncing"


def test_webui_is_served_only_when_streaming_is_enabled(tmp_path):
    dark = ServerSettings(jobs_dir=tmp_path, optimizations=())
    with TestClient(create_app(dark, lambda: object())) as client:
        assert client.get("/").status_code == 404

    live = ServerSettings(
        jobs_dir=tmp_path, optimizations=(), stream_enabled=True
    )
    with TestClient(create_app(live, lambda: object())) as client:
        page = client.get("/")
        script = client.get("/player.js")

    assert page.status_code == 200
    assert 'id="prompt-panel"' in page.text
    assert "<video" in page.text
    for needle in (
        "SourceBuffer",
        "currentTime",
        "keydown",
        "keyup",
        "forward",
        "activeCue",
    ):
        assert needle in script.text


def test_readme_documents_pseudo_streaming():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "ClipChunker does not overlap generation with playback." in readme
    for needle in (
        "OMNI_STREAM_ENABLED",
        "POST /v1/streams",
        "WS /v1/streams/{id}/ws",
        "OMNI_STREAM_FALLBACK_HLS",
    ):
        assert needle in readme


def test_artifact_mux_does_not_wait_behind_an_unrelated_job(
    tmp_path, artifact_result, monkeypatch
):
    send_blocked = threading.Event()
    release_send = threading.Event()
    job_blocked = threading.Event()
    release_job = threading.Event()
    original_send = WebSocket.send_json

    async def slow_send(self, data, mode="text"):
        if data.get("type") == "chunk" and not send_blocked.is_set():
            send_blocked.set()
            while not release_send.is_set():
                await asyncio.sleep(0.01)
        await original_send(self, data, mode)

    class SecondCallBlocksRunner(_FakeRunner):
        def __init__(self, result):
            super().__init__(result)
            self.calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                job_blocked.set()
                assert release_job.wait(timeout=10)
            return super().generate(*args, **kwargs)

    monkeypatch.setattr(WebSocket, "send_json", slow_send)
    runner = SecondCallBlocksRunner(artifact_result)
    settings = _stream_settings(tmp_path)
    with TestClient(create_app(settings, lambda: runner)) as client:
        stream_id = _open_stream(client)
        messages: list[dict] = []
        stream_done = threading.Event()

        def reader():
            with client.websocket_connect(
                f"/v1/streams/{stream_id}/ws"
            ) as socket:
                _read_until_end(socket, messages)
            stream_done.set()

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            assert send_blocked.wait(timeout=10)
            job = client.post("/v1/jobs", json=_body())
            assert job.status_code == 202
            assert job_blocked.wait(timeout=10)
            release_send.set()
            assert stream_done.wait(timeout=3)
            assert messages[-1]["type"] == "end"
            assert not release_job.is_set()
        finally:
            release_send.set()
            release_job.set()
            thread.join(timeout=15)


def test_stream_artifact_executor_shuts_down_with_the_app(
    tmp_path, artifact_result
):
    runner = _FakeRunner(artifact_result)
    settings = _stream_settings(tmp_path)
    with TestClient(create_app(settings, lambda: runner)) as client:
        stream_service = client.app.state.stream_service

    with pytest.raises(RuntimeError, match="cannot schedule new futures"):
        stream_service.artifact_executor.submit(lambda: None)


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


def test_success_transition_failure_keeps_muxed_artifacts(
    tmp_path, artifact_result, monkeypatch
):
    runner = _FakeRunner(artifact_result)
    settings = _stream_settings(tmp_path)
    with TestClient(create_app(settings, lambda: runner)) as client:
        store = client.app.state.store
        real_transition = store.transition
        failed_once = False

        def fail_success_once(job_id, status, **kwargs):
            nonlocal failed_once
            if status == JobStatus.SUCCEEDED and not failed_once:
                failed_once = True
                raise JobStoreError("disk unavailable")
            return real_transition(job_id, status, **kwargs)

        monkeypatch.setattr(store, "transition", fail_success_once)
        stream_id = _open_stream(client)
        with client.websocket_connect(f"/v1/streams/{stream_id}/ws") as socket:
            messages = _read_until_end(socket)

        assert messages[-1]["type"] == "error"
        assert client.get(f"/v1/jobs/{stream_id}").json()["status"] == "failed"
        assert (tmp_path / stream_id / "output.mp4").is_file()
        assert (tmp_path / stream_id / "output.wav").is_file()


def test_closed_socket_runtime_error_is_not_logged_as_server_failure(
    tmp_path, artifact_result, monkeypatch, caplog
):
    async def closed_send(self, data, mode="text"):
        raise RuntimeError(
            'Cannot call "send" once a close message has been sent.'
        )

    monkeypatch.setattr(WebSocket, "send_json", closed_send)
    runner = _FakeRunner(artifact_result)
    settings = _stream_settings(tmp_path)
    with caplog.at_level(logging.ERROR):
        with TestClient(create_app(settings, lambda: runner)) as client:
            stream_id = _open_stream(client)
            with client.websocket_connect(
                f"/v1/streams/{stream_id}/ws"
            ) as socket:
                with pytest.raises(WebSocketDisconnect):
                    socket.receive_json()

    assert not any(
        "socket failed" in record.getMessage() for record in caplog.records
    )


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
