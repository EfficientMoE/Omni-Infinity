# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import re
import threading
import time

import av
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from omni_infinity.demo.models import DemoRequest
from omni_infinity.demo.service import DemoService
from omni_infinity.registry import resolve_profile
from omni_infinity.runner import GenerationResult
from omni_infinity.serve.app import ServerSettings, create_app
from omni_infinity.serve.models import GenerationRequest
from omni_infinity.serve.service import JobService
from omni_infinity.serve.store import JobStore


def test_prompt_count_rejects_short_lists():
    # duration 15s requires 3 prompts
    with pytest.raises(ValidationError) as e1:
        DemoRequest(prompts=["a", "b"], duration="15s")
    assert "duration 15s requires 3 prompts, got 2" in str(e1.value)

    # duration 2min requires 24 prompts
    with pytest.raises(ValidationError) as e2:
        DemoRequest(prompts=["a"] * 23, duration="2min")
    assert "duration 2min requires 24 prompts, got 23" in str(e2.value)

    # duration 5min requires 59 prompts
    with pytest.raises(ValidationError) as e3:
        DemoRequest(prompts=["a"] * 58, duration="5min")
    assert "duration 5min requires 59 prompts, got 58" in str(e3.value)

    # This one should pass
    d = DemoRequest(prompts=["a"] * 3, duration="15s")
    assert d.duration == "15s"


def test_prompt_count_rejects_long_lists():
    with pytest.raises(ValidationError) as e:
        DemoRequest(prompts=["a"] * 4, duration="15s")
    assert "duration 15s requires 3 prompts, got 4" in str(e.value)


def test_schedule_range_rejects_reversed():
    with pytest.raises(ValidationError) as e:
        DemoRequest(
            prompts=["a"] * 3,
            duration="15s",
            schedule_start=5,
            schedule_end=2,
        )
    assert "schedule end is before schedule start" in str(e.value)

    d = DemoRequest(
        prompts=["a"] * 3,
        duration="15s",
        schedule_start=3,
        schedule_end=3,
    )
    assert d.schedule_start == d.schedule_end == 3


def test_duplicate_optimizations_rejected():
    with pytest.raises(ValidationError) as e:
        DemoRequest(
            prompts=["a"] * 3, duration="15s", optimizations=["fp8", "fp8"]
        )
    assert "optimization names must be unique" in str(e.value)


def _first_frame_base64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _segment_result() -> GenerationResult:
    frames = np.zeros((124, 4, 4, 3), dtype=np.float32)
    audio = torch.zeros(2, 124 * 48000 // 24, dtype=torch.float32)
    return GenerationResult(
        videos=[frames],
        audio=audio,
        sampling_rate=48000,
        latents=None,
        audio_latents=None,
    )


def _demo_payload(**overrides):
    payload = {
        "prompts": ["opening", "middle", "close"],
        "duration": "15s",
        "first_frame_base64": _first_frame_base64(),
    }
    payload.update(overrides)
    return payload


def _poll_demo(client, demo_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/v1/demos/{demo_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"succeeded", "failed", "cancelled"}:
            return body
        time.sleep(0.01)
    raise AssertionError(f"demo {demo_id} did not finish")


def test_http_demo_lifecycle_validation_and_artifact(tmp_path):
    class FakeRunner:
        def generate(
            self, prompt, *, step_callback, num_inference_steps, **kwargs
        ):
            for completed in range(1, num_inference_steps + 1):
                step_callback(completed, num_inference_steps)
            return _segment_result()

    settings = ServerSettings(jobs_dir=tmp_path / "jobs", optimizations=())
    with TestClient(create_app(settings, FakeRunner)) as client:
        response = client.post("/v1/demos", json=_demo_payload())
        assert response.status_code == 202
        demo_id = response.json()["id"]
        assert re.fullmatch(r"[0-9a-f]{32}", demo_id)
        assert response.headers["location"] == f"/v1/demos/{demo_id}"
        assert client.get(f"/v1/jobs/{demo_id}").status_code == 404

        finished = _poll_demo(client, demo_id)
        assert finished["status"] == "succeeded"
        assert finished["progress"]["total_steps"] == 24

        artifact = client.get(f"/v1/demos/{demo_id}/artifacts")
        assert artifact.status_code == 200
        assert artifact.headers["content-type"] == "video/mp4"
        downloaded = tmp_path / "demo.mp4"
        downloaded.write_bytes(artifact.content)
        with av.open(downloaded) as container:
            assert sum(1 for _ in container.decode(video=0)) == 360

        assert client.get(f"/v1/demos/{'f' * 32}").status_code == 404
        assert client.get(f"/v1/demos/{'f' * 32}/artifacts").status_code == 404

        legacy = client.post(
            "/v1/jobs",
            json={
                "type": "fl2va",
                "prompts": ["a", "b", "c"],
                "duration": "15s",
            },
        )
        assert legacy.status_code == 422

        short = client.post("/v1/demos", json=_demo_payload(prompts=["a", "b"]))
        assert short.status_code == 422
        assert "duration 15s requires 3 prompts, got 2" in short.text

        missing_frame = client.post(
            "/v1/demos",
            json={"prompts": ["a", "b", "c"], "duration": "15s"},
        )
        assert missing_frame.status_code == 422
        assert missing_frame.json()["detail"] == "first frame is required"

        invalid_frame = client.post(
            "/v1/demos",
            json=_demo_payload(first_frame_base64="not base64!"),
        )
        assert invalid_frame.status_code == 422
        assert (
            invalid_frame.json()["detail"] == "invalid base64 for first frame"
        )

        conflict = client.post(
            "/v1/demos", json=_demo_payload(model_arch="vdn-hybrid")
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"] == (
            "request profile does not match the loaded server profile"
        )


def test_http_demo_artifact_is_conflict_until_succeeded(tmp_path):
    class BlockingRunner:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def generate(
            self, prompt, *, step_callback, num_inference_steps, **kwargs
        ):
            self.started.set()
            assert self.release.wait(timeout=2)
            for completed in range(1, num_inference_steps + 1):
                step_callback(completed, num_inference_steps)
            return _segment_result()

    runner = BlockingRunner()
    settings = ServerSettings(jobs_dir=tmp_path / "jobs", optimizations=())
    with TestClient(create_app(settings, lambda: runner)) as client:
        response = client.post("/v1/demos", json=_demo_payload())
        demo_id = response.json()["id"]
        assert runner.started.wait(timeout=2)
        pending = client.get(f"/v1/demos/{demo_id}/artifacts")
        assert pending.status_code == 409
        assert pending.json()["detail"] == "artifact is not ready"
        runner.release.set()
        assert _poll_demo(client, demo_id)["status"] == "succeeded"


def test_demo_and_job_services_share_the_single_executor(tmp_path):
    from omni_infinity.demo.store import DemoStore

    class BlockingRunner:
        def __init__(self):
            self.job_started = threading.Event()
            self.demo_started = threading.Event()
            self.release_job = threading.Event()
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def generate(self, prompt, *, step_callback, **kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                if prompt == "job":
                    self.job_started.set()
                    assert self.release_job.wait(timeout=2)
                else:
                    self.demo_started.set()
                steps = kwargs.get("num_inference_steps", 1)
                for completed in range(1, steps + 1):
                    step_callback(completed, steps)
                return _segment_result()
            finally:
                with self.lock:
                    self.active -= 1

    runner = BlockingRunner()
    jobs = JobService(
        runner,
        resolve_profile("h3-dense", []),
        JobStore(tmp_path / "jobs"),
    )
    demo_store = DemoStore(tmp_path / "demos")
    demo = DemoService(
        runner,
        "h3-dense",
        store=demo_store,
        executor=jobs.executor,
    )
    try:
        jobs.submit(GenerationRequest(type="fl2va", prompt="job"))
        assert runner.job_started.wait(timeout=2)
        demo_record = demo.submit(DemoRequest(**_demo_payload()))
        assert demo.executor is jobs.executor
        assert not runner.demo_started.wait(timeout=0.1)

        runner.release_job.set()
        assert runner.demo_started.wait(timeout=2)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if demo_store.get(demo_record.id).status.value == "succeeded":
                break
            time.sleep(0.01)
        else:
            raise AssertionError("demo did not finish")
        assert runner.max_active == 1
    finally:
        runner.release_job.set()
        jobs.shutdown()


def test_demo_stream_session_expires_when_never_connected():
    from omni_infinity.serve.stream import DemoStreamService

    service = DemoStreamService(session_ttl=0.0)
    service.open_session("a")
    assert "a" in service._sessions

    service.open_session("b")
    assert "a" not in service._sessions
    assert "b" in service._sessions

    service.publish("a", {"type": "chunk"})

    service._sessions["b"].connected = True
    service.open_session("c")
    assert "b" in service._sessions
    assert "c" in service._sessions


def test_stream_starts_at_the_first_clip(tmp_path):
    class BlockingSecondRunner:
        def __init__(self):
            self.calls = 0
            self.lock = threading.Lock()
            self.second_started = threading.Event()
            self.release_second = threading.Event()

        def generate(
            self, prompt, *, step_callback, num_inference_steps, **kwargs
        ):
            with self.lock:
                self.calls += 1
                call = self.calls
            if call == 2:
                self.second_started.set()
                assert self.release_second.wait(timeout=10)
            for completed in range(1, num_inference_steps + 1):
                step_callback(completed, num_inference_steps)
            return _segment_result()

    prompts = ["opening", "middle", "close"]
    runner = BlockingSecondRunner()
    settings = ServerSettings(
        jobs_dir=tmp_path / "streaming",
        optimizations=(),
        stream_enabled=True,
        stream_chunk_frames=124,
    )
    messages = []
    try:
        with TestClient(create_app(settings, lambda: runner)) as client:
            response = client.post(
                "/v1/demos",
                json=_demo_payload(
                    prompts=prompts, schedule_start=0, schedule_end=0
                ),
            )
            assert response.status_code == 202
            demo_id = response.json()["id"]

            with client.websocket_connect(f"/v1/demos/{demo_id}/ws") as socket:
                first_init = socket.receive_json()
                first_chunk = socket.receive_json()
                assert first_init["type"] == "init"
                assert first_init["clip"] == 0
                assert first_chunk["type"] == "chunk"
                assert first_chunk["prompt"] == prompts[0]
                assert not runner.release_second.is_set()

                runner.release_second.set()
                while True:
                    message = socket.receive_json()
                    messages.append(message)
                    if message["type"] == "end":
                        break

            finished = _poll_demo(client, demo_id)
            assert finished["status"] == "succeeded"
    finally:
        runner.release_second.set()

    messages = [first_init, first_chunk, *messages]
    chunks = [message for message in messages if message["type"] == "chunk"]
    assert all(chunk["pts"] < 15 for chunk in chunks)
    assert chunks[-1]["done"] is True

    for clip in (1, 2):
        init_position = next(
            i
            for i, message in enumerate(messages)
            if message["type"] == "init" and message["clip"] == clip
        )
        chunk_position = next(
            i
            for i, message in enumerate(messages)
            if message["type"] == "chunk" and message["clip"] == clip
        )
        assert init_position < chunk_position

    offset = 124 / 24
    middle = next(chunk for chunk in chunks if chunk["pts"] == offset)
    assert middle["prompt"] == prompts[1]
    assert middle["clip"] == 1
    assert middle["timestamp_offset"] == offset

    middle_init = next(
        message
        for message in messages
        if message["type"] == "init" and message["clip"] == 1
    )
    assert middle_init["timestamp_offset"] == offset
    media = base64.b64decode(middle_init["init_b64"])
    media += base64.b64decode(middle["video_b64"])
    with av.open(io.BytesIO(media)) as container:
        first_packet = next(
            packet
            for packet in container.demux(video=0)
            if packet.pts is not None
        )
    assert float(first_packet.pts * first_packet.time_base) == 0

    disabled = ServerSettings(
        jobs_dir=tmp_path / "disabled", optimizations=(), stream_enabled=False
    )
    disabled_runner = BlockingSecondRunner()
    disabled_runner.release_second.set()
    with TestClient(create_app(disabled, lambda: disabled_runner)) as client:
        response = client.post(
            "/v1/demos",
            json=_demo_payload(schedule_start=0, schedule_end=0),
        )
        assert response.status_code == 202
        demo_id = response.json()["id"]
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/v1/demos/{demo_id}/ws"):
                pass
