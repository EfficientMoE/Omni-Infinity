# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import av
import httpx
import numpy as np
import pytest
import soundfile
import torch
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from omni_infinity.registry import resolve_profile
from omni_infinity.runner import GenerationResult
from omni_infinity.serve import artifacts as artifact_module
from omni_infinity.serve.app import ServerSettings, create_app
from omni_infinity.serve.artifacts import write_artifacts
from omni_infinity.serve.models import (
    TERMINAL_STATUSES,
    ArtifactMetadata,
    GenerationRequest,
    JobRecord,
    JobStatus,
    Progress,
)
from omni_infinity.serve.service import (
    JobService,
    ProfileConflict,
    Ref2VANotImplemented,
)
from omni_infinity.serve.store import (
    CorruptJob,
    InvalidTransition,
    JobNotFound,
    JobStore,
    JobStoreError,
)


@pytest.fixture
def fl2va_request():
    return GenerationRequest(type="fl2va", prompt="a red ball bouncing")


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


def test_fl2va_request_defaults(fl2va_request):
    assert fl2va_request.resolution == "256p"
    assert fl2va_request.num_inference_steps == 8
    assert fl2va_request.num_frames == 120


def test_generation_request_rejects_duplicate_optimizations():
    with pytest.raises(
        ValidationError, match="optimization names must be unique"
    ):
        GenerationRequest(
            type="fl2va",
            prompt="prompt",
            optimizations=["fp8", "fp8"],
        )


def test_progress_percent_is_completed_over_total():
    assert Progress(completed_steps=3, total_steps=8).percent == 37.5


def test_terminal_statuses_cannot_return_to_running(fl2va_request):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    record = JobRecord(
        id="0123456789abcdef0123456789abcdef",
        request=fl2va_request,
        status=JobStatus.SUCCEEDED,
        progress=Progress(completed_steps=8, total_steps=8),
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=now,
    )

    assert record.status in TERMINAL_STATUSES
    assert JobStatus.RUNNING not in TERMINAL_STATUSES
    assert (
        record.model_dump(mode="json")["created_at"] == "2026-09-24T12:00:00Z"
    )


def test_job_store_persists_atomic_state_machine(tmp_path, fl2va_request):
    store = JobStore(tmp_path)
    record = store.create(fl2va_request)
    job_dir = tmp_path / record.id
    assert (job_dir / "job.json").is_file()
    assert not (job_dir / "job.json.tmp").exists()

    running = store.transition(record.id, JobStatus.RUNNING)
    assert running.started_at is not None
    progressed = store.update_progress(record.id, 3, 8)
    assert progressed.progress.percent == 37.5
    succeeded = store.transition(
        record.id,
        JobStatus.SUCCEEDED,
        artifacts=ArtifactMetadata(
            video_url=f"/v1/jobs/{record.id}/artifacts",
            audio_channels=2,
            sampling_rate=48000,
        ),
    )
    assert succeeded.finished_at is not None
    assert succeeded.progress.completed_steps == 8
    with pytest.raises(InvalidTransition):
        store.transition(record.id, JobStatus.RUNNING)


def test_job_store_reads_persisted_record_and_rejects_missing(
    tmp_path, fl2va_request
):
    record = JobStore(tmp_path).create(fl2va_request)
    fresh = JobStore(tmp_path)
    assert fresh.get(record.id) == record
    with pytest.raises(JobNotFound):
        fresh.get("f" * 32)


def test_job_store_reports_corrupt_json(tmp_path, fl2va_request):
    store = JobStore(tmp_path)
    record = store.create(fl2va_request)
    (tmp_path / record.id / "job.json").write_text("not json")
    with pytest.raises(CorruptJob):
        store.get(record.id)


def test_job_store_recovers_interrupted_running_jobs(tmp_path, fl2va_request):
    store = JobStore(tmp_path)
    running = store.create(fl2va_request)
    store.transition(running.id, JobStatus.RUNNING)
    terminal = store.create(fl2va_request)
    store.transition(terminal.id, JobStatus.CANCELLED)

    recovered = JobStore(tmp_path).recover_interrupted()

    assert recovered == [running.id]
    failed = store.get(running.id)
    assert failed.status == JobStatus.FAILED
    assert failed.error == "server restarted while job was running"
    assert store.get(terminal.id).status == JobStatus.CANCELLED


@pytest.mark.parametrize("name", ["", ".", "..", "../escape.png"])
def test_job_store_rejects_unsafe_input_names(tmp_path, name):
    store = JobStore(tmp_path)
    with pytest.raises(ValueError, match="filename"):
        store.input_path("a" * 32, name)


def test_job_store_recovery_skips_corrupt_jobs(tmp_path, fl2va_request):
    store = JobStore(tmp_path)
    corrupt = store.create(fl2va_request)
    (store.job_dir(corrupt.id) / "job.json").write_text("not json")
    running = store.create(fl2va_request)
    store.transition(running.id, JobStatus.RUNNING)

    assert JobStore(tmp_path).recover_interrupted() == [running.id]
    assert store.get(running.id).status == JobStatus.FAILED


def test_job_store_progress_errors_share_store_error_base(
    tmp_path, fl2va_request
):
    store = JobStore(tmp_path)
    record = store.create(fl2va_request)
    store.transition(record.id, JobStatus.RUNNING)
    with pytest.raises(JobStoreError):
        store.update_progress(record.id, 1, 9)


def test_artifact_writer_creates_atomic_stereo_mp4_and_wav(
    tmp_path, artifact_result
):
    metadata = write_artifacts(artifact_result, tmp_path)

    assert (tmp_path / "output.wav").is_file()
    assert (tmp_path / "output.mp4").is_file()
    assert not (tmp_path / "output.partial.wav").exists()
    assert not (tmp_path / "output.partial.mp4").exists()
    assert soundfile.info(tmp_path / "output.wav").channels == 2
    with av.open(tmp_path / "output.mp4") as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.streams.audio[0].layout.name == "stereo"
    assert metadata.audio_channels == 2
    assert metadata.sampling_rate == 48000


def test_artifact_writer_rejects_mono_audio(tmp_path, artifact_result):
    artifact_result.audio = torch.zeros(1, 8000)
    with pytest.raises(ValueError, match="expected stereo audio"):
        write_artifacts(artifact_result, tmp_path)
    assert not (tmp_path / "output.wav").exists()
    assert not (tmp_path / "output.mp4").exists()


def test_artifact_writer_rolls_back_when_video_encoding_fails(
    tmp_path, artifact_result, monkeypatch
):
    def fail_encode(*args, **kwargs):
        assert not (tmp_path / "output.wav").exists()
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(
        "diffusers.utils.export_utils.encode_video", fail_encode
    )
    with pytest.raises(RuntimeError, match="encoder failed"):
        artifact_module.write_artifacts(artifact_result, tmp_path)
    for name in (
        "output.wav",
        "output.mp4",
        "output.partial.wav",
        "output.partial.mp4",
    ):
        assert not (tmp_path / name).exists()


def _wait_for_terminal(store, job_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = store.get(job_id)
        if record.status in TERMINAL_STATUSES:
            return record
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def test_job_service_serializes_jobs_and_persists_artifacts(
    tmp_path, artifact_result
):
    class FakeRunner:
        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def generate(
            self, prompt, *, step_callback, num_inference_steps, **kwargs
        ):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                for completed in range(1, num_inference_steps + 1):
                    time.sleep(0.01)
                    step_callback(completed, num_inference_steps)
                return artifact_result
            finally:
                with self.lock:
                    self.active -= 1

    runner = FakeRunner()
    store = JobStore(tmp_path)
    service = JobService(runner, resolve_profile("h3-dense", []), store)
    try:
        request = GenerationRequest(type="fl2va", prompt="prompt")
        first = service.submit(request)
        second = service.submit(request)
        assert store.get(second.id).status == JobStatus.QUEUED

        first_done = _wait_for_terminal(store, first.id)
        second_done = _wait_for_terminal(store, second.id)

        assert first_done.status == JobStatus.SUCCEEDED
        assert second_done.status == JobStatus.SUCCEEDED
        assert runner.max_active == 1
        assert first_done.progress.completed_steps == 8
        assert second_done.progress.completed_steps == 8
        assert store.video_path(first.id).is_file()
        assert store.video_path(second.id).is_file()
    finally:
        service.shutdown()


def test_job_service_rejects_profile_conflicts(tmp_path):
    service = JobService(
        object(), resolve_profile("h3-dense", []), JobStore(tmp_path)
    )
    try:
        request = GenerationRequest(
            type="fl2va", prompt="prompt", model_arch="vdn-hybrid"
        )
        with pytest.raises(ProfileConflict):
            service.submit(request)
    finally:
        service.shutdown()


def test_job_service_persists_runner_failures(tmp_path):
    class FailingRunner:
        def generate(self, *args, **kwargs):
            raise RuntimeError("GPU exploded")

    store = JobStore(tmp_path)
    service = JobService(
        FailingRunner(), resolve_profile("h3-dense", []), store
    )
    try:
        record = service.submit(
            GenerationRequest(type="fl2va", prompt="prompt")
        )
        failed = _wait_for_terminal(store, record.id)
        assert failed.status == JobStatus.FAILED
        assert failed.error == "RuntimeError: GPU exploded"
    finally:
        service.shutdown()


def test_job_service_rejects_ref2va_before_queueing(tmp_path):
    store = JobStore(tmp_path)
    service = JobService(object(), resolve_profile("h3-dense", []), store)
    try:
        with pytest.raises(Ref2VANotImplemented):
            service.submit(GenerationRequest(type="ref2va", prompt="prompt"))
        assert list(tmp_path.iterdir()) == []
    finally:
        service.shutdown()


def test_job_service_shutdown_cancels_queued_jobs(tmp_path, artifact_result):
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
            return artifact_result

    runner = BlockingRunner()
    store = JobStore(tmp_path)
    service = JobService(runner, resolve_profile("h3-dense", []), store)
    request = GenerationRequest(type="fl2va", prompt="prompt")
    first = service.submit(request)
    assert runner.started.wait(timeout=2)
    second = service.submit(request)

    shutdown = threading.Thread(target=service.shutdown)
    shutdown.start()
    time.sleep(0.05)
    runner.release.set()
    shutdown.join(timeout=5)

    assert not shutdown.is_alive()
    assert store.get(first.id).status == JobStatus.SUCCEEDED
    assert store.get(second.id).status == JobStatus.CANCELLED


def _poll_http_job(client, job_id, timeout=5):
    deadline = time.monotonic() + timeout
    observed = []
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        observed.append(body["progress"]["completed_steps"])
        if body["status"] in {"succeeded", "failed", "cancelled"}:
            return body, observed
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def test_http_job_lifecycle_artifact_and_single_runner_load(
    tmp_path, artifact_result
):
    calls = {"factory": 0, "generate": 0}

    class FakeRunner:
        def generate(
            self, prompt, *, step_callback, num_inference_steps, **kwargs
        ):
            calls["generate"] += 1
            for completed in range(1, num_inference_steps + 1):
                time.sleep(0.005)
                step_callback(completed, num_inference_steps)
            return artifact_result

    runner = FakeRunner()

    def factory():
        calls["factory"] += 1
        return runner

    settings = ServerSettings(jobs_dir=tmp_path, optimizations=())
    with TestClient(create_app(settings, factory)) as client:
        response = client.post(
            "/v1/jobs", json={"type": "fl2va", "prompt": "prompt"}
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert response.headers["location"] == f"/v1/jobs/{job_id}"
        assert response.json()["status"] in {"queued", "running"}

        finished, observed = _poll_http_job(client, job_id)
        assert finished["status"] == "succeeded"
        assert finished["progress"] == {
            "completed_steps": 8,
            "total_steps": 8,
            "percent": 100.0,
        }
        assert observed == sorted(observed)

        artifact = client.get(f"/v1/jobs/{job_id}/artifacts")
        assert artifact.status_code == 200
        assert artifact.headers["content-type"] == "video/mp4"
        downloaded = tmp_path / "downloaded.mp4"
        downloaded.write_bytes(artifact.content)
        with av.open(downloaded) as container:
            assert len(container.streams.video) == 1
            assert len(container.streams.audio) == 1
            assert container.streams.audio[0].layout.name == "stereo"

        second = client.post(
            "/v1/jobs", json={"type": "fl2va", "prompt": "again"}
        )
        assert second.status_code == 202
        second_done, _ = _poll_http_job(client, second.json()["id"])
        assert second_done["status"] == "succeeded"

        assert client.get(f"/v1/jobs/{'f' * 32}").status_code == 404
        assert client.post("/v1/jobs", json={}).status_code == 422
        ref2va = client.post(
            "/v1/jobs", json={"type": "ref2va", "prompt": "prompt"}
        )
        assert ref2va.status_code == 501
        assert "issue #8" in ref2va.json()["detail"]
        conflict = client.post(
            "/v1/jobs",
            json={
                "type": "fl2va",
                "prompt": "prompt",
                "model_arch": "vdn-hybrid",
            },
        )
        assert conflict.status_code == 409
        invalid_media = client.post(
            "/v1/jobs",
            json={
                "type": "fl2va",
                "prompt": "prompt",
                "first_frame_base64": "not base64!",
            },
        )
        assert invalid_media.status_code == 422

    assert calls == {"factory": 1, "generate": 2}


def test_http_artifact_is_conflict_until_job_succeeds(
    tmp_path, artifact_result
):
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
            return artifact_result

    runner = BlockingRunner()
    settings = ServerSettings(jobs_dir=tmp_path, optimizations=())
    with TestClient(create_app(settings, lambda: runner)) as client:
        response = client.post(
            "/v1/jobs", json={"type": "fl2va", "prompt": "prompt"}
        )
        job_id = response.json()["id"]
        assert runner.started.wait(timeout=2)
        assert client.get(f"/v1/jobs/{job_id}/artifacts").status_code == 409
        runner.release.set()
        finished, _ = _poll_http_job(client, job_id)
        assert finished["status"] == "succeeded"


def _free_local_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.mark.timeout(1800)
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="job API integration needs CUDA"
)
@pytest.mark.skipif(
    os.environ.get("OMNI_JOB_API_GPU") != "1",
    reason="set OMNI_JOB_API_GPU=1 to run the real job API gate",
)
def test_real_fl2va_job_api_returns_muxed_mp4(tmp_path):
    port = _free_local_port()
    env = os.environ.copy()
    env.update(
        {
            "OMNI_JOBS_DIR": str(tmp_path / "jobs"),
            "OMNI_HOST": "127.0.0.1",
            "OMNI_PORT": str(port),
            "OMNI_DEVICE": "cuda",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    log_path = tmp_path / "server.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "omni_infinity.serve"],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("server exited during startup")
            try:
                ready = httpx.get(f"{base_url}/v1/jobs/{'f' * 32}", timeout=5)
                if ready.status_code == 404:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        else:
            raise AssertionError("server did not become ready")

        image_buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color="red").save(image_buffer, format="PNG")
        response = httpx.post(
            f"{base_url}/v1/jobs",
            json={
                "type": "fl2va",
                "prompt": "a red ball bouncing",
                "model_arch": "h3-dense",
                "optimizations": [
                    "adaln-host-cache",
                    "block-stream",
                    "text-encoder-stream",
                ],
                "seed": 0,
                "num_inference_steps": 8,
                "resolution": "256p",
                "num_frames": 120,
                "first_frame_base64": base64.b64encode(
                    image_buffer.getvalue()
                ).decode("ascii"),
            },
            timeout=30,
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        observed = []
        while time.monotonic() < deadline:
            status = httpx.get(
                f"{base_url}/v1/jobs/{job_id}", timeout=30
            ).json()
            observed.append(status["progress"]["completed_steps"])
            if status["status"] == "succeeded":
                break
            if status["status"] in {"failed", "cancelled"}:
                raise AssertionError(status.get("error") or status["status"])
            time.sleep(2)
        else:
            raise AssertionError("job did not finish")
        assert observed == sorted(observed)
        assert status["progress"]["completed_steps"] == 8
        assert status["progress"]["total_steps"] == 8

        artifact_response = httpx.get(
            f"{base_url}/v1/jobs/{job_id}/artifacts", timeout=120
        )
        assert artifact_response.status_code == 200
        artifact = tmp_path / "artifact.mp4"
        artifact.write_bytes(artifact_response.content)
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(artifact),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert probe.returncode == 0, probe.stderr
        streams = json.loads(probe.stdout)["streams"]
        assert [stream["codec_type"] for stream in streams].count("video") == 1
        audio_streams = [
            stream for stream in streams if stream["codec_type"] == "audio"
        ]
        assert len(audio_streams) == 1
        assert int(audio_streams[0]["channels"]) == 2
    except Exception as exc:
        log_handle.flush()
        logs = log_path.read_text(encoding="utf-8")
        raise AssertionError(f"{exc}\nserver logs:\n{logs}") from exc
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
        log_handle.close()
