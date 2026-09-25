# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from omni_infinity.serve.models import (
    TERMINAL_STATUSES,
    ArtifactMetadata,
    GenerationRequest,
    JobRecord,
    JobStatus,
    Progress,
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
