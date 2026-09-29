# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from omni_infinity.serve.models import GenerationRequest, JobStatus, Progress
from omni_infinity.serve.store import (
    InvalidProgress,
    InvalidTransition,
    JobNotFound,
    JobStore,
)


@pytest.fixture
def request_body():
    return GenerationRequest(type="fl2va", prompt="a red ball bouncing")


def test_progress_rejects_completed_steps_above_the_total():
    with pytest.raises(ValidationError, match="cannot exceed total steps"):
        Progress(completed_steps=9, total_steps=8)


def test_progress_requires_a_running_job_and_an_in_range_step(
    tmp_path, request_body
):
    store = JobStore(tmp_path)
    record = store.create(request_body)
    with pytest.raises(InvalidTransition, match="running"):
        store.update_progress(record.id, 0, 8)

    store.transition(record.id, JobStatus.RUNNING)
    with pytest.raises(InvalidProgress, match="between zero and total"):
        store.update_progress(record.id, -1, 8)
    with pytest.raises(InvalidProgress, match="between zero and total"):
        store.update_progress(record.id, 9, 8)

    done = store.update_progress(record.id, 8, 8)
    assert done.progress.completed_steps == 8
    assert done.progress.percent == 100.0


def test_failed_transition_truncates_the_error_and_defaults_the_message(
    tmp_path, request_body
):
    store = JobStore(tmp_path)
    missing = store.create(request_body)
    store.transition(missing.id, JobStatus.RUNNING)
    defaulted = store.transition(missing.id, JobStatus.FAILED)
    assert defaulted.error == "job failed"
    assert defaulted.finished_at is not None

    long = store.create(request_body)
    store.transition(long.id, JobStatus.RUNNING)
    failed = store.transition(long.id, JobStatus.FAILED, error="e" * 5000)
    assert failed.error == "e" * 4096


def test_queued_job_cannot_succeed(tmp_path, request_body):
    store = JobStore(tmp_path)
    record = store.create(request_body)
    with pytest.raises(InvalidTransition, match="queued to succeeded"):
        store.transition(record.id, JobStatus.SUCCEEDED)


def test_cancel_from_running_does_not_invent_an_error(tmp_path, request_body):
    store = JobStore(tmp_path)
    record = store.create(request_body)
    store.transition(record.id, JobStatus.RUNNING)
    cancelled = store.transition(record.id, JobStatus.CANCELLED)
    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.error is None
    assert cancelled.finished_at is not None
    assert cancelled.progress.completed_steps == 0


def test_lookup_rejects_ids_that_are_not_32_hex_characters(
    tmp_path, request_body
):
    store = JobStore(tmp_path)
    store.create(request_body)
    with pytest.raises(JobNotFound):
        store.get("not-a-job")
    with pytest.raises(ValueError, match="32 lowercase hexadecimal"):
        store.job_dir("NOTHEX")


def test_recovery_ignores_stray_directories_and_queued_jobs(
    tmp_path, request_body
):
    store = JobStore(tmp_path)
    stray = tmp_path / "notes"
    stray.mkdir()
    (stray / "job.json").write_text("{}")
    queued = store.create(request_body)

    assert store.recover_interrupted() == []
    assert store.get(queued.id).status == JobStatus.QUEUED
