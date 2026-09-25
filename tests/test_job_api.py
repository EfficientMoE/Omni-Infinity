# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from omni_infinity.serve.models import (
    TERMINAL_STATUSES,
    GenerationRequest,
    JobRecord,
    JobStatus,
    Progress,
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
