# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
)


class GenerationRequest(BaseModel):
    type: Literal["fl2va", "ref2va"]
    prompt: str = Field(min_length=1, max_length=20000)
    model_arch: Literal["h3-dense", "vdn-hybrid"] = "h3-dense"
    optimizations: list[
        Literal[
            "adaln-host-cache", "fp8", "block-stream", "text-encoder-stream"
        ]
    ] = Field(default_factory=list)
    seed: int = 0
    num_inference_steps: int = Field(default=8, ge=1, le=100)
    resolution: Literal["256p", "512p", "768p"] = "256p"
    num_frames: int = Field(default=120, ge=120, le=360)
    first_frame_base64: str | None = None
    last_frame_base64: str | None = None
    references_base64: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_unique_optimizations(self):
        if len(set(self.optimizations)) != len(self.optimizations):
            raise ValueError("optimization names must be unique")
        return self


class Progress(BaseModel):
    completed_steps: int = Field(ge=0)
    total_steps: int = Field(gt=0)
    percent: float = 0.0

    @model_validator(mode="after")
    def calculate_percent(self):
        if self.completed_steps > self.total_steps:
            raise ValueError("completed steps cannot exceed total steps")
        self.percent = self.completed_steps / self.total_steps * 100.0
        return self


class ArtifactMetadata(BaseModel):
    video_url: str
    audio_channels: Literal[2] = 2
    sampling_rate: int = Field(gt=0)


class JobRecord(BaseModel):
    id: str
    request: GenerationRequest
    status: JobStatus
    progress: Progress
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = Field(default=None, max_length=4096)
    artifacts: ArtifactMetadata | None = None


class JobResponse(BaseModel):
    id: str
    request: GenerationRequest
    status: JobStatus
    progress: Progress
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    artifacts: ArtifactMetadata | None = None

    @classmethod
    def from_record(cls, record: JobRecord) -> JobResponse:
        return cls.model_validate(record.model_dump())
