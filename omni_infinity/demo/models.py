# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

from omni_infinity.demo.schedule import PRESETS, segment_count
from omni_infinity.serve.models import JobRecord, JobResponse


class DemoRequest(BaseModel):
    prompts: list[
        Annotated[str, StringConstraints(min_length=1, max_length=20000)]
    ]
    duration: Literal["15s", "1min", "2min", "5min"]
    schedule_start: int = Field(default=1, ge=0)
    schedule_end: int = Field(default=5, ge=0)
    model_arch: Literal["h3-dense", "vdn-hybrid"] = "h3-dense"
    optimizations: list[
        Literal[
            "adaln-host-cache",
            "fp8",
            "block-stream",
            "text-encoder-stream",
        ]
    ] = Field(default_factory=list)
    seed: int = 0
    num_inference_steps: int = Field(default=8, ge=1, le=100)
    resolution: Literal["256p", "512p", "768p"] = "256p"
    first_frame_base64: str | None = None

    @model_validator(mode="after")
    def validate_schedule_range(self):
        if self.schedule_end < self.schedule_start:
            raise ValueError("schedule end is before schedule start")
        return self

    @model_validator(mode="after")
    def validate_prompt_count(self):
        if self.duration and self.prompts is not None:
            expected = segment_count(PRESETS[self.duration])
            got = len(self.prompts)
            if got != expected:
                raise ValueError(
                    f"duration {self.duration} requires {expected} "
                    f"prompts, got {got}"
                )
        return self

    @model_validator(mode="after")
    def unique_optimizations(self):
        if len(set(self.optimizations)) != len(self.optimizations):
            raise ValueError("optimization names must be unique")
        return self


class DemoRecord(JobRecord):
    request: DemoRequest


class DemoResponse(JobResponse):
    request: DemoRequest
