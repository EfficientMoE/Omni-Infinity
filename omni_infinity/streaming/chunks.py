# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from omni_infinity.serve.models import GenerationRequest


@dataclass(frozen=True)
class MediaChunk:
    index: int
    pts: float
    duration: float
    keyframe: bool
    video_bytes: bytes | None
    audio_bytes: bytes | None
    prompt: str
    instruction: str | None = None
    action: str | None = None
    done: bool = False


class ActionCue(BaseModel):
    t: float = Field(ge=0)
    action: str = Field(min_length=1, max_length=64)
    instruction: str = Field(min_length=1, max_length=2000)


class StreamRequest(GenerationRequest):
    action_script: list[ActionCue] = Field(default_factory=list, max_length=64)
    source: Literal["clip", "native"] = "clip"


class StreamInput(BaseModel):
    type: Literal["input"] = "input"
    key: str | None = None
    down: bool = True
    action: str | None = None
    prompt: str | None = None


class ChunkSource(Protocol):
    def iter_chunks(
        self,
        request: StreamRequest,
        *,
        step_callback: Callable[[int, int], None] | None = None,
        image: Any = None,
        last_image: Any = None,
    ) -> Iterator[MediaChunk]: ...


def active_cue(cues: Sequence[MediaChunk], time: float) -> MediaChunk | None:
    for cue in cues:
        if cue.pts <= time < cue.pts + cue.duration:
            return cue
    return None
