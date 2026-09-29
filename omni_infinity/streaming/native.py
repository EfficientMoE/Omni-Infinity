# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import torch

from omni_infinity.runner import GenerationResult
from omni_infinity.streaming.chunks import (
    MediaChunk,
    StreamInput,
    StreamRequest,
)
from omni_infinity.streaming.fragment import fragment_clip


class NativeChunker:
    def __init__(self, *, chunk_frames: int = 2, chunks: int = 2):
        self.chunk_frames = chunk_frames
        self.chunks = chunks
        self.init: bytes | None = None
        self.result: GenerationResult | None = None
        self._pending_input: StreamInput | None = None

    def push_input(self, incoming: StreamInput) -> None:
        self._pending_input = incoming

    def iter_chunks(
        self,
        request: StreamRequest,
        *,
        step_callback: Callable[[int, int], None] | None = None,
        image: Any = None,
        last_image: Any = None,
    ) -> Iterator[MediaChunk]:
        del step_callback, image, last_image
        frame_count = self.chunk_frames * self.chunks
        frames = np.full((frame_count, 16, 16, 3), 0.5, dtype=np.float32)
        sample_rate = 48_000
        audio = torch.zeros(
            (2, frame_count * sample_rate // 24), dtype=torch.float32
        )
        self.result = GenerationResult(
            videos=[frames],
            audio=audio,
            sampling_rate=sample_rate,
            latents=None,
            audio_latents=None,
        )
        self.init, fragments = fragment_clip(
            frames,
            audio,
            sample_rate,
            chunk_frames=self.chunk_frames,
        )

        for fragment in fragments:
            incoming = self._pending_input
            self._pending_input = None
            prompt = request.prompt
            action = instruction = None
            if incoming is not None:
                if incoming.prompt is not None:
                    prompt = incoming.prompt
                if incoming.down and incoming.action is not None:
                    action = instruction = incoming.action
            yield MediaChunk(
                index=fragment.index,
                pts=fragment.pts,
                duration=fragment.duration,
                keyframe=fragment.keyframe,
                video_bytes=fragment.video_bytes,
                audio_bytes=None,
                prompt=prompt,
                instruction=instruction,
                action=action,
                done=fragment.index == len(fragments) - 1,
            )
