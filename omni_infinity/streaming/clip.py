# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from omni_infinity.runner import GenerationResult
from omni_infinity.serve.artifacts import _stereo_audio, _video_frames
from omni_infinity.streaming.chunks import MediaChunk, StreamRequest
from omni_infinity.streaming.fragment import MediaFragment, fragment_clip


class ClipChunker:
    def __init__(self, runner: Any, *, arch: str, chunk_frames: int = 24):
        self.runner = runner
        self.arch = arch
        self.chunk_frames = chunk_frames
        self.result: GenerationResult | None = None
        self.init: bytes | None = None

    def iter_chunks(
        self,
        request: StreamRequest,
        *,
        step_callback: Callable[[int, int], None] | None = None,
        image: Any = None,
        last_image: Any = None,
    ) -> Iterator[MediaChunk]:
        if self.arch == "vdn-hybrid" and request.resolution != "768p":
            raise ValueError("vdn-hybrid requires the 768p canvas")

        kwargs = {
            "seed": request.seed,
            "num_frames": request.num_frames,
            "image": image,
            "last_image": last_image,
            "step_callback": step_callback or (lambda _completed, _total: None),
        }
        if self.arch == "vdn-hybrid":
            kwargs["num_evaluations"] = request.num_inference_steps
        else:
            kwargs.update(
                num_inference_steps=request.num_inference_steps,
                resolution=request.resolution,
            )

        self.result = self.runner.generate(request.prompt, **kwargs)
        self.init, fragments = fragment_clip(
            _video_frames(self.result.videos),
            _stereo_audio(self.result.audio),
            int(self.result.sampling_rate or 48_000),
            chunk_frames=self.chunk_frames,
        )

        for fragment in fragments:
            cue = _cue_for_fragment(request, fragment)
            yield MediaChunk(
                index=fragment.index,
                pts=fragment.pts,
                duration=fragment.duration,
                keyframe=fragment.keyframe,
                video_bytes=fragment.video_bytes,
                audio_bytes=None,
                prompt=request.prompt,
                instruction=cue.instruction if cue else None,
                action=cue.action if cue else None,
                done=fragment.index == len(fragments) - 1,
            )


def _cue_for_fragment(request: StreamRequest, fragment: MediaFragment):
    matching = [
        cue
        for cue in request.action_script
        if fragment.pts <= cue.t < fragment.pts + fragment.duration
    ]
    return matching[-1] if matching else None
