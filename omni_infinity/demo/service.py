# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable

from PIL import Image

from omni_infinity.demo.models import DemoRequest
from omni_infinity.demo.schedule import (
    PRESETS,
    SEGMENT_FRAMES,
    playback_frames,
    segment_count,
)
from omni_infinity.demo.stitch import last_frame_image, stitch_results
from omni_infinity.runner import GenerationResult

StepCallback = Callable[[int, int], None]


class DemoService:
    def __init__(self, runner, model_arch: str):
        self.runner = runner
        self.model_arch = model_arch

    def run(
        self,
        request: DemoRequest,
        first: Image.Image,
        *,
        step_callback: StepCallback | None = None,
    ) -> GenerationResult:
        results: list[GenerationResult] = []
        frame: Image.Image = first
        for i in range(len(request.prompts)):
            result = self._segment(
                i, request, frame, step_callback=step_callback
            )
            results.append(result)
            frame = last_frame_image(result.videos[0])
        return stitch_results(
            results,
            playback_frames=playback_frames(PRESETS[request.duration]),
        )

    def _segment(
        self,
        i: int,
        request: DemoRequest,
        image: Image.Image,
        *,
        step_callback: StepCallback | None = None,
    ) -> GenerationResult:
        total = segment_count(PRESETS[request.duration])
        total_steps = total * request.num_inference_steps

        inner: StepCallback | None = None
        if step_callback is not None:

            def inner(completed: int, _segment_total: int) -> None:
                step_callback(
                    i * request.num_inference_steps + completed, total_steps
                )

        common = {
            "seed": request.seed + i,
            "num_frames": SEGMENT_FRAMES,
            "image": image,
            "last_image": None,
            "step_callback": inner,
        }
        if self.model_arch == "h3-dense":
            return self.runner.generate(
                request.prompts[i],
                num_inference_steps=request.num_inference_steps,
                resolution=request.resolution,
                **common,
            )
        if request.resolution != "768p":
            raise ValueError("vdn-hybrid requires the 768p canvas")
        return self.runner.generate(
            request.prompts[i],
            num_evaluations=request.num_inference_steps,
            **common,
        )
