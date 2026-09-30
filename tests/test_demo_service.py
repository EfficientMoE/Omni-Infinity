# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from omni_infinity.demo.models import DemoRequest
from omni_infinity.demo.service import DemoService
from omni_infinity.demo.stitch import last_frame_image
from omni_infinity.runner import GenerationResult

SAMPLING_RATE = 48000
SEGMENT_FRAMES = 124


@dataclass
class RecordedCall:
    prompt: str
    seed: int
    num_frames: int
    image: Any
    last_image: Any
    num_inference_steps: Any = None
    resolution: Any = None
    num_evaluations: Any = None


@dataclass
class FakeRunner:
    calls: list[RecordedCall] = field(default_factory=list)

    def generate(
        self,
        prompt,
        *,
        seed,
        num_frames,
        image,
        last_image,
        num_inference_steps=None,
        resolution=None,
        num_evaluations=None,
        step_callback=None,
    ):
        index = len(self.calls)
        self.calls.append(
            RecordedCall(
                prompt=prompt,
                seed=seed,
                num_frames=num_frames,
                image=image,
                last_image=last_image,
                num_inference_steps=num_inference_steps,
                resolution=resolution,
                num_evaluations=num_evaluations,
            )
        )
        frames = np.full(
            (num_frames, 2, 2, 3), (index + 1) / 10.0, dtype=np.float32
        )
        audio = torch.zeros(2, num_frames * SAMPLING_RATE // 24)
        steps = num_inference_steps or num_evaluations or 1
        if step_callback is not None:
            for completed in range(1, steps + 1):
                step_callback(completed, steps)
        return GenerationResult(
            videos=[frames],
            audio=audio,
            sampling_rate=SAMPLING_RATE,
            latents=None,
            audio_latents=None,
        )


def _request(**overrides) -> DemoRequest:
    fields = {
        "prompts": ["opening", "middle", "close"],
        "duration": "15s",
        "seed": 7,
        "num_inference_steps": 8,
    }
    fields.update(overrides)
    return DemoRequest(**fields)


def test_run_sequential_handoff_and_stitch():
    runner = FakeRunner()
    service = DemoService(runner, "h3-dense")
    request = _request()
    first = Image.new("RGB", (2, 2), (255, 0, 0))

    progress: list[tuple[int, int]] = []
    result = service.run(
        request, first, step_callback=lambda c, t: progress.append((c, t))
    )

    assert len(runner.calls) == 3
    assert [c.seed for c in runner.calls] == [7, 8, 9]
    assert [c.num_frames for c in runner.calls] == [124, 124, 124]
    assert all(c.last_image is None for c in runner.calls)
    assert [c.resolution for c in runner.calls] == ["256p"] * 3
    assert [c.num_inference_steps for c in runner.calls] == [8, 8, 8]
    assert runner.calls[0].image is first

    expected_handoff = last_frame_image(
        np.full((SEGMENT_FRAMES, 2, 2, 3), 1 / 10.0, dtype=np.float32)
    )
    assert np.array_equal(
        np.asarray(runner.calls[1].image), np.asarray(expected_handoff)
    )
    expected_handoff_2 = last_frame_image(
        np.full((SEGMENT_FRAMES, 2, 2, 3), 2 / 10.0, dtype=np.float32)
    )
    assert np.array_equal(
        np.asarray(runner.calls[2].image), np.asarray(expected_handoff_2)
    )

    assert result.videos[0].shape[0] == 360

    # completed = i * num_inference_steps + segment_completed against a
    # total of segment_count * num_inference_steps.
    total = 3 * 8
    assert progress[-1] == (total, total)
    assert [c for c, _ in progress] == list(range(1, total + 1))
    assert all(t == total for _, t in progress)


def test_vdn_hybrid_passes_num_evaluations_without_resolution():
    runner = FakeRunner()
    service = DemoService(runner, "vdn-hybrid")
    request = _request(model_arch="vdn-hybrid", resolution="768p")
    first = Image.new("RGB", (2, 2), (255, 0, 0))

    result = service.run(request, first, step_callback=lambda c, t: None)

    assert len(runner.calls) == 3
    assert [c.num_evaluations for c in runner.calls] == [8, 8, 8]
    assert all(c.resolution is None for c in runner.calls)
    assert all(c.num_inference_steps is None for c in runner.calls)
    assert result.videos[0].shape[0] == 360


def test_vdn_hybrid_requires_768p():
    runner = FakeRunner()
    service = DemoService(runner, "vdn-hybrid")
    request = _request(model_arch="vdn-hybrid", resolution="256p")
    first = Image.new("RGB", (2, 2), (255, 0, 0))

    with pytest.raises(ValueError, match="768p"):
        service.run(request, first, step_callback=lambda c, t: None)

    assert runner.calls == []
