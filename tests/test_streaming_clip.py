# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from omni_infinity.runner import GenerationResult
from omni_infinity.streaming import ActionCue, ClipChunker, StreamRequest


@pytest.fixture
def artifact_result():
    frames = np.zeros((6, 16, 16, 3), dtype=np.float32)
    frames[:, :, :, 0] = np.linspace(0.0, 1.0, 6)[:, None, None]
    return GenerationResult(
        videos=[frames],
        audio=torch.zeros(1, 2, 8000, dtype=torch.float32),
        sampling_rate=48000,
        latents=None,
        audio_latents=None,
    )


def _short_request(**kwargs):
    values = {
        "type": "fl2va",
        "prompt": "a red ball bouncing",
        "num_frames": 6,
        "action_script": [ActionCue(t=0.0, action="wait", instruction="hold")],
    }
    values.update(kwargs)
    return StreamRequest.model_construct(**values)


def test_clip_chunker_finishes_generate_before_the_first_yield(artifact_result):
    seen = {}

    class FakeRunner:
        def generate(self, prompt, *, step_callback, **kwargs):
            seen["kwargs"] = kwargs
            step_callback(
                kwargs["num_inference_steps"], kwargs["num_inference_steps"]
            )
            seen["returned"] = True
            return artifact_result

    request = _short_request()
    chunker = ClipChunker(FakeRunner(), arch="h3-dense", chunk_frames=4)
    iterator = chunker.iter_chunks(request)
    first = next(iterator)
    rest = list(iterator)
    chunks = [first, *rest]

    assert seen["returned"] is True
    assert seen["kwargs"]["num_inference_steps"] == 8
    assert seen["kwargs"]["resolution"] == "256p"
    assert seen["kwargs"]["num_frames"] == 6
    assert chunker.result is artifact_result
    assert chunker.init
    assert first.prompt == "a red ball bouncing"
    assert first.instruction == "hold"
    assert first.action == "wait"
    assert chunks[-1].done is True
    assert all(chunk.keyframe and chunk.audio_bytes is None for chunk in chunks)


def test_clip_chunker_rejects_vdn_below_768p(artifact_result):
    class FakeRunner:
        def generate(self, *args, **kwargs):
            raise AssertionError("generate must not run")

    request = _short_request(resolution="256p")
    chunker = ClipChunker(FakeRunner(), arch="vdn-hybrid", chunk_frames=4)
    with pytest.raises(ValueError, match="768p"):
        next(chunker.iter_chunks(request))


@pytest.mark.parametrize("transpose", [False, True])
def test_clip_chunker_normalizes_unbatched_stereo_audio(
    artifact_result, transpose
):
    audio = artifact_result.audio.squeeze(0)
    artifact_result.audio = audio.transpose(0, 1) if transpose else audio

    class FakeRunner:
        def generate(self, *args, **kwargs):
            return artifact_result

    chunker = ClipChunker(FakeRunner(), arch="h3-dense", chunk_frames=4)
    chunks = list(chunker.iter_chunks(_short_request()))
    assert chunks[-1].done is True
