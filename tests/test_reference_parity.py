# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Bitwise golden-latent parity against the recorded reference run.

Goldens are produced on the large-VRAM reference host by
``examples/fl2va_smoke.py --record-goldens tests/fixtures/goldens`` and are
too large to commit; the test skips when they are absent so repo CI stays
green. When present, the runner must reproduce the recorded latents
bitwise (deterministic seed; the recording's torch version must match).
"""

import os
from pathlib import Path

import pytest
import torch

GOLDENS = Path(__file__).parent / "fixtures" / "goldens" / "fl2va_goldens.pt"


@pytest.mark.skipif(not GOLDENS.is_file(), reason="golden fixtures absent")
def test_reference_runner_reproduces_golden_latents():
    from omni_infinity.runner import ReferenceRunner

    payload = torch.load(GOLDENS, weights_only=False)
    assert (
        payload["torch_version"] == torch.__version__
    ), "goldens recorded on a different torch; re-record on this stack"

    runner = ReferenceRunner.from_pretrained(
        os.environ.get("OMNI_H3_CHECKPOINT", "MiniMaxAI/MiniMax-H3")
    )
    result = runner.generate(
        payload["prompt"],
        seed=payload["seed"],
        num_inference_steps=payload["steps"],
        resolution=payload["resolution"],
        num_frames=payload["frames"],
    )

    assert torch.equal(result.latents.cpu(), payload["latents"])
    if payload["audio_latents"] is not None:
        assert torch.equal(result.audio_latents.cpu(), payload["audio_latents"])
