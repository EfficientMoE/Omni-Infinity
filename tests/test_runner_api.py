# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import pytest
import torch

from omni_infinity.runner import (
    GenerationResult,
    ReferenceRunner,
    resolve_resolution,
)


def test_resolutions_cover_rfc_targets():
    assert resolve_resolution("256p") == (256, 256)
    assert resolve_resolution("768p") == (768, 768)
    with pytest.raises(ValueError, match="unknown resolution"):
        resolve_resolution("1080p")


def test_diffusers_ships_h3_modular_pipeline():
    diffusers = pytest.importorskip("diffusers")
    for name in (
        "MiniMaxH3ModularPipeline",
        "MiniMaxH3Transformer3DModel",
        "AutoencoderKLMiniMaxH3",
    ):
        assert hasattr(diffusers, name), name


def test_h3_blocks_accept_runner_call_parameters():
    pytest.importorskip("diffusers")
    from diffusers import MiniMaxH3Blocks

    inputs = {
        spec.name
        for block in MiniMaxH3Blocks().sub_blocks.values()
        for spec in getattr(block, "inputs", [])
    }
    for name in (
        "prompt",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "generator",
        "output_type",
    ):
        assert name in inputs, name


def test_generate_maps_parameters_into_pipeline_call():
    calls = {}

    class FakeState:
        values = {
            "videos": ["video"],
            "audios": ["audio"],
            "latents": torch.zeros(1),
            "audio_latents": torch.zeros(2),
        }

    class FakePipeline:
        def __call__(self, **kwargs):
            calls.update(kwargs)
            return FakeState()

    result = ReferenceRunner(FakePipeline()).generate(
        "a red ball bouncing",
        seed=7,
        num_inference_steps=8,
        resolution="256p",
        num_frames=8,
    )

    assert isinstance(result, GenerationResult)
    assert calls["prompt"] == "a red ball bouncing"
    assert (calls["height"], calls["width"]) == (256, 256)
    assert calls["num_frames"] == 8
    assert calls["num_inference_steps"] == 8
    assert calls["output_type"] == "np"
    assert isinstance(calls["generator"], torch.Generator)
    assert calls["generator"].initial_seed() == 7
    assert result.videos == ["video"]
    assert torch.equal(result.audio_latents, torch.zeros(2))
