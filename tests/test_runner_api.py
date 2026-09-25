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
            "audio": torch.zeros(2, 4),
            "sampling_rate": 48000,
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
    assert result.sampling_rate == 48000
    assert torch.equal(result.audio, torch.zeros(2, 4))
    assert torch.equal(result.audio_latents, torch.zeros(2))


def test_generate_reports_each_denoising_step_and_removes_hook():
    calls = {}

    class FakeTransformer(torch.nn.Module):
        def forward(self, value):
            return value

    class FakeState:
        values = {
            "videos": ["video"],
            "audio": torch.zeros(2, 4),
            "sampling_rate": 48000,
            "latents": torch.zeros(1),
            "audio_latents": torch.zeros(2),
        }

    class FakePipeline:
        def __init__(self):
            self.transformer = FakeTransformer()

        def __call__(self, **kwargs):
            calls.update(kwargs)
            for _ in range(kwargs["num_inference_steps"]):
                self.transformer(torch.ones(1))
            return FakeState()

    pipeline = FakePipeline()
    progress = []
    first = object()
    last = object()
    ReferenceRunner(pipeline).generate(
        "prompt",
        num_inference_steps=3,
        image=first,
        last_image=last,
        step_callback=lambda completed, total: progress.append(
            (completed, total)
        ),
    )

    assert calls["image"] is first
    assert calls["last_image"] is last
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert pipeline.transformer._forward_hooks == {}


def test_block_streaming_invokes_group_offload_and_requires_offload():
    from omni_infinity import runner as runner_mod

    calls = {}

    class FakeTransformer:
        def enable_group_offload(self, **kwargs):
            calls["group_offload"] = kwargs

    class FakePipeline:
        def __init__(self):
            self.transformer = FakeTransformer()

    runner_mod._enable_block_streaming(
        FakePipeline(), torch.device("cuda"), blocks_per_group=1, to_disk=None
    )
    assert calls["group_offload"]["offload_type"] == "block_level"
    assert calls["group_offload"]["num_blocks_per_group"] == 1
    assert calls["group_offload"]["use_stream"] is True
    assert calls["group_offload"]["offload_device"] == torch.device("cpu")


def test_stream_text_encoder_invokes_group_offloading():
    from omni_infinity import runner as runner_mod

    calls = {}

    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(4)]
            )
            self.visual = torch.nn.Sequential(torch.nn.Linear(4, 4))

    class FakePipeline:
        def __init__(self):
            self.text_encoder = FakeEncoder()

    original = runner_mod._APPLY_GROUP_OFFLOADING

    def fake_apply(module, **kwargs):
        calls.setdefault("applications", []).append((module, kwargs))

    runner_mod._APPLY_GROUP_OFFLOADING = fake_apply
    pipeline = FakePipeline()
    try:
        runner_mod._stream_text_encoder(pipeline, torch.device("cuda"))
    finally:
        runner_mod._APPLY_GROUP_OFFLOADING = original

    applications = calls["applications"]
    assert [module for module, _ in applications] == [
        pipeline.text_encoder.model,
        pipeline.text_encoder.visual,
    ]
    for _, kwargs in applications:
        assert kwargs["use_stream"] is True
        assert kwargs["offload_device"] == torch.device("cpu")
        assert kwargs["offload_type"] == "leaf_level"


def test_from_pretrained_threads_fp8_scale_into_store_loader(monkeypatch):
    pytest.importorskip("diffusers")
    from omni_infinity import runner as runner_mod

    captured = {}

    class FakePipeline:
        def load_components(self, **kwargs):
            pass

        def update_components(self, **kwargs):
            captured["built"] = kwargs

        def to(self, device):
            return self

    class FakeModularPipeline:
        @classmethod
        def from_pretrained(cls, checkpoint, components_manager=None):
            return FakePipeline()

    def fake_load_transformer(
        component_cls,
        checkpoint,
        source,
        torch_dtype,
        component="transformer",
        **kwargs,
    ):
        captured["fp8_mode"] = kwargs.get("fp8_mode")
        return "fake-transformer"

    # from_pretrained imports from diffusers / omni_infinity.store at call time,
    # so patching the source-module attributes wins.
    monkeypatch.setattr(
        "diffusers.MiniMaxH3ModularPipeline", FakeModularPipeline, raising=False
    )
    monkeypatch.setattr(
        "omni_infinity.store.StoreComponentSource",
        lambda store_dir: object(),
        raising=False,
    )
    monkeypatch.setattr(
        "omni_infinity.store.load_transformer_with_adaln_cache",
        fake_load_transformer,
        raising=False,
    )
    monkeypatch.setattr(runner_mod, "_h3_component_class", lambda name: object)

    runner_mod.ReferenceRunner.from_pretrained(
        checkpoint="dummy",
        device="cpu",
        offload=False,
        components=("transformer",),
        store_dir="/tmp/fake-store",
        store_components=("transformer",),
        transformer_fp8=True,
        fp8_scale="per_row",
    )

    assert captured["fp8_mode"] == "per_row"
    assert captured["built"]["transformer"] == "fake-transformer"
