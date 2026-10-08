# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import inspect
from contextlib import contextmanager

import pytest
import torch

from omni_infinity import runner as runner_mod
from omni_infinity.runner import (
    GenerationResult,
    ReferenceRunner,
    resolve_resolution,
)


def test_resolutions_cover_rfc_targets():
    assert resolve_resolution("256p") == (256, 256)
    assert resolve_resolution("512p") == (512, 512)
    assert resolve_resolution("768p") == (768, 768)
    with pytest.raises(ValueError, match="unknown resolution"):
        resolve_resolution("1080p")


def test_cuda_graph_bucket_uses_effective_video_frame_shape():
    assert runner_mod._effective_video_frames(120) == 124
    assert runner_mod._effective_video_frames(124) == 124


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


def test_generate_passes_references_to_modular_pipeline():
    calls = {}

    class FakePipeline:
        def __call__(self, **kwargs):
            calls.update(kwargs)
            return type("State", (), {"values": {}})()

    references = [object()]
    ReferenceRunner(FakePipeline()).generate(
        "animate this subject",
        references=references,
        num_frames=120,
    )
    assert calls["references"] is references
    assert calls["num_frames"] == 120


def test_generate_progress_uses_ref2va_transformer():
    class FakeTransformer(torch.nn.Module):
        def forward(self, value):
            return value

    class FakePipeline:
        def __init__(self):
            self.transformer_ref = FakeTransformer()

        def __call__(self, **kwargs):
            self.transformer_ref(torch.ones(1))
            return type("State", (), {"values": {}})()

    progress = []
    pipeline = FakePipeline()
    ReferenceRunner(pipeline, transformer_component="transformer_ref").generate(
        "animate this subject",
        num_inference_steps=1,
        step_callback=lambda completed, total: progress.append(
            (completed, total)
        ),
    )
    assert progress == [(1, 1)]
    assert pipeline.transformer_ref._forward_hooks == {}


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


def test_compile_blocks_rejects_block_streaming():
    with pytest.raises(
        ValueError, match="compile-blocks is incompatible with block streaming"
    ):
        runner_mod._validate_compile_blocks(
            compile_blocks=True, block_stream_blocks_per_group=1
        )

    runner_mod._validate_compile_blocks(
        compile_blocks=True, block_stream_blocks_per_group=0
    )
    runner_mod._validate_compile_blocks(
        compile_blocks=False, block_stream_blocks_per_group=1
    )


def test_cuda_graph_rejects_block_streaming_but_allows_compile_blocks():
    with pytest.raises(
        ValueError, match="cuda-graph is incompatible with block streaming"
    ):
        runner_mod._validate_cuda_graph(
            cuda_graph=True, block_stream_blocks_per_group=1
        )

    runner_mod._validate_cuda_graph(
        cuda_graph=True, block_stream_blocks_per_group=0
    )
    runner_mod._validate_compile_blocks(
        compile_blocks=True, block_stream_blocks_per_group=0
    )


def test_cuda_graph_registry_profile_binds_reference_runner_api():
    from omni_infinity.registry import resolve_profile

    profile = resolve_profile("h3-dense", ["cuda-graph"])

    bound = inspect.signature(profile.runner.from_pretrained).bind(
        profile.checkpoint, device="cuda", **profile.runner_kwargs
    )
    assert bound.arguments["cuda_graph"] is True


def test_c5_wrapper_installed_later_bypasses_cuda_graph_on_cache_hit():
    from omni_infinity.caches.denoise import (
        DenoiseCacheConfig,
        denoise_step_cache,
    )
    from omni_infinity.cuda_graph import GraphKey

    class FakeTransformer(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 1

    class FakePipeline:
        transformer = FakeTransformer()

    class FakeManager:
        current_bucket = GraphKey(256, 256, 124)

        def __init__(self):
            self.calls = 0

        def try_execute(self, key, function, *args, **kwargs):
            assert key == self.current_bucket
            self.calls += 1
            return None

    manager = FakeManager()
    pipeline = FakePipeline()
    runner_mod._wrap_transformer_with_cuda_graph(pipeline, manager)
    config = DenoiseCacheConfig(
        coefficients=(0.0,), threshold=1.0, warmup_steps=1, final_steps=1
    )
    signal = torch.ones(1)

    with denoise_step_cache(pipeline.transformer, config, total_steps=3):
        first = pipeline.transformer(hidden_states=signal)
        second = pipeline.transformer(hidden_states=signal)

    assert torch.equal(first, second)
    assert manager.calls == 1


def test_cuda_graph_wrapper_preserves_forward_signature_for_diffusers_layout():
    class FakeTransformer(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            token_tags,
            position_ids,
            video_indices,
        ):
            return hidden_states

    class FakePipeline:
        transformer = FakeTransformer()

    class FakeManager:
        current_bucket = None

        def try_execute(self, key, function, *args, **kwargs):
            return None

    before = inspect.signature(FakePipeline.transformer.forward)
    runner_mod._wrap_transformer_with_cuda_graph(FakePipeline, FakeManager())
    after = inspect.signature(FakePipeline.transformer.forward)

    assert after == before


def test_cuda_graph_pins_adaln_only_with_sufficient_host_headroom():
    from omni_infinity.adaln import AdaLNEntry, HostResidentAdaLN

    entry = AdaLNEntry(
        torch.zeros(12, 4, dtype=torch.bfloat16),
        torch.zeros(12, dtype=torch.bfloat16),
    )
    transformer = torch.nn.Sequential(HostResidentAdaLN(entry, hidden_size=2))
    required = entry.host_nbytes
    seen = []

    pinned = runner_mod._pin_adaln_for_cuda_graph(
        transformer,
        available_bytes=(required * 6) // 5,
        pin=lambda tensor: seen.append(tensor) or tensor,
    )

    assert pinned == required
    assert seen == [entry.weight, entry.bias]
    with pytest.raises(MemoryError, match="20% headroom"):
        runner_mod._pin_adaln_for_cuda_graph(
            transformer,
            available_bytes=(required * 6) // 5 - 1,
            pin=lambda tensor: tensor,
        )


def test_component_offload_invalidates_graphs_between_generations():
    class FakePipeline:
        def __call__(self, **kwargs):
            return type("State", (), {"values": {}})()

    class FakeManager:
        def __init__(self):
            self.invalidations = []

        @contextmanager
        def bucket(self, key):
            yield

        def invalidate(self, reason):
            self.invalidations.append(reason)

    manager = FakeManager()
    runner = ReferenceRunner(
        FakePipeline(),
        cuda_graph_manager=manager,
        cuda_graph_invalidate_between_generations=True,
    )

    runner.generate("first", num_frames=120)
    runner.generate("second", num_frames=120)

    assert manager.invalidations == ["component offload generation boundary"]


def test_compile_blocks_uses_diffusers_regional_compile_api():
    calls = []

    class FakeTransformer:
        def compile_repeated_blocks(self, **kwargs):
            calls.append(kwargs)

    class FakePipeline:
        transformer = FakeTransformer()

    runner_mod._compile_repeated_transformer_blocks(FakePipeline())

    assert calls == [{"fullgraph": True, "dynamic": True}]


def test_stream_text_encoder_invokes_group_offloading():
    from omni_infinity import runner as runner_mod

    calls = {}

    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.visual = torch.nn.Module()
            self.model.visual.patch_embed = torch.nn.Conv3d(3, 4, 1)
            self.model.layers = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(4)]
            )

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
        pipeline.text_encoder.model.visual,
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
        def from_pretrained(
            cls, checkpoint, workflow="fl2va", components_manager=None
        ):
            captured["workflow"] = workflow
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
    # Cache support must not alter the default pipeline construction path.
    assert captured["workflow"] == "fl2va"
    assert captured["built"]["transformer"] == "fake-transformer"
