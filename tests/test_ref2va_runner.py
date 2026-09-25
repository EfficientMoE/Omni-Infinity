# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import pytest

from omni_infinity.runner import ReferenceRunner


def test_ref2va_routes_transformer_ref_through_task2_path(monkeypatch):
    from omni_infinity import runner as runner_mod
    from omni_infinity import store as store_mod

    captured = {}
    ordering = []

    class FakeTransformer:
        adaln_host_cache = True

    class FakePipeline:
        def __init__(self):
            self.components = {}

        def load_components(self, *, names, torch_dtype):
            captured["load_names"] = names
            captured["load_dtype"] = torch_dtype

        def update_components(self, **components):
            ordering.append("update_components")
            self.components.update(components)
            if "transformer_ref" in components:
                captured["built_key"] = "transformer_ref"

        def get_component(self, name):
            return self.components[name]

    pipeline = FakePipeline()

    def fake_from_pretrained(checkpoint, *, workflow, components_manager):
        captured["checkpoint"] = checkpoint
        captured["workflow"] = workflow
        captured["components_manager"] = components_manager
        return pipeline

    class FakeModularPipeline:
        from_pretrained = staticmethod(fake_from_pretrained)

    class FakeComponentsManager:
        def enable_auto_cpu_offload(self, **kwargs):
            ordering.append("components_manager_offload")
            captured["auto_offload"] = kwargs

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, checkpoint, *, subfolder):
            captured["processor"] = (checkpoint, subfolder)
            return cls()

    class FakeSource:
        def __init__(self, store_dir):
            captured["store_dir"] = store_dir

    def fake_load_transformer(
        component_cls,
        checkpoint,
        source,
        torch_dtype,
        component="transformer",
        **kwargs,
    ):
        captured["store_component"] = component
        captured["adaln_host_cache"] = True
        return FakeTransformer()

    def fake_block_stream(
        pipeline, device, blocks_per_group, to_disk, component_name
    ):
        ordering.append("block_streaming")
        captured["block_stream_component"] = component_name

    overlap_controller = object()

    def fake_step_overlap(transformer):
        ordering.append("step_overlap")
        captured["step_overlap_transformer"] = transformer
        return overlap_controller

    def fake_text_stream(pipeline, device):
        ordering.append("text_encoder_streaming")
        captured["stream_text_encoder"] = True

    monkeypatch.setattr(
        "diffusers.MiniMaxH3ModularPipeline", FakeModularPipeline
    )
    monkeypatch.setattr(
        "diffusers.modular_pipelines.ComponentsManager", FakeComponentsManager
    )
    monkeypatch.setattr("transformers.AutoProcessor", FakeProcessor)
    monkeypatch.setattr(store_mod, "StoreComponentSource", FakeSource)
    monkeypatch.setattr(
        store_mod, "load_transformer_with_adaln_cache", fake_load_transformer
    )
    monkeypatch.setattr(
        runner_mod, "_h3_component_class", lambda name: object
    )
    monkeypatch.setattr(
        runner_mod, "_enable_block_streaming", fake_block_stream
    )
    monkeypatch.setattr(
        "omni_infinity.step_overlap.enable_step_overlap", fake_step_overlap
    )
    monkeypatch.setattr(runner_mod, "_stream_text_encoder", fake_text_stream)

    runner = ReferenceRunner.from_pretrained(
        "/checkpoint",
        workflow="ref2va",
        offload=True,
        store_dir="/store",
        store_components=("transformer_ref",),
        adaln_host_cache=True,
        block_stream_blocks_per_group=1,
        stream_text_encoder=True,
        step_overlap=True,
    )

    assert captured["workflow"] == "ref2va"
    assert captured["load_names"] == [
        "text_encoder",
        "tokenizer",
        "vae",
        "audio_vae",
        "scheduler",
        "audio_scheduler",
    ]
    assert captured["store_component"] == "transformer_ref"
    assert captured["built_key"] == "transformer_ref"
    assert captured["adaln_host_cache"] is True
    assert captured["block_stream_component"] == "transformer_ref"
    assert captured["step_overlap_transformer"] is pipeline.components[
        "transformer_ref"
    ]
    assert captured["stream_text_encoder"] is True
    assert ordering == [
        "update_components",
        "block_streaming",
        "step_overlap",
        "text_encoder_streaming",
        "components_manager_offload",
    ]
    assert runner.overlap_controller is overlap_controller


def test_step_overlap_requires_bf16_block_streaming():
    with pytest.raises(
        ValueError, match="^step_overlap requires bf16 block streaming$"
    ):
        ReferenceRunner.from_pretrained(step_overlap=True)


def test_ref2va_smoke_issue_command_uses_optimized_defaults(monkeypatch):
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

    from examples import ref2va_smoke

    monkeypatch.setenv("OMNI_H3_CHECKPOINT", "/offline/checkpoint")

    command = [
        "--ref",
        "tests/fixtures/ref.png",
        "--seed",
        "0",
        "--steps",
        "8",
        "--resolution",
        "256p",
        "--frames",
        "120",
        "--max-vram",
        "22GiB",
    ]
    args = ref2va_smoke.parse_args(command)
    reference = object()
    monkeypatch.setattr(
        MiniMaxH3ImageReference,
        "from_file",
        staticmethod(lambda path: reference),
    )

    references = ref2va_smoke.build_references(args.ref)
    options = ref2va_smoke.runner_options(args)

    assert args.frames == 120
    assert args.checkpoint == "/offline/checkpoint"
    assert references == [reference]
    assert options["workflow"] == "ref2va"
    assert options["offload"] is True
    assert options["store_dir"] == "/mnt/raid0nvme0/leyang/h3-store-v2"
    assert options["store_components"] == (
        "transformer_ref",
        "vae",
        "audio_vae",
    )
    assert options["adaln_host_cache"] is True
    assert options["block_stream_blocks_per_group"] == 1
    assert options["stream_text_encoder"] is True
    assert options["step_overlap"] is True

    calls = {}

    class FakeRunner:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            calls["runner_options"] = kwargs
            return cls()

        def generate(self, prompt, **kwargs):
            calls["generate"] = kwargs
            return object()

    monkeypatch.setattr(ref2va_smoke, "ReferenceRunner", FakeRunner)
    monkeypatch.setattr(
        ref2va_smoke, "report_latent_parity", lambda *args: True
    )
    monkeypatch.setattr(ref2va_smoke, "export_outputs", lambda *args: None)
    monkeypatch.setattr(ref2va_smoke, "_offload_margin", lambda value: None)
    monkeypatch.setattr(
        ref2va_smoke.torch.cuda, "reset_peak_memory_stats", lambda: None
    )
    monkeypatch.setattr(
        ref2va_smoke.torch.cuda, "max_memory_allocated", lambda: 1
    )

    assert ref2va_smoke.main(command) == 0
    assert calls["generate"]["references"] == [reference]
    assert calls["generate"]["num_frames"] == 120
    assert calls["runner_options"]["step_overlap"] is True


def test_ref2va_smoke_full_resident_recording_disables_optimizations():
    from examples import ref2va_smoke

    args = ref2va_smoke.parse_args(
        [
            "--ref",
            "tests/fixtures/ref.png",
            "--full-resident",
            "--record-goldens",
            "tests/fixtures/goldens",
        ]
    )

    options = ref2va_smoke.runner_options(args)

    assert options["workflow"] == "ref2va"
    assert options["offload"] is True
    assert options["store_dir"] is None
    assert options["store_components"] == ()
    assert options["adaln_host_cache"] is False
    assert options["block_stream_blocks_per_group"] == 0
    assert options["stream_text_encoder"] is False
    assert options["step_overlap"] is False
