# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from omni_infinity.runner import ReferenceRunner


def test_ref2va_routes_transformer_ref_through_task2_path(monkeypatch):
    from omni_infinity import runner as runner_mod
    from omni_infinity import store as store_mod

    captured = {}

    class FakeTransformer:
        adaln_host_cache = True

    class FakePipeline:
        def __init__(self):
            self.components = {}

        def load_components(self, *, names, torch_dtype):
            captured["load_names"] = names
            captured["load_dtype"] = torch_dtype

        def update_components(self, **components):
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
        captured["block_stream_component"] = component_name

    def fake_text_stream(pipeline, device):
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
    monkeypatch.setattr(runner_mod, "_stream_text_encoder", fake_text_stream)

    ReferenceRunner.from_pretrained(
        "/checkpoint",
        workflow="ref2va",
        offload=True,
        store_dir="/store",
        store_components=("transformer_ref",),
        adaln_host_cache=True,
        block_stream_blocks_per_group=1,
        stream_text_encoder=True,
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
    assert captured["stream_text_encoder"] is True
