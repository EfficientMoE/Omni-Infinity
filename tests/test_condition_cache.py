# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from omni_infinity import registry
from omni_infinity.caches import condition as condition_module
from omni_infinity.caches.condition import (
    ConditionCache,
    build_conditioned_pipeline,
    condition_key,
    prepare,
)


class _UnsafePayload:
    pass


class _Input:
    def __init__(self, name):
        self.name = name


class _Block:
    def __init__(self, *inputs):
        self.inputs = [_Input(name) for name in inputs]


class _Blocks:
    def __init__(self, sub_blocks):
        self.sub_blocks = sub_blocks

    @property
    def inputs(self):
        return [
            spec for block in self.sub_blocks.values() for spec in block.inputs
        ]


class _Pipeline:
    def __init__(self, blocks=None, components=None):
        self.blocks = blocks
        self.components = components or {"transformer": object()}
        self.calls = []
        self._execution_device = torch.device("cpu")

    def update_components(self, **components):
        self.components.update(components)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)


class _ReducedBlocks(_Blocks):
    execution_device = torch.device("cpu")

    def init_pipeline(self):
        pipeline = _Pipeline(blocks=self, components={})
        pipeline._execution_device = self.execution_device
        return pipeline


class _FakeSequentialPipelineBlocks:
    @classmethod
    def from_blocks_dict(cls, blocks_dict):
        return _ReducedBlocks(blocks_dict)


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(
        condition_module,
        "SequentialPipelineBlocks",
        _FakeSequentialPipelineBlocks,
    )
    blocks = _Blocks(
        {
            "text_encoder": _Block("prompt"),
            "vae_encoder": _Block("image"),
            "denoise": _Block(
                "prompt_embeds",
                "text_token_tags",
                "condition_latents",
                "generator",
            ),
        }
    )
    return _Pipeline(blocks=blocks)


def _state(**kwargs):
    return SimpleNamespace(**kwargs)


def test_identical_condition_gives_identical_key():
    kwargs = dict(
        namespace="ReferenceRunner:/ckpt",
        prompt="[Shot 1] a",
        media=(b"png", None),
        height=368,
        width=640,
        num_frames=120,
    )
    digest = condition_key(**kwargs)

    assert digest == condition_key(**kwargs)
    assert len(digest) == 64
    int(digest, 16)


def test_shared_shot1_opener_is_not_a_hit():
    common = dict(
        namespace="ReferenceRunner:/ckpt",
        media=(),
        height=368,
        width=640,
        num_frames=120,
    )
    left = condition_key(prompt="[Shot 1] same\n[Shot 2] left", **common)
    right = condition_key(prompt="[Shot 1] same\n[Shot 2] right", **common)

    assert left != right


def test_canvas_and_checkpoint_and_none_slots_are_in_the_key():
    base = dict(
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(b"a", None),
        height=368,
        width=640,
        num_frames=120,
    )

    assert condition_key(**base) != condition_key(**{**base, "height": 512})
    assert condition_key(**base) != condition_key(
        **{**base, "namespace": "ReferenceRunner:/other"}
    )
    assert condition_key(**base) != condition_key(
        **{**base, "media": (None, b"a")}
    )


def test_unhashable_media_raises_type_error():
    with pytest.raises(TypeError, match="unsupported hash value"):
        condition_key(
            namespace="ReferenceRunner:/ckpt",
            prompt="p",
            media=(object(),),
            height=368,
            width=640,
            num_frames=120,
        )


def test_cache_requires_prompt_embeds_and_tracks_hits_and_misses():
    cache = ConditionCache()

    assert cache.put("missing", {"text_token_tags": torch.ones(1)}) is None
    assert cache.get("missing") is None

    source = torch.zeros(2) + 1
    entry = cache.put("present", {"prompt_embeds": source})
    assert entry is not None
    stored = entry.to("cpu")
    assert stored["prompt_embeds"].device.type == "cpu"
    assert torch.equal(stored["prompt_embeds"], source)

    loaded = cache.get("present")
    assert loaded is not None
    assert torch.equal(loaded.to("cpu")["prompt_embeds"], source)
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_cache_validates_capacity_and_required_capture_names():
    with pytest.raises(ValueError, match="max_entries"):
        ConditionCache(max_entries=0)
    with pytest.raises(ValueError, match="required"):
        ConditionCache(capture=("text_token_tags",))


def test_max_entries_keeps_the_newest_key():
    cache = ConditionCache(max_entries=1)
    cache.put("old", {"prompt_embeds": torch.zeros(1)})
    cache.put("new", {"prompt_embeds": torch.ones(1)})

    assert cache.get("old") is None
    assert cache.get("new") is not None


def test_max_bytes_drops_the_oldest_entry():
    entry_bytes = torch.zeros(2).nbytes
    cache = ConditionCache(max_bytes=entry_bytes)
    cache.put("old", {"prompt_embeds": torch.zeros(2)})
    cache.put("new", {"prompt_embeds": torch.ones(2)})

    assert cache.get("old") is None
    loaded = cache.get("new")
    assert loaded is not None
    assert torch.equal(loaded.to("cpu")["prompt_embeds"], torch.ones(2))


def test_disk_cache_round_trip_uses_cpu_tensors(tmp_path):
    source = torch.arange(3)
    first = ConditionCache(cache_dir=tmp_path)
    first.put(
        "disk",
        {
            "prompt_embeds": source,
            "condition_latents": None,
            "ignored": torch.ones(1),
        },
    )

    second = ConditionCache(cache_dir=tmp_path)
    loaded = second.get("disk")

    assert loaded is not None
    assert set(loaded.to("cpu")) == {"prompt_embeds"}
    assert torch.equal(loaded.to("cpu")["prompt_embeds"], source)


def test_truncated_disk_entry_is_a_miss(tmp_path):
    (tmp_path / "broken.pt").write_bytes(b"not a torch archive")

    cache = ConditionCache(cache_dir=tmp_path)

    assert cache.get("broken") is None


def test_disk_load_rejects_non_tensor_payload_with_weights_only(tmp_path):
    torch.save(_UnsafePayload(), tmp_path / "unsafe.pt")

    cache = ConditionCache(cache_dir=tmp_path)

    assert cache.get("unsafe") is None


def test_disk_load_rejects_none_for_required_value(tmp_path):
    torch.save({"prompt_embeds": None}, tmp_path / "missing.pt")

    cache = ConditionCache(cache_dir=tmp_path)

    assert cache.get("missing") is None


def test_disk_write_failure_keeps_memory_entry(tmp_path, monkeypatch):
    cache = ConditionCache(cache_dir=tmp_path)

    def fail_mkstemp(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(
        condition_module.tempfile, "NamedTemporaryFile", fail_mkstemp
    )

    entry = cache.put("key", {"prompt_embeds": torch.ones(1)})

    assert entry is not None
    assert cache.get("key") is entry


def test_miss_captures_and_hit_skips_encoders(pipeline):
    cache = ConditionCache()
    generator = object()
    request = {"prompt": "p", "generator": generator}

    first = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(b"img",),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs=request,
    )
    assert first.hit is False
    assert first.pipeline is None
    assert first.call_kwargs() is request
    assert first.call_kwargs()["generator"] is generator
    first.observe(_state(prompt_embeds=torch.ones(2)))

    second = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(b"img",),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs={"prompt": "p", "generator": generator},
    )

    assert second.hit is True
    assert second.pipeline is not pipeline
    kwargs = second.call_kwargs()
    assert torch.equal(kwargs["prompt_embeds"], torch.ones(2))
    assert "prompt" not in kwargs
    assert kwargs["generator"] is generator
    second.pipeline(**kwargs)
    assert pipeline.calls == []
    assert second.pipeline.calls == [kwargs]


def test_image_hit_preserves_workflow_selector_for_retained_blocks(pipeline):
    cache = ConditionCache()
    image = object()
    request = {"prompt": "p", "image": image, "generator": object()}
    first = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(b"image",),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs=request,
    )
    first.observe(
        _state(
            prompt_embeds=torch.ones(2),
            text_token_tags=torch.ones(2, dtype=torch.long),
            condition_latents=[torch.ones(1)],
        )
    )

    second = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(b"image",),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs=request,
    )

    assert second.hit is True
    assert second.call_kwargs()["image"] is image


def test_hit_moves_cached_condition_to_pipeline_execution_device(
    pipeline, monkeypatch
):
    cache = ConditionCache()
    monkeypatch.setattr(
        _ReducedBlocks, "execution_device", torch.device("meta")
    )
    generator = SimpleNamespace(device=torch.device("cpu"))
    request = {"prompt": "p", "generator": generator}
    first = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs=request,
    )
    first.observe(_state(prompt_embeds=torch.ones(2)))

    second = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs=request,
    )

    assert second.hit is True
    replay_kwargs = second.call_kwargs()
    assert replay_kwargs["prompt_embeds"].device.type == "meta"
    assert replay_kwargs["generator"] is generator


def test_hit_keeps_h3_layout_inputs_on_cpu(pipeline, monkeypatch):
    cache = ConditionCache()
    monkeypatch.setattr(
        _ReducedBlocks, "execution_device", torch.device("meta")
    )
    request = {"prompt": "p", "generator": object()}
    first = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs=request,
    )
    first.observe(
        _state(
            prompt_embeds=torch.ones(2),
            text_token_tags=torch.ones(2, dtype=torch.long),
            condition_latents=[torch.ones(1)],
        )
    )

    second = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs=request,
    )

    replay_kwargs = second.call_kwargs()
    assert replay_kwargs["prompt_embeds"].device.type == "meta"
    assert replay_kwargs["text_token_tags"].device.type == "cpu"
    assert replay_kwargs["condition_latents"][0].device.type == "cpu"


def test_miss_observe_accepts_a_mapping(pipeline):
    cache = ConditionCache()
    replay = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs={"prompt": "p"},
    )

    replay.observe({"prompt_embeds": torch.full((1,), 3)})

    assert cache.stats()["entries"] == 1


def test_different_prompt_is_a_miss(pipeline):
    cache = ConditionCache()
    first = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="left",
        media=(),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs={"prompt": "left"},
    )
    first.observe(_state(prompt_embeds=torch.ones(1)))

    second = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="right",
        media=(),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs={"prompt": "right"},
    )

    assert second.hit is False


def test_unhashable_reference_fails_open(pipeline):
    cache = ConditionCache()
    request = {"prompt": "p", "generator": object()}

    replay = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(object(),),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs=request,
    )

    assert replay.hit is False
    assert replay.pipeline is None
    assert replay.call_kwargs() is request
    replay.observe(_state(prompt_embeds=torch.ones(1)))
    assert cache.stats()["entries"] == 0


def test_ref2va_repeated_reference_hits_and_changed_reference_misses(
    pipeline,
):
    pytest.importorskip("diffusers")
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3ImageReference,
    )
    from PIL import Image

    cache = ConditionCache()

    def replay_for(color):
        reference = MiniMaxH3ImageReference(
            image=Image.new("RGB", (2, 2), color=color)
        )
        return prepare(
            pipeline,
            cache,
            namespace="ReferenceRunner:/ckpt",
            prompt="p",
            media=(reference,),
            height=368,
            width=640,
            num_frames=120,
            call_kwargs={"prompt": "p", "references": [reference]},
        )

    first = replay_for("red")
    assert first.hit is False
    first.observe(_state(prompt_embeds=torch.ones(1)))

    assert replay_for("red").hit is True
    assert replay_for("blue").hit is False


def test_pipeline_that_cannot_be_reduced_fails_open():
    cache = ConditionCache()
    pipeline = _Pipeline(blocks=None)
    request = {"prompt": "p", "generator": object()}
    first = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs=request,
    )
    first.observe(_state(prompt_embeds=torch.ones(1)))

    second = prepare(
        pipeline,
        cache,
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(),
        height=368,
        width=640,
        num_frames=120,
        call_kwargs=request,
    )

    assert second.hit is False
    assert second.pipeline is None
    assert second.call_kwargs() is request
    assert cache.stats()["entries"] == 1


def test_loader_publishes_condition_cache():
    optimizations = registry.load_cache_optimizations()

    assert "condition-cache" in optimizations
    assert registry.runner_kwargs_for("h3-dense", ["condition-cache"]) == {
        "condition_cache": True
    }
    assert registry.runner_kwargs_for("vdn-hybrid", ["condition-cache"]) == {
        "condition_cache": True
    }


def test_h3_blocks_can_drop_encoders():
    pytest.importorskip("diffusers")
    from diffusers import MiniMaxH3Blocks
    from diffusers.modular_pipelines import SequentialPipelineBlocks

    monkeypatch_target = condition_module.SequentialPipelineBlocks
    condition_module.SequentialPipelineBlocks = SequentialPipelineBlocks
    try:
        pipeline = MiniMaxH3Blocks().init_pipeline()
        names = set(pipeline.blocks.sub_blocks)
        assert {"text_encoder", "vae_encoder"}.issubset(names)

        reduced = build_conditioned_pipeline(pipeline)

        assert reduced is not None
        assert "text_encoder" not in reduced.blocks.sub_blocks
        assert "vae_encoder" not in reduced.blocks.sub_blocks
        shared = set(pipeline.components) & set(reduced.components)
        for name in shared:
            if pipeline.components[name] is not None:
                assert reduced.components[name] is pipeline.components[name]
    finally:
        condition_module.SequentialPipelineBlocks = monkeypatch_target


def test_documentation_records_safety_and_determinism_contract():
    text = Path("docs/caches_c1_condition.md").read_text()

    for sentence in (
        "A shared [Shot 1] opener is not a hit.",
        "use_cache=False",
        "keyframe_encode_seed",
        "the request generator is not consumed",
    ):
        assert sentence in text
