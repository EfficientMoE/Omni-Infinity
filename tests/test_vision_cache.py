# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers.hooks import HookRegistry, ModelHook
from transformers.modeling_outputs import BaseModelOutputWithPooling

from omni_infinity import registry
from omni_infinity.caches.vision import (
    VisionEmbedCache,
    _visual_module,
    enable_vision_cache,
)


class _Tower(torch.nn.Module):
    def __init__(self, *, tuple_output=False, scale=1.0):
        super().__init__()
        self.calls = 0
        self.tuple_output = tuple_output
        self.scale = scale

    def forward(self, *args, **_kwargs):
        self.calls += 1
        output = args[0] * self.scale
        if self.tuple_output:
            return output, output + 1
        return output


def test_cache_evicts_the_oldest_entry_and_reports_stats():
    cache = VisionEmbedCache(max_entries=1)
    first = torch.ones(2)
    second = torch.ones(3)

    cache.put("first", first)
    assert cache.get("first") is first
    cache.put("second", second)

    assert cache.get("first") is None
    assert cache.get("second") is second
    assert cache.stats() == {
        "hits": 2,
        "misses": 1,
        "entries": 1,
        "bytes": second.nbytes,
    }


def test_cache_requires_at_least_one_entry():
    with pytest.raises(ValueError, match="max_entries"):
        VisionEmbedCache(max_entries=0)


def test_visual_attribute_is_found_on_the_encoder_or_its_model():
    tower = torch.nn.Linear(1, 1)
    assert _visual_module(SimpleNamespace(visual=tower)) is tower
    nested = SimpleNamespace(model=SimpleNamespace(visual=tower))
    assert _visual_module(nested) is tower
    with pytest.raises(AttributeError, match="visual"):
        _visual_module(SimpleNamespace())


def test_repeated_call_skips_the_tower():
    tower = _Tower()
    encoder = SimpleNamespace(visual=tower)
    controller = enable_vision_cache(encoder)

    first = encoder.visual(torch.ones(2))
    second = encoder.visual(torch.ones(2))

    assert torch.equal(first, second)
    assert tower.calls == 1
    controller.close()


def test_cache_survives_removal_of_an_earlier_diffusers_hook():
    tower = _Tower()
    registry = HookRegistry.check_if_exists_or_initialize(tower)

    class _SelfRemovingHook(ModelHook):
        def post_forward(self, module, output):
            module._diffusers_hook.remove_hook("temporary-hook", recurse=False)
            return output

    registry.register_hook(_SelfRemovingHook(), "temporary-hook")
    controller = enable_vision_cache(SimpleNamespace(visual=tower))

    tower(torch.ones(2))
    tower(torch.ones(2))

    assert tower.calls == 1
    assert controller.cache.stats()["hits"] == 1
    controller.close()


def test_a_changed_second_tensor_is_a_miss():
    tower = _Tower()
    encoder = SimpleNamespace(visual=tower)
    enable_vision_cache(encoder)

    encoder.visual(torch.ones(2), torch.zeros(2))
    encoder.visual(torch.ones(2), torch.ones(2))

    assert tower.calls == 2


def test_tuple_output_is_stored_on_cpu_and_returned_on_the_input_device():
    tower = _Tower(tuple_output=True)
    cache = VisionEmbedCache()
    encoder = SimpleNamespace(visual=tower)
    enable_vision_cache(encoder, cache)
    value = torch.ones(2)

    first = encoder.visual(value)
    second = encoder.visual(value.clone())
    cached = next(iter(cache._entries.values()))

    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert all(tensor.device.type == "cpu" for tensor in cached)
    assert all(tensor.device == value.device for tensor in second)


def test_transformers_model_output_type_is_preserved_on_a_hit():
    tower = _Tower()

    def model_output_forward(value):
        tower.calls += 1
        return BaseModelOutputWithPooling(
            last_hidden_state=value,
            pooler_output=value + 1,
        )

    tower.forward = model_output_forward
    encoder = SimpleNamespace(visual=tower)
    enable_vision_cache(encoder)

    encoder.visual(torch.ones(2))
    cached = encoder.visual(torch.ones(2))

    assert isinstance(cached, BaseModelOutputWithPooling)
    assert torch.equal(cached.pooler_output, torch.full((2,), 2.0))
    assert tower.calls == 1


def test_close_removes_an_installed_instance_forward():
    tower = _Tower()
    controller = enable_vision_cache(SimpleNamespace(visual=tower))

    assert "forward" in tower.__dict__
    controller.close()
    controller.close()

    assert "forward" not in tower.__dict__


def test_close_restores_an_existing_instance_forward():
    tower = _Tower()

    def custom(value):
        return value + 2

    tower.forward = custom
    controller = enable_vision_cache(SimpleNamespace(visual=tower))
    controller.close()

    assert tower.forward is custom


def test_shared_cache_instance_is_namespaced_by_tower():
    cache = VisionEmbedCache(max_entries=8)
    tower_a, tower_b = _Tower(scale=2.0), _Tower(scale=3.0)
    enable_vision_cache(SimpleNamespace(visual=tower_a), cache)
    enable_vision_cache(SimpleNamespace(visual=tower_b), cache)
    pixels = torch.ones(2)
    output_a = tower_a(pixels)
    output_b = tower_b(pixels)
    assert tower_a.calls == 1 and tower_b.calls == 1
    assert not torch.equal(output_a, output_b)


def test_vision_cache_is_an_opt_in_registry_optimization():
    assert registry.runner_kwargs_for("vdn-hybrid", ["vision-cache"]) == {
        "vision_cache": True
    }


def test_documentation_defines_whole_call_cache_semantics():
    text = Path("docs/caches_c3_vision.md").read_text()
    for sentence in (
        "The key is one tower call, not one image inside the call.",
        "A different prompt with the same image is a hit.",
        "C2 owns per-image splitting.",
    ):
        assert sentence in text
