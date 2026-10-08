# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from omni_infinity import registry
from omni_infinity.caches.prefix import (
    EncoderPrefixCache,
    enable_encoder_cache,
    encoder_prefix_key,
)


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, *, pixel_values=None, **_kwargs):
        self.calls += 1
        hidden = input_ids.to(torch.float32).unsqueeze(-1)
        if pixel_values is not None:
            hidden = hidden + pixel_values.flatten()[0]
        return BaseModelOutput(
            last_hidden_state=hidden,
            hidden_states=(hidden, hidden + 1),
        )


def test_cache_evicts_lru_entries_until_within_byte_budget():
    cache = EncoderPrefixCache(max_entries=3, max_bytes=16)
    first = torch.ones(2)
    second = torch.ones(2) * 2
    third = torch.ones(2) * 3

    cache.put("first", first)
    cache.put("second", second)
    assert cache.get("first") is first
    cache.put("third", third)

    assert cache.get("second") is None
    assert cache.get("first") is first
    assert cache.get("third") is third
    assert cache.stats()["entries"] == 2
    assert cache.stats()["bytes"] == 16


def test_entry_larger_than_byte_budget_is_transient():
    cache = EncoderPrefixCache(max_entries=2, max_bytes=4)

    cache.put("oversized", torch.ones(2))

    assert cache.get("oversized") is None
    assert cache.stats() == {
        "hits": 0,
        "misses": 1,
        "entries": 0,
        "bytes": 0,
    }


def test_repeated_exact_call_hits_and_changed_prefix_or_image_misses():
    model = _Encoder()
    text_encoder = SimpleNamespace(model=model)
    controller = enable_encoder_cache(text_encoder)
    prefix = torch.tensor([[1, 2, 3]])
    image = torch.ones(1)

    first = model(input_ids=prefix, pixel_values=image)
    second = model(input_ids=prefix.clone(), pixel_values=image.clone())
    model(input_ids=torch.tensor([[1, 2, 4]]), pixel_values=image)
    model(input_ids=prefix, pixel_values=torch.zeros(1))

    assert torch.equal(first.hidden_states[1], second.hidden_states[1])
    assert model.calls == 3
    assert controller.cache.stats()["hits"] == 1


def test_entry_cap_uses_last_access_order():
    cache = EncoderPrefixCache(max_entries=2, max_bytes=1024)
    first, second, third = torch.ones(1), torch.ones(1) * 2, torch.ones(1) * 3
    cache.put("first", first)
    cache.put("second", second)
    assert cache.get("first") is first

    cache.put("third", third)

    assert cache.get("second") is None
    assert cache.get("first") is first
    assert cache.get("third") is third


def test_key_is_deterministic_and_changes_with_prefix_or_image():
    prefix = torch.tensor([[1, 2, 3]])
    image = torch.arange(4, dtype=torch.float32).reshape(1, 4)

    first = encoder_prefix_key(prefix, (image,))

    assert first == encoder_prefix_key(prefix.clone(), (image.clone(),))
    assert first != encoder_prefix_key(torch.tensor([[1, 2, 4]]), (image,))
    assert first != encoder_prefix_key(prefix, (image + 1,))
    assert len(first) == 64


def test_key_covers_every_non_vision_encoder_input():
    model = _Encoder()
    cache = EncoderPrefixCache()
    enable_encoder_cache(SimpleNamespace(model=model), cache)
    prefix = torch.tensor([[1, 2, 3]])

    model(input_ids=prefix, attention_mask=torch.tensor([[1, 1, 1]]))
    model(input_ids=prefix, attention_mask=torch.tensor([[1, 1, 0]]))

    assert cache.stats() == {
        "hits": 0,
        "misses": 2,
        "entries": 2,
        "bytes": cache.stats()["bytes"],
    }
    assert model.calls == 2


def test_concurrent_get_and_put_remain_bounded():
    cache = EncoderPrefixCache(max_entries=8, max_bytes=8 * 4)

    def exercise(worker: int):
        for iteration in range(100):
            key = f"{worker}-{iteration % 4}"
            cache.put(key, torch.tensor([iteration], dtype=torch.float32))
            cache.get(key)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(exercise, range(8)))

    stats = cache.stats()
    assert stats["entries"] <= 8
    assert stats["bytes"] <= 8 * 4


def test_cached_output_is_host_resident_and_materialized_to_input_device():
    model = _Encoder()
    cache = EncoderPrefixCache()
    enable_encoder_cache(SimpleNamespace(model=model), cache)
    prefix = torch.tensor([[1, 2, 3]])

    first = model(input_ids=prefix)
    second = model(input_ids=prefix.clone())
    cached = next(iter(cache._entries.values()))

    assert cached.last_hidden_state.device.type == "cpu"
    assert all(tensor.device.type == "cpu" for tensor in cached.hidden_states)
    assert second.last_hidden_state.device == prefix.device
    assert torch.equal(first.last_hidden_state, second.last_hidden_state)


def test_oversized_encoder_output_is_returned_but_not_cached():
    model = _Encoder()
    cache = EncoderPrefixCache(max_bytes=1)
    enable_encoder_cache(SimpleNamespace(model=model), cache)
    prefix = torch.tensor([[1, 2, 3]])

    first = model(input_ids=prefix)
    second = model(input_ids=prefix)

    assert torch.equal(first.last_hidden_state, second.last_hidden_state)
    assert model.calls == 2
    assert cache.stats()["entries"] == 0


def test_controller_close_restores_the_original_forward():
    model = _Encoder()
    controller = enable_encoder_cache(SimpleNamespace(model=model))

    assert "forward" in model.__dict__
    controller.close()
    controller.close()

    assert "forward" not in model.__dict__


def test_encoder_cache_is_h3_dense_only_registry_optimization():
    assert registry.runner_kwargs_for("h3-dense", ["encoder-cache"]) == {
        "encoder_cache": True
    }
    with pytest.raises(ValueError, match="vdn-hybrid"):
        registry.runner_kwargs_for("vdn-hybrid", ["encoder-cache"])


def test_documentation_defines_exact_match_scope_and_bounds():
    text = Path("docs/caches_c2_prefix.md").read_text()
    for sentence in (
        "C2 v1 reuses only an exact tokenized presentation.",
        "Both the entry cap and byte budget are enforced.",
        "Partial-prefix splicing is not implemented.",
    ):
        assert sentence in text
