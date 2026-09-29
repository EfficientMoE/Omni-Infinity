# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Issue #24 C3 — content-hash vision-embedding cache.

The reuse rule copied from vLLM-Omni's multimodal encoder cache: the key
is the hash of the tower's input content, the value is the tower output,
and a repeated image skips the tower entirely — independent of any
block/prefix alignment.
"""

import pytest
import torch

from omni_infinity.caches.vision import (
    VisionEmbedCache,
    enable_vision_cache,
)


class FakeVisualTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixel_values, grid_thw=None):
        self.calls += 1
        return pixel_values * 2.0


class FakeTupleTower(torch.nn.Module):
    """Qwen3-VL shape: (embeds, deepstack feature list)."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixel_values, grid_thw):
        self.calls += 1
        return pixel_values * 2.0, [pixel_values + 1.0, pixel_values - 1.0]


class FakeEncoder(torch.nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = tower


def test_repeated_content_skips_the_tower():
    tower = FakeVisualTower()
    controller = enable_vision_cache(FakeEncoder(tower))
    pixels = torch.arange(12, dtype=torch.float32).view(3, 4)
    first = tower(pixels, grid_thw=torch.tensor([[1, 2, 2]]))
    second = tower(pixels.clone(), grid_thw=torch.tensor([[1, 2, 2]]))
    assert tower.calls == 1
    assert torch.equal(first, second)
    assert controller.cache.stats() == {
        "hits": 1,
        "misses": 1,
        "entries": 1,
        "bytes": second.untyped_storage().nbytes(),
    }


def test_different_content_recomputes():
    tower = FakeVisualTower()
    enable_vision_cache(FakeEncoder(tower))
    tower(torch.ones(2, 2))
    tower(torch.zeros(2, 2))
    assert tower.calls == 2


def test_different_grid_recomputes():
    tower = FakeVisualTower()
    enable_vision_cache(FakeEncoder(tower))
    pixels = torch.ones(2, 2)
    tower(pixels, grid_thw=torch.tensor([[1, 1, 4]]))
    tower(pixels, grid_thw=torch.tensor([[1, 4, 1]]))
    assert tower.calls == 2


def test_tuple_outputs_round_trip():
    tower = FakeTupleTower()
    enable_vision_cache(FakeEncoder(tower))
    pixels = torch.ones(2, 3)
    grid = torch.tensor([[1, 1, 2]])
    embeds, deepstack = tower(pixels, grid)
    embeds2, deepstack2 = tower(pixels, grid)
    assert tower.calls == 1
    assert torch.equal(embeds, embeds2)
    assert len(deepstack2) == 2
    for a, b in zip(deepstack, deepstack2):
        assert torch.equal(a, b)


def test_lru_eviction():
    tower = FakeVisualTower()
    enable_vision_cache(FakeEncoder(tower), max_entries=1)
    a, b = torch.ones(2, 2), torch.zeros(2, 2)
    tower(a)
    tower(b)  # evicts a
    tower(a)
    assert tower.calls == 3


def test_close_restores_the_original_forward():
    tower = FakeVisualTower()
    controller = enable_vision_cache(FakeEncoder(tower))
    pixels = torch.ones(2, 2)
    tower(pixels)
    controller.close()
    tower(pixels)
    tower(pixels)
    assert tower.calls == 3
    controller.close()  # idempotent


def test_close_restores_a_preexisting_instance_forward():
    tower = FakeVisualTower()
    sentinel_calls = []

    def instance_forward(pixel_values, grid_thw=None):
        sentinel_calls.append(1)
        return pixel_values

    tower.forward = instance_forward
    controller = enable_vision_cache(FakeEncoder(tower))
    controller.close()
    assert tower.__dict__["forward"] is instance_forward


def test_visual_attribute_directly_on_encoder():
    tower = FakeVisualTower()
    encoder = torch.nn.Module()
    encoder.visual = tower
    enable_vision_cache(encoder)
    tower(torch.ones(1, 1))
    tower(torch.ones(1, 1))
    assert tower.calls == 1


def test_missing_visual_tower_raises():
    with pytest.raises(AttributeError, match="vision tower"):
        enable_vision_cache(torch.nn.Module())


def test_shared_cache_instance_across_towers():
    cache = VisionEmbedCache(max_entries=8)
    tower_a, tower_b = FakeVisualTower(), FakeVisualTower()
    enable_vision_cache(FakeEncoder(tower_a), cache)
    enable_vision_cache(FakeEncoder(tower_b), cache)
    pixels = torch.ones(2, 2)
    tower_a(pixels)
    tower_b(pixels)
    assert tower_a.calls == 1 and tower_b.calls == 0
