# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from omni_infinity.caches.vision import VisionEmbedCache, _visual_module


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
    assert _visual_module(SimpleNamespace(model=SimpleNamespace(visual=tower))) is tower
    with pytest.raises(AttributeError, match="visual"):
        _visual_module(SimpleNamespace())
