# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from types import MappingProxyType

import pytest

from omni_infinity import registry
from omni_infinity.registry import OptimizationSpec


def test_static_optimization_set_excludes_caches():
    assert set(registry.OPTIMIZATIONS) == {
        "adaln-host-cache",
        "fp8",
        "block-stream",
        "text-encoder-stream",
    }


def test_missing_loader_entries_are_unknown(monkeypatch):
    monkeypatch.setattr(registry, "load_cache_optimizations", lambda: {})
    with pytest.raises(ValueError, match="condition-cache"):
        registry.runner_kwargs_for("h3-dense", ["condition-cache"])


def test_loader_spec_merges_like_a_static_optimization(monkeypatch):
    spec = OptimizationSpec(
        name="condition-cache",
        description="stub",
        supported_archs=("h3-dense",),
        runner_kwargs_by_arch=MappingProxyType(
            {"h3-dense": MappingProxyType({"condition_cache": True})}
        ),
    )
    monkeypatch.setattr(
        registry, "load_cache_optimizations", lambda: {"condition-cache": spec}
    )
    assert registry.runner_kwargs_for("h3-dense", ["condition-cache"]) == {
        "condition_cache": True
    }
    with pytest.raises(ValueError, match="vdn-hybrid"):
        registry.runner_kwargs_for("vdn-hybrid", ["condition-cache"])
