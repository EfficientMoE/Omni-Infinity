# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from types import MappingProxyType

import pytest

from omni_infinity import registry
from omni_infinity.caches.attach import attach_caches, bind_generation
from omni_infinity.registry import OptimizationSpec


class _Runner:
    def __init__(self):
        self.pipeline = object()


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


def test_attach_caches_records_disabled_state_and_namespace():
    runner = _Runner()

    attach_caches(
        runner,
        condition_cache=False,
        condition_cache_dir=None,
        vision_cache=False,
        cache_namespace="ReferenceRunner:/ckpt",
    )

    assert runner.condition_cache is None
    assert runner.vision_cache_controller is None
    assert runner.cache_namespace == "ReferenceRunner:/ckpt"


@pytest.mark.parametrize("cache_name", ["condition_cache", "vision_cache"])
def test_attach_caches_fails_closed_when_module_is_missing(cache_name):
    kwargs = {"condition_cache": False, "vision_cache": False}
    kwargs[cache_name] = True

    with pytest.raises(RuntimeError, match="not installed"):
        attach_caches(
            _Runner(),
            condition_cache_dir=None,
            cache_namespace="ReferenceRunner:/ckpt",
            **kwargs,
        )


def test_bind_generation_is_noop_without_enabled_caches():
    runner = _Runner()
    runner.condition_cache = None
    call_kwargs = {"prompt": "a", "num_frames": 8}

    binding = bind_generation(
        runner,
        prompt="a",
        media=(None, None),
        height=256,
        width=256,
        num_frames=8,
        call_kwargs=call_kwargs,
        denoise_cache=None,
        total_steps=8,
        transformer=object(),
    )

    assert binding.pipeline is runner.pipeline
    assert binding.call_kwargs == call_kwargs
    binding.observe(object())
    with binding.denoise() as controller:
        assert controller is None


def test_bind_generation_fails_closed_for_missing_denoise_cache():
    runner = _Runner()
    runner.condition_cache = None

    with pytest.raises(RuntimeError, match="not installed"):
        bind_generation(
            runner,
            prompt="a",
            media=(None, None),
            height=256,
            width=256,
            num_frames=8,
            call_kwargs={"prompt": "a"},
            denoise_cache=object(),
            total_steps=8,
            transformer=object(),
        )
