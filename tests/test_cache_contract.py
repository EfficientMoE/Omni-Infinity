# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from pydantic import ValidationError

from omni_infinity import registry
from omni_infinity.caches.attach import (
    attach_caches,
    bind_generation,
    denoise_config_from_args,
)
from omni_infinity.registry import OptimizationSpec
from omni_infinity.serve.models import GenerationRequest


class _Runner:
    def __init__(self):
        self.pipeline = object()


def test_static_optimization_set_excludes_caches():
    assert set(registry.OPTIMIZATIONS) == {
        "adaln-host-cache",
        "fp8",
        "fp4",
        "block-stream",
        "sage-attn",
        "compile-blocks",
        "cuda-graph",
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
    assert runner.encoder_cache_controller is None
    assert runner.cache_namespace == "ReferenceRunner:/ckpt"


@pytest.mark.parametrize(
    "cache_name", ["condition_cache", "vision_cache", "encoder_cache"]
)
def test_attach_caches_fails_closed_when_module_is_missing(
    cache_name, monkeypatch
):
    kwargs = {
        "condition_cache": False,
        "vision_cache": False,
        "encoder_cache": False,
    }
    kwargs[cache_name] = True
    module_name = {
        "condition_cache": "condition",
        "vision_cache": "vision",
        "encoder_cache": "prefix",
    }[cache_name]
    monkeypatch.setitem(
        sys.modules, f"omni_infinity.caches.{module_name}", None
    )

    with pytest.raises(RuntimeError, match="not installed"):
        attach_caches(
            _Runner(),
            condition_cache_dir=None,
            cache_namespace="ReferenceRunner:/ckpt",
            **kwargs,
        )


def test_attach_caches_surfaces_broken_vision_module(monkeypatch):
    module_name = "omni_infinity.caches.vision"
    module = types.ModuleType(module_name)

    def missing_dependency(name):
        if name == "enable_vision_cache":
            raise ModuleNotFoundError("No module named 'httpx'", name="httpx")
        raise AttributeError(name)

    module.__getattr__ = missing_dependency
    monkeypatch.setitem(sys.modules, module_name, module)

    with pytest.raises(ModuleNotFoundError, match="httpx") as exc_info:
        attach_caches(
            _Runner(),
            condition_cache=False,
            condition_cache_dir=None,
            vision_cache=True,
            cache_namespace="ReferenceRunner:/ckpt",
        )

    assert exc_info.value.name == "httpx"


def test_attach_caches_wraps_the_text_encoder_for_encoder_cache(monkeypatch):
    module_name = "omni_infinity.caches.prefix"
    module = types.ModuleType(module_name)
    text_encoder = object()
    controller = object()
    calls = []

    def enable_encoder_cache(value):
        calls.append(value)
        return controller

    module.enable_encoder_cache = enable_encoder_cache
    monkeypatch.setitem(sys.modules, module_name, module)
    runner = _Runner()
    runner.pipeline = SimpleNamespace(components={"text_encoder": text_encoder})

    attach_caches(
        runner,
        condition_cache=False,
        condition_cache_dir=None,
        vision_cache=False,
        encoder_cache=True,
        cache_namespace="ReferenceRunner:/ckpt",
    )

    assert calls == [text_encoder]
    assert runner.encoder_cache_controller is controller


def test_cache_optimization_loader_surfaces_broken_module(monkeypatch):
    import_module = registry.importlib.import_module

    def import_with_broken_vision(module_name):
        if module_name == "omni_infinity.caches.vision":
            raise ModuleNotFoundError("No module named 'httpx'", name="httpx")
        return import_module(module_name)

    monkeypatch.setattr(
        registry.importlib, "import_module", import_with_broken_vision
    )

    with pytest.raises(ModuleNotFoundError, match="httpx") as exc_info:
        registry.load_cache_optimizations()

    assert exc_info.value.name == "httpx"


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


def test_bind_generation_fails_closed_for_missing_denoise_cache(monkeypatch):
    runner = _Runner()
    runner.condition_cache = None
    monkeypatch.setitem(sys.modules, "omni_infinity.caches.denoise", None)

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


def test_request_accepts_cache_optimizations_and_rejects_denoise():
    GenerationRequest(
        type="fl2va",
        prompt="a",
        optimizations=["condition-cache", "encoder-cache", "vision-cache"],
    )
    with pytest.raises(ValidationError):
        GenerationRequest(
            type="fl2va", prompt="a", optimizations=["denoise-cache"]
        )


def test_denoise_config_requires_both_cli_values():
    empty = SimpleNamespace(
        denoise_cache_coefficients=None,
        denoise_cache_threshold=None,
    )
    assert denoise_config_from_args(empty) is None

    missing_threshold = SimpleNamespace(
        denoise_cache_coefficients="1.0,0.0",
        denoise_cache_threshold=None,
    )
    with pytest.raises(ValueError, match="both"):
        denoise_config_from_args(missing_threshold)


def test_denoise_config_parses_coefficients_in_cli_order(monkeypatch):
    module = types.ModuleType("omni_infinity.caches.denoise")

    class DenoiseCacheConfig:
        def __init__(
            self,
            *,
            coefficients,
            threshold,
            indicator,
            accumulate,
            approximator,
        ):
            self.coefficients = coefficients
            self.threshold = threshold
            self.indicator = indicator
            self.accumulate = accumulate
            self.approximator = approximator

    module.DenoiseCacheConfig = DenoiseCacheConfig
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = SimpleNamespace(
        denoise_cache_coefficients="2.0, 1.0,0.5",
        denoise_cache_threshold=0.2,
    )

    config = denoise_config_from_args(args)

    assert config.coefficients == (2.0, 1.0, 0.5)
    assert config.threshold == 0.2
    assert config.indicator == "raw"
    assert config.accumulate is False
    assert config.approximator == "reuse"

    v2_args = SimpleNamespace(
        denoise_cache_coefficients="2.0, 1.0,0.5",
        denoise_cache_threshold=0.2,
        denoise_cache_indicator="teacache",
        denoise_cache_accumulate=True,
        denoise_cache_mode="taylor1",
    )

    v2_config = denoise_config_from_args(v2_args)

    assert v2_config.indicator == "teacache"
    assert v2_config.accumulate is True
    assert v2_config.approximator == "taylor1"


def test_denoise_config_fails_closed_when_module_is_missing(monkeypatch):
    args = SimpleNamespace(
        denoise_cache_coefficients="1.0,0.0",
        denoise_cache_threshold=0.2,
    )
    monkeypatch.setitem(sys.modules, "omni_infinity.caches.denoise", None)
    with pytest.raises(RuntimeError, match="not installed"):
        denoise_config_from_args(args)


def test_smoke_parser_exposes_opt_in_cache_flags():
    spec = importlib.util.spec_from_file_location(
        "fl2va_smoke", "examples/fl2va_smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        [
            "--prompt",
            "a",
            "--condition-cache",
            "--condition-cache-dir",
            "/tmp/cache",
            "--vision-cache",
            "--encoder-cache",
            "--denoise-cache-coefficients",
            "1.0,0.0",
            "--denoise-cache-threshold",
            "0.2",
            "--denoise-cache-indicator",
            "fbcache",
            "--denoise-cache-accumulate",
            "--denoise-cache-mode",
            "taylor1",
        ]
    )

    assert args.condition_cache is True
    assert args.condition_cache_dir == "/tmp/cache"
    assert args.vision_cache is True
    assert args.encoder_cache is True
    assert args.denoise_cache_coefficients == "1.0,0.0"
    assert args.denoise_cache_threshold == 0.2
    assert args.denoise_cache_indicator == "fbcache"
    assert args.denoise_cache_accumulate is True
    assert args.denoise_cache_mode == "taylor1"


def test_parity_suites_do_not_enable_caches():
    for name in (
        "test_reference_parity.py",
        "test_ref2va_parity.py",
        "test_vdn_parity.py",
    ):
        text = (Path("tests") / name).read_text()
        assert "condition_cache" not in text
        assert "vision_cache" not in text
        assert "encoder_cache" not in text
        assert "denoise_cache" not in text
        assert "condition-cache" not in text
        assert "vision-cache" not in text
        assert "encoder-cache" not in text


def test_readme_points_at_the_cache_stack():
    text = Path("README.md").read_text()
    for sentence in (
        "Nothing is on by default.",
        "C2 is an exact encoder-prefix cache.",
        "C5 refuses to run without H3-calibrated coefficients.",
        "C4 is unchanged upstream.",
    ):
        assert sentence in text
