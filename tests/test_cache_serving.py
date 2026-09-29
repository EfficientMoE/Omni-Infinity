# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Issue #24 caches through the job-serving surface.

The server keeps its one-profile contract: a request must repeat the
loaded profile exactly, which is also SGLang's rule that requests with
different cache settings never share a batch — here they cannot even
share a server.
"""

import pytest

pytest.importorskip("fastapi")

from pydantic import ValidationError  # noqa: E402

from omni_infinity.serve.app import ServerSettings, load_runner  # noqa: E402
from omni_infinity.serve.models import GenerationRequest  # noqa: E402


def test_request_accepts_the_cache_optimizations():
    request = GenerationRequest(
        type="fl2va",
        prompt="a red ball bouncing",
        optimizations=["condition-cache", "vision-cache"],
    )
    assert request.optimizations == ["condition-cache", "vision-cache"]


def test_request_rejects_the_unregistered_denoise_cache():
    # C5 is approximate and needs calibration; it is not requestable.
    with pytest.raises(ValidationError):
        GenerationRequest(
            type="fl2va",
            prompt="a red ball bouncing",
            optimizations=["denoise-cache"],
        )


def test_default_request_and_server_profile_have_no_caches():
    request = GenerationRequest(type="fl2va", prompt="p")
    assert request.optimizations == []
    settings = ServerSettings()
    assert "condition-cache" not in settings.optimizations
    assert "vision-cache" not in settings.optimizations


def test_settings_read_the_condition_cache_dir(monkeypatch):
    monkeypatch.setenv("OMNI_CONDITION_CACHE_DIR", "/var/cache/omni")
    settings = ServerSettings.from_env()
    assert settings.condition_cache_dir == "/var/cache/omni"
    monkeypatch.delenv("OMNI_CONDITION_CACHE_DIR")
    assert ServerSettings.from_env().condition_cache_dir is None


def test_load_runner_threads_the_cache_dir(monkeypatch):
    captured = {}

    class FakeRunner:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            captured["checkpoint"] = checkpoint
            captured["kwargs"] = kwargs
            return cls()

    import omni_infinity.registry as registry

    monkeypatch.setattr(
        registry,
        "runner_class",
        lambda spec: FakeRunner,
    )
    settings = ServerSettings(
        model_arch="h3-dense",
        optimizations=("condition-cache", "vision-cache"),
        condition_cache_dir="/var/cache/omni",
        device="cpu",
    )
    load_runner(settings)
    assert captured["kwargs"]["condition_cache"] is True
    assert captured["kwargs"]["vision_cache"] is True
    assert captured["kwargs"]["condition_cache_dir"] == "/var/cache/omni"


def test_load_runner_without_the_optimization_passes_no_cache_dir(
    monkeypatch,
):
    captured = {}

    class FakeRunner:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            captured["kwargs"] = kwargs
            return cls()

    import omni_infinity.registry as registry

    monkeypatch.setattr(registry, "runner_class", lambda spec: FakeRunner)
    settings = ServerSettings(
        model_arch="h3-dense",
        optimizations=(),
        condition_cache_dir="/var/cache/omni",
        device="cpu",
    )
    load_runner(settings)
    assert "condition_cache_dir" not in captured["kwargs"]
    assert "condition_cache" not in captured["kwargs"]
