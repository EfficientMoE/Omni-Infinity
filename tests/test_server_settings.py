# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

from omni_infinity.registry import ResolvedProfile
from omni_infinity.serve.__main__ import main
from omni_infinity.serve.app import (
    ServerSettings,
    load_runner,
    parse_bool,
    parse_bytes,
)

_ENV_KEYS = (
    "OMNI_JOBS_DIR",
    "OMNI_MODEL_ARCH",
    "OMNI_OPTIMIZATIONS",
    "OMNI_CHECKPOINT",
    "OMNI_DEVICE",
    "OMNI_STORE_DIR",
    "OMNI_STORE_COMPONENTS",
    "OMNI_MAX_VRAM",
    "OMNI_CONDITION_CACHE_DIR",
    "OMNI_HOST",
    "OMNI_PORT",
    "OMNI_WORKERS",
    "OMNI_STREAM_ENABLED",
    "OMNI_STREAM_MAX_SESSIONS",
    "OMNI_STREAM_SESSION_TTL",
    "OMNI_STREAM_CHUNK_FRAMES",
    "OMNI_STREAM_QUEUE_CHUNKS",
    "OMNI_STREAM_FALLBACK_HLS",
)


@pytest.fixture
def clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("22GiB", 22 * 1024**3),
        ("  8GiB ", 8 * 1024**3),
        ("2GB", 2_000_000_000),
        ("1.5MiB", 1_572_864),
        ("3MB", 3_000_000),
        ("4096", 4096),
    ],
)
def test_parse_bytes_accepts_binary_and_decimal_suffixes(text, expected):
    assert parse_bytes(text) == expected


def test_parse_bytes_rejects_a_bare_word():
    with pytest.raises(ValueError):
        parse_bytes("lots")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),
        ("  ", False),
    ],
)
def test_parse_bool_accepts_the_documented_words(text, expected):
    assert parse_bool(text) is expected


def test_settings_from_env_use_the_streamed_dense_defaults(clean_env):
    settings = ServerSettings.from_env()
    assert settings.jobs_dir == Path("./jobs")
    assert settings.model_arch == "h3-dense"
    assert settings.optimizations == (
        "adaln-host-cache",
        "block-stream",
        "text-encoder-stream",
    )
    assert settings.checkpoint is None
    assert settings.device == "cuda"
    assert settings.store_dir is None
    assert settings.store_components == ("transformer", "vae", "audio_vae")
    assert settings.max_vram is None
    assert settings.condition_cache_dir is None
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.workers == 1
    assert settings.stream_enabled is False
    assert settings.stream_max_sessions == 1
    assert settings.stream_session_ttl == 30.0
    assert settings.stream_chunk_frames == 124
    assert settings.stream_queue_chunks == 8
    assert settings.stream_fallback_hls is False


def test_settings_from_env_split_lists_and_blank_paths(clean_env, monkeypatch):
    monkeypatch.setenv("OMNI_JOBS_DIR", "/tmp/jobs")
    monkeypatch.setenv("OMNI_MODEL_ARCH", "vdn-hybrid")
    monkeypatch.setenv("OMNI_OPTIMIZATIONS", " fp8, ,block-stream ")
    monkeypatch.setenv("OMNI_CHECKPOINT", "")
    monkeypatch.setenv("OMNI_DEVICE", "cuda:1")
    monkeypatch.setenv("OMNI_STORE_DIR", "")
    monkeypatch.setenv("OMNI_STORE_COMPONENTS", "vae, audio_vae,")
    monkeypatch.setenv("OMNI_MAX_VRAM", "22GiB")
    monkeypatch.setenv("OMNI_HOST", "0.0.0.0")
    monkeypatch.setenv("OMNI_PORT", "9001")
    monkeypatch.setenv("OMNI_WORKERS", "1")
    monkeypatch.setenv("OMNI_STREAM_ENABLED", "yes")
    monkeypatch.setenv("OMNI_STREAM_MAX_SESSIONS", "2")
    monkeypatch.setenv("OMNI_STREAM_SESSION_TTL", "5.5")
    monkeypatch.setenv("OMNI_STREAM_CHUNK_FRAMES", "12")
    monkeypatch.setenv("OMNI_STREAM_QUEUE_CHUNKS", "3")
    monkeypatch.setenv("OMNI_STREAM_FALLBACK_HLS", "on")

    settings = ServerSettings.from_env()

    assert settings.jobs_dir == Path("/tmp/jobs")
    assert settings.model_arch == "vdn-hybrid"
    assert settings.optimizations == ("fp8", "block-stream")
    assert settings.checkpoint is None
    assert settings.device == "cuda:1"
    assert settings.store_dir is None
    assert settings.store_components == ("vae", "audio_vae")
    assert settings.max_vram == "22GiB"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9001
    assert settings.stream_enabled is True
    assert settings.stream_max_sessions == 2
    assert settings.stream_session_ttl == 5.5
    assert settings.stream_chunk_frames == 12
    assert settings.stream_queue_chunks == 3
    assert settings.stream_fallback_hls is True


class _CaptureRunner:
    calls = []

    @classmethod
    def from_pretrained(cls, checkpoint, device="cuda", **kwargs):
        cls.calls.append((checkpoint, device, kwargs))
        return "runner"


def _profile(arch, checkpoint=None, **kwargs):
    default = {
        "h3-dense": "MiniMaxAI/MiniMax-H3",
        "vdn-hybrid": "OpenVDN/vdn-minimax-h3",
    }[arch]
    return ResolvedProfile(
        model_arch=arch,
        optimizations=(),
        runner=_CaptureRunner,
        checkpoint=checkpoint or default,
        runner_kwargs=dict(kwargs),
    )


def _stub_profile(monkeypatch, arch, **kwargs):
    def resolve(model_arch, optimizations, checkpoint=None):
        assert model_arch == arch
        return _profile(arch, checkpoint=checkpoint, **kwargs)

    monkeypatch.setattr("omni_infinity.serve.app.resolve_profile", resolve)


def test_load_runner_skips_the_margin_without_cuda(monkeypatch):
    _CaptureRunner.calls = []
    _stub_profile(monkeypatch, "h3-dense", offload=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    settings = ServerSettings(
        checkpoint="/ckpt",
        store_dir="/store",
        store_components=("transformer", "vae"),
        max_vram="22GiB",
        device="cuda",
    )

    assert load_runner(settings) == "runner"
    checkpoint, device, kwargs = _CaptureRunner.calls[-1]
    assert checkpoint == "/ckpt"
    assert device == "cuda"
    assert kwargs["store_dir"] == "/store"
    assert kwargs["store_components"] == ("transformer", "vae")
    assert kwargs["offload"] is True
    assert "offload_memory_margin" not in kwargs


def test_load_runner_sets_margin_on_a_gpu_larger_than_24_gib(monkeypatch):
    _CaptureRunner.calls = []
    _stub_profile(monkeypatch, "h3-dense")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class Props:
        total_memory = 40 * 1024**3

    seen = {}

    def properties(index):
        seen["index"] = index
        return Props()

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    settings = ServerSettings(
        max_vram="22GiB", device="cuda:1", checkpoint="/ckpt"
    )

    load_runner(settings)

    assert seen["index"] == 1
    assert _CaptureRunner.calls[-1][2]["offload_memory_margin"] == "19.3GB"


def test_load_runner_leaves_margin_unset_on_a_24_gib_gpu(monkeypatch):
    _CaptureRunner.calls = []
    _stub_profile(monkeypatch, "h3-dense")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class Props:
        total_memory = 24 * 1024**3

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: Props()
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    load_runner(ServerSettings(max_vram="22GiB", device="cuda"))

    assert "offload_memory_margin" not in _CaptureRunner.calls[-1][2]


def test_load_runner_forces_the_fl2va_workflow_for_vdn(monkeypatch):
    _CaptureRunner.calls = []
    _stub_profile(monkeypatch, "vdn-hybrid", fp8=True)

    load_runner(ServerSettings(model_arch="vdn-hybrid", store_dir="/store"))

    kwargs = _CaptureRunner.calls[-1][2]
    assert kwargs["workflow"] == "fl2va"
    assert kwargs["fp8"] is True
    assert "store_dir" not in kwargs


def test_condition_cache_dir_is_unset_by_default(clean_env):
    assert ServerSettings.from_env().condition_cache_dir is None


def test_load_runner_forwards_the_dir_only_with_the_flag(monkeypatch):
    _CaptureRunner.calls = []
    _stub_profile(monkeypatch, "h3-dense", condition_cache=True)
    load_runner(ServerSettings(condition_cache_dir="/tmp/cond"))
    assert _CaptureRunner.calls[-1][2]["condition_cache_dir"] == "/tmp/cond"

    _CaptureRunner.calls = []
    _stub_profile(monkeypatch, "h3-dense", offload=True)
    load_runner(ServerSettings(condition_cache_dir="/tmp/cond"))
    assert "condition_cache_dir" not in _CaptureRunner.calls[-1][2]


def test_main_rejects_more_than_one_worker(clean_env, monkeypatch):
    monkeypatch.setenv("OMNI_WORKERS", "2")
    with pytest.raises(ValueError, match="OMNI_WORKERS must be 1"):
        main()


def test_main_rejects_unwired_split_roles(clean_env, monkeypatch):
    monkeypatch.setenv("OMNI_ROLE", "denoiser")
    with pytest.raises(NotImplementedError, match="OMNI_ROLE"):
        main()


def test_main_binds_the_configured_host_and_port(clean_env, monkeypatch):
    captured = {}

    def run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setenv("OMNI_HOST", "0.0.0.0")
    monkeypatch.setenv("OMNI_PORT", "9001")
    monkeypatch.setattr("omni_infinity.serve.__main__.uvicorn.run", run)

    main()

    assert captured["args"] == ("omni_infinity.serve.app:create_app",)
    assert captured["kwargs"]["factory"] is True
    assert captured["kwargs"]["host"] == "0.0.0.0"
    assert captured["kwargs"]["port"] == 9001
    assert captured["kwargs"]["workers"] == 1
