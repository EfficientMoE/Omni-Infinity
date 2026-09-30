# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def _text(path: str) -> str:
    return " ".join(Path(path).read_text(encoding="utf-8").split())


def test_readme_matches_serving_and_ci():
    readme = _text("README.md")
    assert "until Task 3 / issue #8 lands" not in readme
    for needle in (
        "both return HTTP 501",
        "examples/ref2va_smoke.py",
        "Unknown job IDs return HTTP 404",
        "`OMNI_WORKERS` must stay `1`",
        "init_b64",
        "video_b64",
        "audio_b64",
        "audio is muxed into the fMP4 fragment",
        "artifact_url",
        '"type": "error"',
        "checked when the next session is created",
        "docs/streaming_chunk_playback.md",
        "docs/streaming_bench.md",
        "pip install -e '.[dev]'",
        '-m "not gpu and not weights"',
        "--cov-fail-under=80",
        "--interactive",
    ):
        assert needle in readme


def test_streaming_design_matches_landed_contract():
    doc = _text("docs/streaming_chunk_playback.md")
    assert "It does not land the feature." not in doc
    assert "optional AAC fragment" not in doc
    for needle in (
        "OMNI_STREAM_SESSION_TTL",
        "/v1/streams/{id}/init.mp4",
        "audio is muxed into the fMP4 fragment",
        "already listed in the `serve` and `dev` extras",
        "checked when the next session is created",
    ):
        assert needle in doc


def test_streaming_bench_rerun_uses_a_route_bearing_server():
    doc = _text("docs/streaming_bench.md")
    assert "not present on `bench/stream-*`" not in doc
    assert "OMNI_STREAM_ENABLED=1 python -m omni_infinity.serve &" not in doc
    for needle in (
        "bench/stream-harness",
        "bench/stream-baseline",
        "bench/stream-micro",
        "bench/stream-ablation",
        "--base-url",
        "OMNI_CHECKPOINT",
        "OMNI_STORE_DIR",
        "OMNI_MAX_VRAM",
        "ablation.py writes a skip manifest",
        "restart_command",
    ):
        assert needle in doc


def test_docker_readme_matches_ci_markers():
    doc = _text("docker/README.md")
    assert '-k "not parity"' not in doc
    assert '-m "not gpu and not weights"' in doc
    assert ".github/workflows/ci.yml" in doc
    assert "pytest-cov" in doc
