# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

import omni_infinity.demo.stitch as stitch
from omni_infinity.runner import GenerationResult


def make_result(frames_len, scale=1.0, sampling_rate=24):
    # Video frames: float32, shape
    # (frames_len, 2, 2, 3), values scaled into [0, 1]
    frames = np.arange(
        frames_len * 2 * 2 * 3,
        dtype=np.float32,
    ).reshape((frames_len, 2, 2, 3))
    frames = frames / (frames_len * 2 * 2 * 3 - 1) * scale

    # Audio: torch.ones(2, 8)
    audio = torch.ones(2, 8)

    return GenerationResult(
        videos=[frames],
        audio=audio,
        sampling_rate=sampling_rate,
        latents=None,
        audio_latents=None,
    )


def test_stitch_results_basic():
    # Build two GenerationResults
    result0 = make_result(4, scale=1.0)
    result1 = make_result(4, scale=0.5)

    # Stitch with playback_frames=6
    stitched = stitch.stitch_results([result0, result1], playback_frames=6)

    # Assert video shape is 6 frames
    assert stitched.videos[0].shape[0] == 6

    # Assert the first 4 frames equal result0 frames
    np.testing.assert_array_equal(stitched.videos[0][:4], result0.videos[0])

    # Assert the next 2 frames equal the first two frames of result1
    np.testing.assert_array_equal(
        stitched.videos[0][4:6], result1.videos[0][:2]
    )

    # Assert audio shape is (2, 6)
    assert stitched.audio.shape == (2, 6)

    # Assert latents and audio_latents are None
    assert stitched.latents is None
    assert stitched.audio_latents is None


def test_stitch_results_uses_disk_backed_video_without_concatenate(monkeypatch):
    results = [make_result(4), make_result(4, scale=0.5)]

    def reject_concatenate(*args, **kwargs):
        raise AssertionError(
            "stitching must not materialize a concatenated copy"
        )

    monkeypatch.setattr(stitch.np, "concatenate", reject_concatenate)

    stitched = stitch.stitch_results(results, playback_frames=6)

    assert isinstance(stitched.videos[0], np.memmap)
    np.testing.assert_array_equal(stitched.videos[0][:4], results[0].videos[0])
    np.testing.assert_array_equal(
        stitched.videos[0][4:], results[1].videos[0][:2]
    )


def test_last_frame_image():
    # Create a single frame of shape (1, 1, 3) with float values 1.0 (white)
    frame = np.ones((1, 1, 3), dtype=np.float32)

    img = stitch.last_frame_image(frame)
    assert img.mode == "RGB"
    assert img.getpixel((0, 0)) == (255, 255, 255)


def test_last_frame_image_uint8_copy():
    frame = np.full((1, 1, 3), (10, 20, 30), dtype=np.uint8)

    img = stitch.last_frame_image(frame)

    assert img.mode == "RGB"
    assert img.getpixel((0, 0)) == (10, 20, 30)
    frame[0, 0] = (0, 0, 0)
    assert img.getpixel((0, 0)) == (10, 20, 30)


def test_last_frame_image_stack_returns_last():
    frames = np.zeros((3, 1, 1, 3), dtype=np.float32)
    frames[2] = 1.0

    img = stitch.last_frame_image(frames)

    assert img.getpixel((0, 0)) == (255, 255, 255)


def test_stitch_rejects_short_video():
    results = [make_result(4), make_result(4)]

    with pytest.raises(ValueError, match="less than playback_frames"):
        stitch.stitch_results(results, playback_frames=100)


def test_sampling_rate_mismatch_raises():
    result0 = make_result(4, sampling_rate=24)
    result1 = make_result(4, sampling_rate=16_000)

    with pytest.raises(ValueError):
        stitch.stitch_results([result0, result1], playback_frames=6)
