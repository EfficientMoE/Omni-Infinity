# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import io

import av
import numpy as np
import pytest
import torch

from omni_infinity.streaming import fragment_clip


def test_fragment_clip_keyframes_and_round_trips():
    frames = np.zeros((5, 16, 16, 3), dtype=np.float32)
    frames[:, :, :, 1] = np.linspace(0.0, 1.0, 5)[:, None, None]
    audio = torch.zeros(2, 48000, dtype=torch.float32)
    init, fragments = fragment_clip(
        frames, audio, 48000, fps=24, chunk_frames=2
    )
    assert init.startswith(b"\x00\x00\x00")
    assert len(fragments) == 3
    assert [item.keyframe for item in fragments] == [True, True, True]
    assert fragments[0].pts == 0.0
    assert fragments[1].pts == pytest.approx(2 / 24)
    assert fragments[2].duration == pytest.approx(1 / 24)
    assert all(item.video_bytes.startswith(b"\x00\x00") for item in fragments)
    container = av.open(
        io.BytesIO(init + b"".join(item.video_bytes for item in fragments))
    )
    decoded = sum(1 for _ in container.decode(video=0))
    container.close()
    assert decoded == 5
    assert b"moov" in init
    assert all(b"moof" in item.video_bytes for item in fragments)
    assert b"mfra" not in fragments[-1].video_bytes


def test_fragment_clip_rejects_invalid_inputs():
    frames = np.zeros((2, 16, 16, 3), dtype=np.float32)
    audio = torch.zeros(2, 48000, dtype=torch.float32)

    with pytest.raises(ValueError, match="frames"):
        fragment_clip(frames[:, :, :, :2], audio, 48000)
    with pytest.raises(ValueError, match="audio"):
        fragment_clip(frames, audio[:1], 48000)
    with pytest.raises(ValueError, match="fps"):
        fragment_clip(frames, audio, 48000, fps=0)
