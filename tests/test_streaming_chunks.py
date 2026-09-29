# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from omni_infinity.streaming import MediaChunk, StreamRequest, active_cue


def test_active_cue_is_half_open_and_empty_is_none():
    cues = [
        MediaChunk(0, 0.0, 1.0, True, None, None, "a"),
        MediaChunk(1, 1.0, 1.0, True, None, None, "b"),
    ]
    assert active_cue([], 0.0) is None
    assert active_cue(cues, -0.1) is None
    assert active_cue(cues, 0.0).prompt == "a"
    assert active_cue(cues, 0.999).prompt == "a"
    assert active_cue(cues, 1.0).prompt == "b"
    assert active_cue(cues, 2.0) is None


def test_stream_request_defaults_and_rejects_a_long_script():
    request = StreamRequest(type="fl2va", prompt="a red ball bouncing")
    assert request.source == "clip"
    assert request.action_script == []
    assert request.resolution == "256p"
    with pytest.raises(ValidationError):
        StreamRequest(
            type="fl2va",
            prompt="p",
            action_script=[{"t": 0, "action": "x" * 65, "instruction": "go"}],
        )
