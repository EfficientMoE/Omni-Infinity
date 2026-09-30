# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import pytest

from omni_infinity.demo.schedule import (
    generated_frames,
    playback_frames,
    prompt_times,
    segment_count,
)


@pytest.mark.parametrize(
    ("seconds", "segments", "play", "generated"),
    [
        (15, 3, 360, 372),
        (60, 12, 1440, 1488),
        (120, 24, 2880, 2976),
        (300, 59, 7200, 7316),
    ],
)
def test_covering_counts(seconds, segments, play, generated):
    assert segment_count(seconds) == segments
    assert playback_frames(seconds) == play
    assert generated_frames(seconds) == generated


def test_short_counts_do_not_cover():
    assert segment_count(120) != 23
    assert segment_count(300) != 58


def test_prompt_times_are_inclusive():
    assert prompt_times(4, 4, 4, seed=0) == [None, 4, 4, 4]
    assert prompt_times(3, 0, 0, seed=1) == [None, 0, 0]
    for value in prompt_times(8, 2, 5, seed=7):
        assert value is None or value in range(2, 6)
