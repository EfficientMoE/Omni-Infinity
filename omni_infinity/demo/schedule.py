# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import math

FPS: int = 24
SEGMENT_FRAMES: int = 124

PRESETS: dict[str, int] = {
    "15s": 15,
    "1min": 60,
    "2min": 120,
    "5min": 300,
}


def segment_count(duration_s: int) -> int:
    return math.ceil(duration_s * FPS / SEGMENT_FRAMES)


def playback_frames(duration_s: int) -> int:
    return duration_s * FPS


def generated_frames(duration_s: int) -> int:
    return segment_count(duration_s) * SEGMENT_FRAMES
