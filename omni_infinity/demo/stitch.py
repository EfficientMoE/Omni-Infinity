# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import numpy as np
import torch
from PIL import Image

from omni_infinity.runner import GenerationResult
from omni_infinity.serve.artifacts import _stereo_audio, _video_frames


def last_frame_image(frames: np.ndarray) -> Image.Image:
    # Accept either a stack of frames (F,H,W,C) or a single frame (H,W,C)
    if frames.ndim == 4:
        frame = frames[-1]
    elif frames.ndim == 3:
        frame = frames
    else:
        raise ValueError(
            f"Unsupported frame ndim={frames.ndim}, expected 3 or 4"
        )

    if frame.dtype == np.uint8:
        img_array = frame.copy()
    else:
        # Clip float to [0,1] and convert to uint8
        img_array = np.clip(frame, 0, 1) * 255
        img_array = np.round(img_array).astype(np.uint8)
    img_array = np.ascontiguousarray(img_array)
    return Image.fromarray(img_array, mode="RGB")


def stitch_results(
    results: Sequence[GenerationResult], *, playback_frames: int
) -> GenerationResult:
    if not results:
        raise ValueError("No results to stitch")

    # Extract sampling rate from first result
    sampling_rate = results[0].sampling_rate

    videos = []
    audios = []

    for r in results:
        if r.sampling_rate != sampling_rate:
            raise ValueError(
                f"Sampling rate mismatch: {r.sampling_rate} != {sampling_rate}"
            )
        videos.append(_video_frames(r.videos))
        audios.append(_stereo_audio(r.audio))

    video_concat = np.concatenate(videos, axis=0)
    audio_concat = torch.cat(audios, dim=1)

    if video_concat.shape[0] < playback_frames:
        raise ValueError(
            (
                "Concatenated video length {} is less than playback_frames {}"
            ).format(video_concat.shape[0], playback_frames)
        )

    # Trim video frames
    video_trim = video_concat[:playback_frames]

    # Calculate audio trim length: playback_frames / 24 * sampling_rate, rounded
    audio_trim_length = round(playback_frames / 24 * sampling_rate)

    audio_trim = audio_concat[:, :audio_trim_length]

    return GenerationResult(
        videos=[video_trim],
        audio=audio_trim,
        sampling_rate=sampling_rate,
        latents=None,
        audio_latents=None,
    )
