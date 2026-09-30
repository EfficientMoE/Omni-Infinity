# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
from collections.abc import Iterable

import numpy as np
import torch
from PIL import Image

from omni_infinity.demo.schedule import FPS
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
            f"unsupported frame ndim={frames.ndim}, expected 3 or 4"
        )

    if frame.dtype == np.uint8:
        img_array = frame.copy()
    else:
        img_array = np.clip(frame, 0, 1) * 255
        img_array = np.round(img_array).astype(np.uint8)
    return Image.fromarray(img_array, mode="RGB")


class ResultStitcher:
    """Incrementally assemble clips into bounded, disk-backed storage."""

    def __init__(self, playback_frames: int):
        self.playback_frames = playback_frames
        self.sampling_rate: int | None = None
        self.video: np.memmap | None = None
        self.audio: torch.Tensor | None = None
        self.video_frames = 0
        self.audio_samples = 0
        self.result_count = 0

    def append(self, result: GenerationResult) -> None:
        frames = _video_frames(result.videos)
        if not isinstance(frames, np.ndarray):
            raise ValueError("expected video frames as a numpy array")
        stereo = _stereo_audio(result.audio)

        if self.sampling_rate is None:
            self.sampling_rate = result.sampling_rate
            with tempfile.TemporaryFile() as backing_file:
                self.video = np.memmap(
                    backing_file,
                    dtype=frames.dtype,
                    mode="w+",
                    shape=(self.playback_frames, *frames.shape[1:]),
                )
            audio_length = round(
                self.playback_frames / FPS * self.sampling_rate
            )
            self.audio = torch.empty((2, audio_length), dtype=torch.float32)
        elif result.sampling_rate != self.sampling_rate:
            raise ValueError(
                f"sampling rate mismatch: {result.sampling_rate} "
                f"!= {self.sampling_rate}"
            )

        assert self.video is not None
        assert self.audio is not None
        if frames.shape[1:] != self.video.shape[1:]:
            raise ValueError("video frame shape mismatch")
        if frames.dtype != self.video.dtype:
            raise ValueError("video frame dtype mismatch")

        video_count = min(len(frames), self.playback_frames - self.video_frames)
        if video_count > 0:
            end = self.video_frames + video_count
            self.video[self.video_frames : end] = frames[:video_count]
            self.video_frames = end

        audio_count = min(
            stereo.shape[1], self.audio.shape[1] - self.audio_samples
        )
        if audio_count > 0:
            end = self.audio_samples + audio_count
            self.audio[:, self.audio_samples : end] = stereo[:, :audio_count]
            self.audio_samples = end
        self.result_count += 1

    def finish(self) -> GenerationResult:
        if self.result_count == 0:
            raise ValueError("no results to stitch")
        assert self.video is not None
        assert self.audio is not None
        assert self.sampling_rate is not None
        if self.video_frames < self.playback_frames:
            raise ValueError(
                f"concatenated video length {self.video_frames} "
                f"is less than playback_frames {self.playback_frames}"
            )
        self.video.flush()
        return GenerationResult(
            videos=[self.video],
            audio=self.audio[:, : self.audio_samples],
            sampling_rate=self.sampling_rate,
            latents=None,
            audio_latents=None,
        )


def stitch_results(
    results: Iterable[GenerationResult], *, playback_frames: int
) -> GenerationResult:
    stitcher = ResultStitcher(playback_frames)
    for result in results:
        stitcher.append(result)
    return stitcher.finish()
