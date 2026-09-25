# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile
import torch

from omni_infinity.serve.models import ArtifactMetadata

if TYPE_CHECKING:
    from omni_infinity.runner import GenerationResult


def _stereo_audio(audio: Any) -> torch.Tensor:
    waveform = torch.as_tensor(audio).detach().cpu()
    while waveform.ndim > 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    if waveform.ndim != 2:
        raise ValueError("expected stereo audio")
    if waveform.shape[0] == 2:
        stereo = waveform
    elif waveform.shape[1] == 2:
        stereo = waveform.transpose(0, 1)
    else:
        raise ValueError("expected stereo audio")
    return stereo.to(dtype=torch.float32).contiguous()


def _video_frames(videos: Any) -> Any:
    if videos is None or len(videos) == 0:
        raise ValueError("generation result contains no video")
    frames = videos[0]
    if isinstance(frames, np.ndarray) and frames.ndim != 4:
        raise ValueError("expected video frames with rank four")
    return frames


def _encode_video(*args, **kwargs) -> None:
    try:
        from diffusers.utils.export_utils import encode_video
    except ImportError as exc:
        raise ImportError(
            "diffusers >= 0.40 is required to encode serving artifacts"
        ) from exc
    encode_video(*args, **kwargs)


def write_artifacts(
    result: GenerationResult,
    job_dir: Path,
    artifact_url: str = "",
) -> ArtifactMetadata:
    directory = Path(job_dir)
    directory.mkdir(parents=True, exist_ok=True)
    partial_wav = directory / "output.partial.wav"
    final_wav = directory / "output.wav"
    partial_mp4 = directory / "output.partial.mp4"
    final_mp4 = directory / "output.mp4"
    sampling_rate = int(result.sampling_rate or 48000)

    stereo = _stereo_audio(result.audio)
    frames = _video_frames(result.videos)
    try:
        soundfile.write(
            partial_wav,
            stereo.transpose(0, 1).numpy(),
            sampling_rate,
            format="WAV",
            subtype="PCM_16",
        )
        _encode_video(
            frames,
            fps=24,
            output_path=str(partial_mp4),
            audio=stereo,
            audio_sample_rate=sampling_rate,
        )
        os.replace(partial_wav, final_wav)
        os.replace(partial_mp4, final_mp4)
    except Exception:
        for path in (partial_wav, partial_mp4, final_wav, final_mp4):
            path.unlink(missing_ok=True)
        raise

    return ArtifactMetadata(
        video_url=artifact_url,
        audio_channels=2,
        sampling_rate=sampling_rate,
    )
