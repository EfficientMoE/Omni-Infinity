# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import soundfile
import torch

from omni_infinity.runner import GenerationResult
from omni_infinity.serve.artifacts import write_artifacts


def _result(**overrides):
    values = {
        "videos": [np.zeros((4, 16, 16, 3), dtype=np.float32)],
        "audio": torch.zeros(2, 1600, dtype=torch.float32),
        "sampling_rate": 16000,
        "latents": None,
        "audio_latents": None,
    }
    values.update(overrides)
    return GenerationResult(**values)


def test_artifact_writer_accepts_channel_last_stereo(tmp_path):
    audio = torch.zeros(1600, 2, dtype=torch.float32)
    audio[:, 0] = 0.25
    metadata = write_artifacts(_result(audio=audio), tmp_path, "/v1/jobs/x")

    info = soundfile.info(tmp_path / "output.wav")
    assert info.channels == 2
    assert info.samplerate == 16000
    samples, _ = soundfile.read(tmp_path / "output.wav")
    assert samples.shape[1] == 2
    assert metadata.video_url == "/v1/jobs/x"
    assert metadata.sampling_rate == 16000


def test_artifact_writer_squeezes_a_leading_batch_axis(tmp_path):
    audio = torch.zeros(1, 2, 800, dtype=torch.float32)
    write_artifacts(_result(audio=audio, sampling_rate=8000), tmp_path)
    assert soundfile.info(tmp_path / "output.wav").channels == 2
    assert soundfile.info(tmp_path / "output.wav").frames == 800


def test_artifact_writer_defaults_a_missing_sample_rate(tmp_path):
    metadata = write_artifacts(_result(sampling_rate=None), tmp_path)
    assert metadata.sampling_rate == 48000
    assert soundfile.info(tmp_path / "output.wav").samplerate == 48000


def test_artifact_writer_rejects_a_rank_one_waveform(tmp_path):
    with pytest.raises(ValueError, match="expected stereo audio"):
        write_artifacts(_result(audio=torch.zeros(800)), tmp_path)
    assert not (tmp_path / "output.wav").exists()


def test_artifact_writer_rejects_missing_or_flat_video(tmp_path):
    with pytest.raises(ValueError, match="no video"):
        write_artifacts(_result(videos=[]), tmp_path)
    with pytest.raises(ValueError, match="no video"):
        write_artifacts(_result(videos=None), tmp_path)
    flat = [np.zeros((16, 16, 3), dtype=np.float32)]
    with pytest.raises(ValueError, match="rank four"):
        write_artifacts(_result(videos=flat), tmp_path)
    assert not (tmp_path / "output.mp4").exists()
    assert not (tmp_path / "output.wav").exists()
