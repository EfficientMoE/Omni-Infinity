# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Bitwise golden-latent parity against the recorded reference run.

Goldens are produced by ``examples/fl2va_smoke.py --record-goldens
tests/fixtures/goldens`` (256p/120-frame latents are <1 MB and are
committed). The test needs a GPU and skips unless the local torch version
and GPU model match the recording — bitwise reproduction is only defined
on the same kernel stack. Re-record on stack upgrades.
"""

import os
from pathlib import Path

import pytest
import torch

GOLDENS = Path(__file__).parent / "fixtures" / "goldens" / "fl2va_goldens.pt"


@pytest.mark.skipif(not GOLDENS.is_file(), reason="golden fixtures absent")
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="reference parity needs a GPU"
)
def test_reference_runner_reproduces_golden_latents():
    from omni_infinity.runner import ReferenceRunner

    payload = torch.load(GOLDENS, weights_only=False)
    if payload["torch_version"] != torch.__version__:
        pytest.skip("goldens recorded on a different torch; re-record")
    recorded_gpu = payload.get("gpu_name")
    if recorded_gpu and recorded_gpu != torch.cuda.get_device_name(0):
        pytest.skip(
            f"goldens recorded on {recorded_gpu!r}; bitwise parity is only "
            "guaranteed on the same GPU model"
        )

    runner = ReferenceRunner.from_pretrained(
        os.environ.get("OMNI_H3_CHECKPOINT", "MiniMaxAI/MiniMax-H3"),
        offload=os.environ.get("OMNI_H3_OFFLOAD", "0") == "1",
        store_dir=os.environ.get("OMNI_H3_STORE"),
        store_components=tuple(
            os.environ.get("OMNI_H3_STORE_COMPONENTS", "vae,audio_vae").split(
                ","
            )
        ),
        adaln_host_cache=os.environ.get("OMNI_H3_ADALN_CACHE", "0") == "1",
    )
    result = runner.generate(
        payload["prompt"],
        seed=payload["seed"],
        num_inference_steps=payload["steps"],
        resolution=payload["resolution"],
        num_frames=payload["frames"],
    )

    assert torch.equal(result.latents.cpu(), payload["latents"])
    if payload["audio_latents"] is not None:
        assert torch.equal(result.audio_latents.cpu(), payload["audio_latents"])
