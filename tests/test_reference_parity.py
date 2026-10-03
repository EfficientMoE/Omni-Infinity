# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Bitwise golden-latent parity against the recorded reference run.

Goldens are produced by ``examples/fl2va_smoke.py --first-frame
tests/fixtures/ref.png --record-goldens tests/fixtures/goldens`` (256p/
120-frame latents are <1 MB and are committed). FL2VA requires a keyframe;
an image-less request routes through the separate ``t2va`` graph upstream.
The test needs a GPU and skips unless the local torch/diffusers versions
and GPU model match the recording — bitwise reproduction is only defined
on the same kernel stack. Re-record on stack upgrades.
"""

import hashlib
import os
from pathlib import Path

import diffusers
import pytest
import torch

GOLDENS = Path(__file__).parent / "fixtures" / "goldens" / "fl2va_goldens.pt"
REFERENCE = Path(__file__).parent / "fixtures" / "ref.png"


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.skipif(not GOLDENS.is_file(), reason="golden fixtures absent")
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="reference parity needs a GPU"
)
def test_reference_runner_reproduces_golden_latents():
    from omni_infinity.runner import ReferenceRunner

    payload = torch.load(GOLDENS, weights_only=False)
    if payload["torch_version"] != torch.__version__:
        pytest.skip("goldens recorded on a different torch; re-record")
    golden_diffusers = payload.get("diffusers_version")
    if golden_diffusers != diffusers.__version__:
        pytest.skip(
            "goldens recorded with a different diffusers "
            f"(recorded={golden_diffusers!r}, installed="
            f"{diffusers.__version__!r}); bitwise parity is only defined on "
            "the recording stack — re-record"
        )
    recorded_gpu = payload.get("gpu_name")
    if recorded_gpu and recorded_gpu != torch.cuda.get_device_name(0):
        pytest.skip(
            f"goldens recorded on {recorded_gpu!r}; bitwise parity is only "
            "guaranteed on the same GPU model"
        )
    reference_sha = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    if payload.get("first_frame_sha256") != reference_sha:
        pytest.skip(
            "first-frame fixture SHA differs from the golden; re-record"
        )

    runner = ReferenceRunner.from_pretrained(
        os.environ.get("OMNI_H3_CHECKPOINT", "MiniMaxAI/MiniMax-H3"),
        # Default on: the full FL2VA set exceeds a single GPU, so parity runs
        # the offload path (matching Ref2VA). Set OMNI_H3_OFFLOAD=0 only when
        # the whole pipeline fits resident.
        offload=os.environ.get("OMNI_H3_OFFLOAD", "1") == "1",
        store_dir=os.environ.get("OMNI_H3_STORE"),
        store_components=tuple(
            os.environ.get(
                "OMNI_H3_STORE_COMPONENTS", "transformer,vae,audio_vae"
            ).split(",")
        ),
        adaln_host_cache=os.environ.get("OMNI_H3_ADALN_CACHE", "0") == "1",
        block_stream_blocks_per_group=int(
            os.environ.get("OMNI_H3_BLOCK_STREAM", "0")
        ),
        stream_text_encoder=os.environ.get("OMNI_H3_STREAM_TEXT_ENCODER", "0")
        == "1",
    )
    from PIL import Image

    result = runner.generate(
        payload["prompt"],
        image=Image.open(REFERENCE).convert("RGB"),
        seed=payload["seed"],
        num_inference_steps=payload["steps"],
        resolution=payload["resolution"],
        num_frames=payload["frames"],
    )

    assert torch.equal(result.latents.cpu(), payload["latents"])
    if payload["audio_latents"] is not None:
        assert torch.equal(result.audio_latents.cpu(), payload["audio_latents"])
