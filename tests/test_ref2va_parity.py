# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Bitwise Ref2VA parity against the full-resident golden run."""

import hashlib
import os
from pathlib import Path

import diffusers
import pytest
import torch

GOLDENS = Path(__file__).parent / "fixtures" / "goldens" / "ref2va_goldens.pt"
REFERENCE = Path(__file__).parent / "fixtures" / "ref.png"


@pytest.mark.skipif(
    not GOLDENS.is_file(), reason="Ref2VA golden fixture absent"
)
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Ref2VA parity needs a GPU"
)
def test_ref2va_runner_reproduces_full_resident_golden_latents():
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

    from omni_infinity.runner import ReferenceRunner

    payload = torch.load(GOLDENS, weights_only=False)
    if payload["torch_version"] != torch.__version__:
        pytest.skip("goldens recorded on a different torch; re-record")
    if payload["diffusers_version"] != diffusers.__version__:
        pytest.skip("goldens recorded with different diffusers; re-record")
    recorded_gpu = payload.get("gpu_name")
    if recorded_gpu and recorded_gpu != torch.cuda.get_device_name(0):
        pytest.skip(
            f"goldens recorded on {recorded_gpu!r}; bitwise parity is only "
            "guaranteed on the same GPU model"
        )
    reference_sha = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    if payload["reference_sha256"] != reference_sha:
        pytest.skip("reference fixture SHA differs from the golden; re-record")

    runner = ReferenceRunner.from_pretrained(
        os.environ.get("OMNI_H3_CHECKPOINT", "MiniMaxAI/MiniMax-H3"),
        workflow="ref2va",
        offload=os.environ.get("OMNI_H3_OFFLOAD", "1") == "1",
        store_dir=os.environ.get("OMNI_H3_STORE"),
        store_components=tuple(
            os.environ.get(
                "OMNI_H3_STORE_COMPONENTS", "transformer_ref,vae,audio_vae"
            ).split(",")
        ),
        adaln_host_cache=os.environ.get("OMNI_H3_ADALN_CACHE", "0") == "1",
        block_stream_blocks_per_group=int(
            os.environ.get("OMNI_H3_BLOCK_STREAM", "0")
        ),
        stream_text_encoder=os.environ.get("OMNI_H3_STREAM_TEXT_ENCODER", "0")
        == "1",
        step_overlap=os.environ.get("OMNI_H3_STEP_OVERLAP", "0") == "1",
    )
    result = runner.generate(
        payload["prompt"],
        references=[MiniMaxH3ImageReference.from_file(REFERENCE)],
        seed=payload["seed"],
        num_inference_steps=payload["steps"],
        resolution=payload["resolution"],
        num_frames=payload["frames"],
    )

    assert payload["frames"] >= 120
    assert torch.equal(result.latents.cpu(), payload["latents"])
    assert torch.equal(result.audio_latents.cpu(), payload["audio_latents"])
