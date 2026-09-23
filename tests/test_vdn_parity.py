# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise parity gate for the vdn-hybrid arch (bf16 reference path)."""

import os
import pathlib

import pytest
import torch

GOLDENS = pathlib.Path(__file__).parent / "fixtures" / "vdn_goldens"
CHECKPOINT = os.environ.get("VDN_CHECKPOINT")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(CHECKPOINT is None, reason="set VDN_CHECKPOINT to run"),
    pytest.mark.skipif(
        not (GOLDENS / "vdn_goldens.pt").exists(),
        reason="record goldens with examples/vdn_smoke.py first",
    ),
]


def test_vdn_bf16_reproduces_goldens_bitwise():
    from omni_infinity.arch.vdn import VdnRunner

    golden = torch.load(GOLDENS / "vdn_goldens.pt", weights_only=False)
    if golden["torch_version"] != torch.__version__:
        pytest.skip("goldens recorded on a different torch; re-record")
    if golden["gpu_name"] != torch.cuda.get_device_name(0):
        pytest.skip(
            "goldens recorded on a different GPU model; bitwise "
            "parity is only guaranteed on the same model"
        )
    runner = VdnRunner.from_pretrained(
        CHECKPOINT,
        # 66 GB transformer resident; encoder leaf-streamed and
        # decoders hooked (bitwise-safe device-only moves) so the
        # full component set fits a 96 GB card.
        offload=True,
        stream_text_encoder=True,
    )
    result = runner.generate(
        golden["prompt"],
        seed=golden["seed"],
        num_evaluations=golden["evals"],
        num_frames=golden["frames"],
    )
    assert torch.equal(result.latents.cpu(), golden["latents"].cpu())
    assert torch.equal(
        result.audio_latents.cpu(), golden["audio_latents"].cpu()
    )
