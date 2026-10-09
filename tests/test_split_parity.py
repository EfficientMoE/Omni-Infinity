# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise parity gate: split-stage execution vs the one-shot call.

P7 Phase 1 contract: running the five top-level modular block groups
stage-by-stage (``run_all_stages``) on one device must be
data-movement-only — bitwise-identical video/audio latents to
``pipeline(**call_kwargs)`` via ``ReferenceRunner.generate``.  Needs a
GPU and the real H3 checkpoint (same env knobs as
``test_reference_parity.py``); no caches are enabled so the generation
binding is pass-through on both paths.
"""

import os
from pathlib import Path

import pytest
import torch

from omni_infinity.serve.split import run_all_stages

REFERENCE = Path(__file__).parent / "fixtures" / "ref.png"


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="split parity needs a GPU"
)
def test_split_stages_match_one_shot_latents_bitwise():
    from PIL import Image

    from omni_infinity.runner import ReferenceRunner, resolve_resolution

    runner = ReferenceRunner.from_pretrained(
        os.environ.get("OMNI_H3_CHECKPOINT", "MiniMaxAI/MiniMax-H3"),
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
    prompt = "a red ball bouncing"
    seed = 0
    steps = int(os.environ.get("OMNI_H3_STEPS", "8"))
    resolution = "256p"
    frames = 120
    image = Image.open(REFERENCE).convert("RGB")

    one_shot = runner.generate(
        prompt,
        image=image,
        seed=seed,
        num_inference_steps=steps,
        resolution=resolution,
        num_frames=frames,
    )

    height, width = resolve_resolution(resolution)
    state = run_all_stages(
        runner.pipeline,
        prompt=prompt,
        image=image,
        height=height,
        width=width,
        num_frames=frames,
        num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="np",
    )

    assert torch.equal(
        state.get("latents").cpu(), one_shot.latents.cpu()
    ), "video latents diverged between split-stage and one-shot execution"
    audio = state.get("audio_latents")
    if one_shot.audio_latents is not None:
        assert torch.equal(audio.cpu(), one_shot.audio_latents.cpu()), (
            "audio latents diverged between split-stage and one-shot "
            "execution"
        )
