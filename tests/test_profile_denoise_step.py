# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import torch

from benchmarks import profile_denoise_step


def test_compile_resident_profile_enables_compile_without_block_streaming():
    assert profile_denoise_step.PROFILE_OPTIMIZATIONS["compile-resident"] == (
        "adaln-host-cache",
        "compile-blocks",
    )


def test_latent_parity_reports_bitwise_and_allclose_tiers():
    reference = torch.tensor([1.0, 2.0])

    exact = profile_denoise_step.latent_parity(reference.clone(), reference)
    assert exact["bitwise"] is True
    assert exact["tier"] == "bitwise"
    assert exact["rms_rel"] == 0.0

    candidate = reference + torch.tensor([1e-3, -1e-3])
    close = profile_denoise_step.latent_parity(candidate, reference)
    assert close["bitwise"] is False
    assert close["allclose"]["rtol=atol=1e-5"] is False
    assert close["allclose"]["rtol=atol=1e-3"] is True
    assert close["allclose"]["rtol=atol=2e-2"] is True
    assert close["tier"] == "allclose-1e-3"

    far = profile_denoise_step.latent_parity(reference * 2, reference)
    assert (
        profile_denoise_step.parity_gate_passes(exact, require_bitwise=True)
        is True
    )
    assert (
        profile_denoise_step.parity_gate_passes(close, require_bitwise=True)
        is False
    )
    assert (
        profile_denoise_step.parity_gate_passes(close, require_bitwise=False)
        is True
    )
    assert (
        profile_denoise_step.parity_gate_passes(far, require_bitwise=False)
        is False
    )


def test_golden_provenance_reports_stack_gpu_and_input_mismatches():
    golden = {
        "torch_version": "2.12.0",
        "diffusers_version": "0.40.0",
        "gpu_name": "server-gpu",
        "first_frame_sha256": "abc",
        "prompt": "prompt",
        "seed": 0,
        "steps": 8,
        "resolution": "256p",
        "frames": 120,
    }
    exact = profile_denoise_step.golden_provenance(
        golden,
        torch_version="2.12.0",
        diffusers_version="0.40.0",
        gpu_name="server-gpu",
        first_frame_sha256="abc",
        prompt="prompt",
        seed=0,
        steps=8,
        resolution="256p",
        frames=120,
    )
    assert exact == {"comparable": True, "mismatches": []}

    mismatch = profile_denoise_step.golden_provenance(
        golden,
        torch_version="2.13.0",
        diffusers_version="0.40.0",
        gpu_name="workstation-gpu",
        first_frame_sha256="def",
        prompt="other prompt",
        seed=1,
        steps=4,
        resolution="512p",
        frames=121,
    )
    assert mismatch["comparable"] is False
    assert mismatch["mismatches"] == [
        "torch_version: recorded='2.12.0', runtime='2.13.0'",
        "gpu_name: recorded='server-gpu', runtime='workstation-gpu'",
        "first_frame_sha256: recorded='abc', runtime='def'",
        "prompt: recorded='prompt', runtime='other prompt'",
        "seed: recorded=0, runtime=1",
        "steps: recorded=8, runtime=4",
        "resolution: recorded='256p', runtime='512p'",
        "frames: recorded=120, runtime=121",
    ]
