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
