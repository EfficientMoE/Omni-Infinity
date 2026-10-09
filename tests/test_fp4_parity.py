# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""MXFP4 path: ScaledFp4Linear, the casting walk, and the opt-in refusal.

P5 phase 1 (issue #42). These CI tests gate the host-side packing and the
"refuses without explicit scale mode" contract. The end-to-end FP4 accuracy
(latents vs goldens, expected to FAIL the rtol=2e-2 gate with recorded
rms_rel) is measured by ``fl2va_smoke.py --transformer-fp4 --fp4-scale
mxfp4 --goldens``.
"""

import pytest
import torch
import torch.nn as nn

from omni_infinity.fp4 import ScaledFp4Linear, apply_scaled_fp4_casting
from omni_infinity.kernels import dequant_mxfp4


def test_scaled_fp4_linear_rejects_an_unknown_mode():
    linear = nn.Linear(64, 4, bias=False)
    with pytest.raises(AssertionError):
        ScaledFp4Linear(linear, torch.bfloat16, mode="block")


def test_scaled_fp4_linear_buffers_and_approximation():
    torch.manual_seed(0)
    linear = nn.Linear(512, 256).to(torch.bfloat16).eval()
    with torch.no_grad():
        linear.weight.mul_(0.02)
    fp4_linear = ScaledFp4Linear(linear, torch.bfloat16).eval()
    assert fp4_linear.weight_packed.dtype == torch.uint8
    assert tuple(fp4_linear.weight_packed.shape) == (256, 256)
    assert fp4_linear.weight_scales.dtype == torch.uint8
    assert tuple(fp4_linear.weight_scales.shape) == (256, 16)
    x = torch.randn(8, 512, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = linear(x).to(torch.float32)
        approx = fp4_linear(x).to(torch.float32)
    rel = (approx - ref).norm() / ref.norm()
    assert rel < 0.25, rel.item()


def test_scaled_fp4_linear_matches_flinear_of_dequant():
    torch.manual_seed(1)
    linear = nn.Linear(256, 128).to(torch.bfloat16).eval()
    with torch.no_grad():
        linear.weight.mul_(0.02)
    fp4_linear = ScaledFp4Linear(linear, torch.bfloat16).eval()
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    with torch.no_grad():
        got = fp4_linear(x).to(torch.float32)
    w = dequant_mxfp4(fp4_linear.weight_packed, fp4_linear.weight_scales)
    ref = torch.nn.functional.linear(
        x.to(torch.float32), w, linear.bias.to(torch.float32)
    )
    rel = (got - ref).norm() / ref.norm()
    assert rel < 1e-2, rel.item()


class _TinyFp4Model(nn.Module):
    _keep_in_fp32_modules = ["proj_in"]
    _skip_layerwise_casting_patterns = ["norm"]

    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(32, 8)
        self.norm = nn.Linear(32, 8)
        self.blocks = nn.ModuleList([nn.Linear(32, 8), nn.Linear(64, 4)])


def test_apply_scaled_fp4_casting_respects_skip_set():
    model = _TinyFp4Model()
    replaced = apply_scaled_fp4_casting(model, torch.bfloat16)

    assert replaced == 2
    assert isinstance(model.proj_in, nn.Linear)
    assert not isinstance(model.proj_in, ScaledFp4Linear)
    assert isinstance(model.norm, nn.Linear)
    assert not isinstance(model.norm, ScaledFp4Linear)
    assert all(isinstance(block, ScaledFp4Linear) for block in model.blocks)


def test_apply_scaled_fp4_casting_skips_unpackable_in_features():
    model = nn.Module()
    model.narrow = nn.Linear(40, 8)
    model.wide = nn.Linear(64, 8)
    replaced = apply_scaled_fp4_casting(model, torch.bfloat16)

    assert replaced == 1
    assert isinstance(model.narrow, nn.Linear)
    assert not isinstance(model.narrow, ScaledFp4Linear)
    assert isinstance(model.wide, ScaledFp4Linear)


class _Fp4Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 4)


class _TinyFp4BlockStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Fp4Block() for _ in range(4)])


def test_apply_scaled_fp4_casting_keeps_skip_blocks_bf16():
    model = _TinyFp4BlockStack()
    replaced = apply_scaled_fp4_casting(
        model, torch.bfloat16, skip_blocks=frozenset({2, 3})
    )

    assert replaced == 2
    assert isinstance(model.transformer_blocks[0].proj, ScaledFp4Linear)
    assert isinstance(model.transformer_blocks[1].proj, ScaledFp4Linear)
    assert not isinstance(model.transformer_blocks[2].proj, ScaledFp4Linear)
    assert not isinstance(model.transformer_blocks[3].proj, ScaledFp4Linear)


def test_runner_refuses_fp4_without_explicit_scale_mode():
    from omni_infinity.runner import ReferenceRunner

    with pytest.raises(ValueError, match="fp4-scale"):
        ReferenceRunner.from_pretrained(
            "unused-checkpoint", transformer_fp4=True
        )


def test_runner_refuses_an_unknown_fp4_scale_mode():
    from omni_infinity.runner import ReferenceRunner

    with pytest.raises(ValueError, match="unsupported fp4_scale"):
        ReferenceRunner.from_pretrained(
            "unused-checkpoint", transformer_fp4=True, fp4_scale="block"
        )


def test_runner_refuses_fp4_combined_with_fp8():
    from omni_infinity.runner import ReferenceRunner

    with pytest.raises(ValueError, match="mutually"):
        ReferenceRunner.from_pretrained(
            "unused-checkpoint",
            transformer_fp4=True,
            fp4_scale="mxfp4",
            transformer_fp8=True,
        )
