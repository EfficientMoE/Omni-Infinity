# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""FP8 AdaLN path: per-row quantization and the bf16 non-regression guard.

Task 2 increment 4 (issue #2). These CI tests gate the host-side quantizer
and confirm the bf16 path is untouched by the FP8 wiring. The end-to-end FP8
tolerance (latents within allclose(rtol=2e-2) of the goldens) is asserted by
the real-store GPU run and by ``fl2va_smoke.py --transformer-fp8 --goldens``.
"""

import torch
import torch.nn as nn

from omni_infinity.adaln import AdaLNEntry
from omni_infinity.fp8 import (
    ScaledFp8Linear,
    apply_scaled_fp8_casting,
    quantize_per_row_fp8,
)


def test_quantize_adaln_fp8_shapes_and_dtype():
    weight = torch.randn(256, 64, dtype=torch.bfloat16)
    quantized, scale = quantize_per_row_fp8(weight)
    assert quantized.dtype == torch.float8_e4m3fn
    assert tuple(quantized.shape) == (256, 64)
    assert tuple(scale.shape) == (256, 1)
    assert scale.dtype == torch.float32


def test_quantize_adaln_fp8_reconstruction_error_bounded():
    torch.manual_seed(0)
    weight = (torch.randn(512, 256) * 0.02).to(torch.bfloat16)
    quantized, scale = quantize_per_row_fp8(weight)
    dequantized = (quantized.to(torch.float32) * scale).to(torch.float32)
    reference = weight.to(torch.float32)
    relative_frobenius = (dequantized - reference).norm() / reference.norm()
    assert relative_frobenius < 0.05, relative_frobenius.item()


def test_fp8_entry_materialize_dequantizes_per_row():
    torch.manual_seed(1)
    weight = (torch.randn(128, 64) * 0.03).to(torch.bfloat16)
    bias = torch.randn(128, dtype=torch.bfloat16)
    quantized, scale = quantize_per_row_fp8(weight)
    entry = AdaLNEntry(quantized, bias, scale=scale)

    materialized_weight, materialized_bias = entry.materialize(
        torch.device("cpu")
    )

    assert materialized_weight.dtype == torch.bfloat16
    expected = (quantized.to(torch.float32) * scale).to(torch.bfloat16)
    assert torch.equal(materialized_weight, expected)
    assert torch.equal(materialized_bias, bias)


def test_bf16_entry_materialize_stays_bitwise():
    weight = torch.randn(96, 32, dtype=torch.bfloat16)
    bias = torch.randn(96, dtype=torch.bfloat16)
    entry = AdaLNEntry(weight, bias)

    materialized_weight, materialized_bias = entry.materialize(
        torch.device("cpu")
    )

    assert torch.equal(materialized_weight, weight)
    assert torch.equal(materialized_bias, bias)


def test_scaled_fp8_linear_uses_block_scale_and_approximates():
    torch.manual_seed(0)
    linear = nn.Linear(512, 256).to(torch.bfloat16).eval()
    with torch.no_grad():
        linear.weight.mul_(0.02)
    fp8_linear = ScaledFp8Linear(linear, torch.bfloat16).eval()
    assert fp8_linear.weight_fp8.dtype == torch.float8_e4m3fn
    # block-wise scale grid, not per-row [N,1]:
    # (ceil(256/128), ceil(512/128)) == (2, 4)
    assert tuple(fp8_linear.weight_scale.shape) == (2, 4)
    x = torch.randn(8, 512, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = linear(x).to(torch.float32)
        approx = fp8_linear(x).to(torch.float32)
    rel = (approx - ref).norm() / ref.norm()
    assert rel < 0.05, rel.item()


def test_scaled_fp8_linear_per_row_mode_preserves_legacy():
    torch.manual_seed(0)
    linear = nn.Linear(512, 256).to(torch.bfloat16).eval()
    with torch.no_grad():
        linear.weight.mul_(0.02)
    fp8_linear = ScaledFp8Linear(linear, torch.bfloat16, mode="per_row").eval()
    # legacy per-row scale is [N, 1], NOT the block-wise
    # [ceil(N/128), ceil(K/128)]
    assert tuple(fp8_linear.weight_scale.shape) == (256, 1)
    x = torch.randn(8, 512, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = linear(x).to(torch.float32)
        approx = fp8_linear(x).to(torch.float32)
    rel = (approx - ref).norm() / ref.norm()
    assert rel < 0.05, rel.item()


class _TinyFp8Model(nn.Module):
    _keep_in_fp32_modules = ["proj_in"]
    _skip_layerwise_casting_patterns = ["norm"]

    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(4, 8)
        self.norm = nn.Linear(4, 8)
        self.blocks = nn.ModuleList([nn.Linear(4, 8), nn.Linear(8, 4)])


def test_apply_scaled_fp8_casting_respects_skip_set():
    model = _TinyFp8Model()
    replaced = apply_scaled_fp8_casting(model, torch.bfloat16)

    assert replaced == 2
    assert isinstance(model.proj_in, nn.Linear)
    assert not isinstance(model.proj_in, ScaledFp8Linear)
    assert isinstance(model.norm, nn.Linear)
    assert not isinstance(model.norm, ScaledFp8Linear)
    assert all(isinstance(block, ScaledFp8Linear) for block in model.blocks)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _TinyBlockStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(4)])


def test_apply_scaled_fp8_casting_keeps_skip_blocks_bf16():
    model = _TinyBlockStack()
    replaced = apply_scaled_fp8_casting(
        model, torch.bfloat16, skip_blocks=frozenset({2, 3})
    )

    assert replaced == 2
    assert isinstance(model.transformer_blocks[0].proj, ScaledFp8Linear)
    assert isinstance(model.transformer_blocks[1].proj, ScaledFp8Linear)
    assert not isinstance(model.transformer_blocks[2].proj, ScaledFp8Linear)
    assert not isinstance(model.transformer_blocks[3].proj, ScaledFp8Linear)
