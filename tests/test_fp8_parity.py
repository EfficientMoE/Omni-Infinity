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
from omni_infinity.kernels import dequant_block_fp8, quantize_block_fp8


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


def test_fp8_entry_materialize_fp8_returns_weight_scale_bias():
    torch.manual_seed(1)
    # weight is 6*hidden x hidden
    weight = (torch.randn(768, 256) * 0.03).to(torch.bfloat16)
    bias = torch.randn(768, dtype=torch.bfloat16)
    q, s = quantize_block_fp8(weight)
    entry = AdaLNEntry(q, bias, scale=s, block_scaled=True)
    w_fp8, w_scale, out_bias = entry.materialize_fp8(torch.device("cpu"))
    assert w_fp8.dtype == torch.float8_e4m3fn
    assert tuple(w_scale.shape) == (6, 2)  # ceil(768/128), ceil(256/128)
    assert torch.equal(out_bias, bias)


def test_hostresident_adaln_fp8_matches_reference_projection():
    torch.manual_seed(2)
    hidden = 128
    weight = (torch.randn(6 * hidden, hidden) * 0.03).to(torch.bfloat16)
    bias = torch.randn(6 * hidden, dtype=torch.bfloat16)
    q, s = quantize_block_fp8(weight)
    from omni_infinity.adaln import AdaLNEntry, HostResidentAdaLN

    entry = AdaLNEntry(q, bias, scale=s, block_scaled=True)
    mod = HostResidentAdaLN(entry, hidden).eval()
    temb = torch.randn(4, hidden, dtype=torch.bfloat16)
    outs = mod(temb)
    # reference: silu -> F.linear(dequant) -> view -> chunk(6)
    import torch.nn.functional as F

    w = dequant_block_fp8(q, s).to(torch.float32)
    ref = F.linear(F.silu(temb).to(torch.float32), w, bias.to(torch.float32))
    ref = ref.view(-1, 6 * hidden).chunk(6, dim=-1)
    for got, exp in zip(outs, ref):
        rel = (got.float() - exp).norm() / exp.norm()
        assert rel < 2e-2, rel.item()
    assert len(outs) == 6


def test_bf16_entry_materialize_stays_bitwise():
    weight = torch.randn(96, 32, dtype=torch.bfloat16)
    bias = torch.randn(96, dtype=torch.bfloat16)
    entry = AdaLNEntry(weight, bias)

    materialized_weight, materialized_bias = entry.materialize(
        torch.device("cpu")
    )

    assert torch.equal(materialized_weight, weight)
    assert torch.equal(materialized_bias, bias)


def test_scaled_fp8_linear_stores_float8_and_approximates():
    torch.manual_seed(0)
    linear = nn.Linear(256, 128).to(torch.bfloat16).eval()
    with torch.no_grad():
        linear.weight.mul_(0.02)
    fp8_linear = ScaledFp8Linear(linear, torch.bfloat16).eval()

    assert fp8_linear.weight_fp8.dtype == torch.float8_e4m3fn
    assert fp8_linear.weight_fp8.element_size() == 1

    x = torch.randn(8, 256, dtype=torch.bfloat16)
    with torch.no_grad():
        reference = linear(x).to(torch.float32)
        approx = fp8_linear(x).to(torch.float32)
    rel = (approx - reference).norm() / reference.norm()
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
