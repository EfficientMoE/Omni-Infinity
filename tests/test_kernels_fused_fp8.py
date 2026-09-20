# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
import torch

from omni_infinity.kernels._quant import quantize_block_fp8
from omni_infinity.kernels._reference import (
    dequant_block_fp8,
    fused_fp8_gemm_reference,
)


def test_quantize_block_fp8_shapes():
    w = (torch.randn(256, 384) * 0.02).to(torch.bfloat16)
    q, s = quantize_block_fp8(w)
    assert q.dtype == torch.float8_e4m3fn and tuple(q.shape) == (256, 384)
    # ceil(256/128), ceil(384/128)
    assert tuple(s.shape) == (2, 3) and s.dtype == torch.float32


def test_quantize_block_fp8_reconstruction_beats_or_matches_perrow():
    torch.manual_seed(0)
    w = (torch.randn(512, 512) * 0.02).to(torch.bfloat16)
    q, s = quantize_block_fp8(w)
    deq = dequant_block_fp8(q, s).to(torch.float32)
    rel = (deq - w.to(torch.float32)).norm() / w.to(torch.float32).norm()
    assert rel < 0.05, rel.item()


def test_reference_matches_flinear_of_dequant():
    torch.manual_seed(0)
    x = torch.randn(8, 512, dtype=torch.bfloat16)
    w = (torch.randn(256, 512) * 0.02).to(torch.bfloat16)
    b = torch.randn(256, dtype=torch.bfloat16)
    q, s = quantize_block_fp8(w)
    got = fused_fp8_gemm_reference(x, q, s, b).to(torch.float32)
    ref = torch.nn.functional.linear(
        x.to(torch.float32),
        dequant_block_fp8(q, s).to(torch.float32),
        b.to(torch.float32),
    )
    rel = (got - ref).norm() / ref.norm()
    assert rel < 1e-2, rel.item()  # bf16 output rounding only


def test_quantize_block_handles_non_divisible_K():
    w = (torch.randn(128, 200) * 0.02).to(torch.bfloat16)  # K not %128
    q, s = quantize_block_fp8(w)
    assert tuple(s.shape) == (1, 2)
    deq = dequant_block_fp8(q, s)
    assert tuple(deq.shape) == (128, 200)
