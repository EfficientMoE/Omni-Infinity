# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Rung A: true-FP8-compute GEMM via ``torch._scaled_mm`` (w8a8).

Unlike the Triton weight-only path (fp8 weight dequantized to bf16 inside
the GEMM, MMA on bf16 tensor cores), this backend quantizes BOTH operands
to e4m3 and runs the MMA on SM120's native FP8 tensor cores. It exists to
establish the true-FP8 speed ceiling (P1 plan, rung A) with zero build
complexity.

Scale-granularity contract: the facade hands us a 128x128 block-quantized
weight. ``torch._scaled_mm`` only supports per-tensor or per-row scales,
so the weight is dequantized and requantized per output row on the fly
(correctness over convenience; the one-off bf16-sized temp is acceptable
for a speed-ceiling baseline). The activation is dynamically quantized
per token. Constraint: K must be 16-aligned (fp8 operand requirement);
all real H3 Linear shapes satisfy this.
"""

from __future__ import annotations

import torch

from ._quant import WEIGHT_BLOCK
from ._reference import dequant_block_fp8

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_FP32_TINY = torch.finfo(torch.float32).tiny


def _quantize_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x32 = x.to(torch.float32)
    scale = (x32.abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(
        _FP32_TINY
    )
    q = (x32 / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def fused_fp8_gemm_scaled_mm(
    a: torch.Tensor,
    b_fp8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
    scale_block: tuple[int, int] = WEIGHT_BLOCK,
) -> torch.Tensor:
    """C = a @ dequant(b_fp8).T + bias on native FP8 tensor cores.

    a: [..., K] any float dtype; b_fp8: [N, K] float8_e4m3fn;
    scale: [ceil(N/128), ceil(K/128)] fp32 (block-wise). Returns
    [..., N] bf16. CUDA-only; K must be a multiple of 16.
    """
    if a.device.type != "cuda":
        raise RuntimeError("scaled_mm backend requires a CUDA device")
    orig = a.shape
    a2d = a.reshape(-1, orig[-1])
    N = b_fp8.shape[0]

    w = dequant_block_fp8(b_fp8, scale, scale_block)
    q_w, s_w = _quantize_per_row(w)
    q_a, s_a = _quantize_per_row(a2d)

    c = torch._scaled_mm(
        q_a,
        q_w.t(),
        scale_a=s_a,
        scale_b=s_w.t(),
        out_dtype=torch.bfloat16,
    )
    if bias is not None:
        c = c + bias.to(torch.bfloat16)
    c = c.reshape(*orig[:-1], N)
    if out is not None:
        out.copy_(c)
        return out
    return c
