# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Pure-torch reference for the fused block-scaled FP8 weight-only GEMM.

Doubles as the unit-test oracle and the no-Triton / non-CUDA fallback.
"""

from __future__ import annotations

import torch

from ._quant import WEIGHT_BLOCK


def dequant_block_fp8(
    b_fp8: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int] = WEIGHT_BLOCK,
) -> torch.Tensor:
    """[N, K] fp8 + [ceil(N/bn), ceil(K/bk)] scale -> [N, K] fp32 weight."""
    bn, bk = block
    N, K = b_fp8.shape
    s = scale.reshape((N + bn - 1) // bn, (K + bk - 1) // bk).to(torch.float32)
    s_full = s.repeat_interleave(bn, 0).repeat_interleave(bk, 1)[:N, :K]
    return b_fp8.to(torch.float32) * s_full


def fused_fp8_gemm_reference(
    a: torch.Tensor,
    b_fp8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
    scale_block: tuple[int, int] = WEIGHT_BLOCK,
) -> torch.Tensor:
    """C = a @ dequant(b_fp8).T + bias. a:[..., K] -> C:[..., N] bf16."""
    orig = a.shape
    a2d = a.reshape(-1, orig[-1]).to(torch.float32)
    N, _ = b_fp8.shape
    w = dequant_block_fp8(b_fp8, scale, scale_block)
    c = a2d @ w.t()
    if bias is not None:
        c = c + bias.to(torch.float32)
    c = c.to(torch.bfloat16).reshape(*orig[:-1], N)
    if out is not None:
        out.copy_(c)
        return out
    return c
