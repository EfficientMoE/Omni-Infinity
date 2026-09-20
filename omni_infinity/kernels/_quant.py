# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Block-wise (128x128) FP8 weight quantization for the fused
weight-only GEMM."""

from __future__ import annotations

import torch
import torch.nn.functional as F

WEIGHT_BLOCK = (128, 128)


def quantize_block_fp8(
    weight: torch.Tensor, block: tuple[int, int] = WEIGHT_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [N, K] weight to float8_e4m3fn with one fp32 scale per
    (block[0] x block[1]) tile. Quantize from the ORIGINAL hi-precision weight.

    Returns (q_fp8 [N, K], scale [ceil(N/bn), ceil(K/bk)] fp32).
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    bn, bk = block
    N, K = weight.shape
    w = weight.detach().to(torch.float32)
    n_blk, k_blk = (N + bn - 1) // bn, (K + bk - 1) // bk
    wp = F.pad(w, (0, k_blk * bk - K, 0, n_blk * bn - N))
    tiles = wp.reshape(n_blk, bn, k_blk, bk)
    amax = tiles.abs().amax(dim=(1, 3))  # [n_blk, k_blk]
    scale = (amax / fp8_max).clamp_min(torch.finfo(torch.float32).tiny)
    s_full = scale.repeat_interleave(bn, 0).repeat_interleave(bk, 1)[:N, :K]
    q = (w / s_full).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q, scale.contiguous()
