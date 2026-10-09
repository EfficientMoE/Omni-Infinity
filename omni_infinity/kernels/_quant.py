# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Weight quantizers for the fused weight-only GEMMs: block-wise
(128x128) FP8 and MXFP4 (E2M1, per-32 E8M0 scales)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

WEIGHT_BLOCK = (128, 128)
MXFP4_BLOCK = 32

# E2M1 magnitudes (sign lives in nibble bit 3) and the midpoints between
# adjacent magnitudes used for nearest-value rounding.
_MXFP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MXFP4_MIDPOINTS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
)


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


def quantize_mxfp4(
    weight: torch.Tensor, block: int = MXFP4_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [N, K] weight to MXFP4 (E2M1 nibbles, E8M0 scales).

    Layout matches the fused kernel: two FP4 values per uint8 along K
    (low nibble = even K index, high nibble = odd), sign in nibble
    bit 3. Each uint8 scale encodes a power-of-two exponent
    (``byte - 127``) shared by ``block`` consecutive K elements.
    Quantize from the ORIGINAL hi-precision weight.

    Returns (packed [N, K//2] u8, scales [N, K//block] u8).
    """
    N, K = weight.shape
    assert (
        K % block == 0 and block % 2 == 0
    ), f"K={K} must be divisible by block={block}"
    w = weight.detach().to(torch.float32)
    tiny = torch.finfo(torch.float32).tiny
    blocks = w.reshape(N, K // block, block)
    amax = blocks.abs().amax(dim=2)  # [N, K//block]
    e = torch.ceil(torch.log2(amax.clamp_min(tiny) / 6.0))
    e = torch.where(amax > 0, e, torch.zeros_like(e))
    e = e.clamp(-126.0, 127.0)
    # Guard against log2 rounding leaving amax just above 6 * 2^e.
    bump = amax > 6.0 * torch.exp2(e)
    e = torch.where(bump, (e + 1).clamp(max=127.0), e)
    scale = torch.exp2(e)
    normalized = blocks / scale.unsqueeze(2)
    mids = _MXFP4_MIDPOINTS.to(w.device)
    mag_idx = torch.bucketize(normalized.abs(), mids, right=True)
    nibble = torch.where(normalized < 0, mag_idx + 8, mag_idx)
    nib = nibble.reshape(N, K // 2, 2).to(torch.uint8)
    packed = nib[:, :, 0] | (nib[:, :, 1] << 4)
    scales = (e + 127.0).to(torch.uint8)
    return packed.contiguous(), scales.contiguous()
