# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Geometry adapter for vLLM's CUTLASS SM120 blockwise FP8 GEMM."""

from __future__ import annotations

import torch

from ..._quant import WEIGHT_BLOCK
from ._loader import load_extension

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_FP32_TINY = torch.finfo(torch.float32).tiny


def _quantize_activation_1x128(
    activation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize row-major [M,K] activations with one scale per 1x128 group."""
    M, K = activation.shape
    groups = activation.float().reshape(M, K // 128, 128)
    scale = (groups.abs().amax(dim=2) / _FP8_MAX).clamp_min(_FP32_TINY)
    quantized = (
        (groups / scale.unsqueeze(2))
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
        .reshape(M, K)
    )
    # CUTLASS's SFA layout is M-major: address = k_group * M + row.
    scale_m_major = scale.t().contiguous().t()
    return quantized, scale_m_major


def fused_fp8_gemm_cutlass_sm120(
    a: torch.Tensor,
    b_fp8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
    scale_block: tuple[int, int] = WEIGHT_BLOCK,
) -> torch.Tensor:
    """C = a @ dequant(b_fp8).T using SM120 blockwise FP8 tensor cores."""
    if a.device.type != "cuda":
        raise RuntimeError("cutlass_sm120 backend requires a CUDA device")
    if torch.cuda.get_device_capability(a.device) != (12, 0):
        raise RuntimeError(
            "cutlass_sm120 backend requires compute capability 12.0"
        )
    if scale_block != WEIGHT_BLOCK:
        raise ValueError("cutlass_sm120 requires 128x128 weight scale blocks")
    if b_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("cutlass_sm120 requires float8_e4m3fn weights")
    if scale.dtype != torch.float32:
        raise TypeError("cutlass_sm120 requires float32 weight scales")

    original_shape = a.shape
    K = original_shape[-1]
    N, weight_k = b_fp8.shape
    if K != weight_k:
        raise ValueError(f"activation K={K} does not match weight K={weight_k}")
    if K % 128 or N % 128:
        raise ValueError("cutlass_sm120 requires N and K divisible by 128")
    expected_scale_shape = (N // 128, K // 128)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"expected weight scales shaped {expected_scale_shape}, "
            f"got {tuple(scale.shape)}"
        )

    a2d = a.reshape(-1, K)
    q_a, scale_a = _quantize_activation_1x128(a2d)
    # A contiguous [N,K] facade weight transposes to CUTLASS column-major [K,N].
    q_b = b_fp8.contiguous().t()
    # Facade scales [N/128,K/128] transpose directly into CUTLASS K-major SFB.
    scale_b = scale.contiguous().t()
    result = torch.empty(
        (a2d.shape[0], N), dtype=torch.bfloat16, device=a.device
    )

    extension = load_extension()
    extension.scaled_mm_blockwise_sm120_fp8(result, q_a, q_b, scale_a, scale_b)
    if bias is not None:
        result.add_(bias.to(dtype=torch.bfloat16, device=a.device))
    result = result.reshape(*original_shape[:-1], N)
    if out is not None:
        out.copy_(result)
        return out
    return result
