# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Omni-Infinity compute kernels — op-centric facade.

Public API is deliberately framework-agnostic and stable: call sites import
ONLY from here. Extraction to a shared ``moe-kernels`` package later is a
mechanical import swap (see .sisyphus/plans/...-fused-fp8-weight-only-kernel.md,
"Deferred"). Do NOT expose tile params or arch tuples here.
"""
from __future__ import annotations

import os

import torch

from ._quant import WEIGHT_BLOCK, quantize_block_fp8
from ._reference import dequant_block_fp8, fused_fp8_gemm_reference

__all__ = [
    "fused_fp8_gemm",
    "quantize_block_fp8",
    "dequant_block_fp8",
    "WEIGHT_BLOCK",
]

_FORCE_REFERENCE = os.environ.get("OMO_KERNELS_REFERENCE") == "1"


def _select_impl(device: torch.device):
    # v0.1: one Triton impl + pure-torch reference. The (op, qtype, arch)
    # registry lands HERE on extraction; call sites never change.
    if _FORCE_REFERENCE or device.type != "cuda":
        return fused_fp8_gemm_reference
    try:
        from ._fused_fp8_gemm import fused_fp8_gemm_triton
    except Exception:
        return fused_fp8_gemm_reference
    return fused_fp8_gemm_triton


def fused_fp8_gemm(
    a: torch.Tensor,
    b_fp8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """C = a @ dequant(b_fp8).T + bias.

    a: [..., K] (cast to bf16); b_fp8: [N, K] float8_e4m3fn;
    scale: [ceil(N/128), ceil(K/128)] fp32 (block-wise). Returns [..., N] bf16.
    """
    return _select_impl(a.device)(a, b_fp8, scale, bias, out=out)
