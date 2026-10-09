# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Rung B: CUTLASS SM120 blockwise FP8 backend."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch

from omni_infinity.kernels import fused_fp8_gemm, quantize_block_fp8
from omni_infinity.kernels._reference import fused_fp8_gemm_reference


def _jit_prerequisites_available() -> bool:
    cutlass_dir = os.environ.get("OMO_CUTLASS_DIR")
    return bool(
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (12, 0)
        and shutil.which("nvcc")
        and cutlass_dir
        and (Path(cutlass_dir) / "include" / "cutlass" / "cutlass.h").is_file()
    )


gpu = pytest.mark.gpu
cutlass_sm120 = pytest.mark.skipif(
    not _jit_prerequisites_available(),
    reason="needs SM120, nvcc, and OMO_CUTLASS_DIR with CUTLASS headers",
)

# This path preserves the weight's original 128x128 scales and quantizes the
# activation per 1x128 group. It therefore removes scaled_mm's lossy
# block-to-per-row weight requantization (measured there at rel ~= 0.02-0.04).
# This backend measures ~=0.026 on all three shapes below, versus ~=0.037 for
# the per-row path. A 0.035 relative-Frobenius limit leaves modest hardware
# headroom while exposing scale-layout, transpose, or major-order bugs (>0.5).
BLOCKWISE_W8A8_REL_TOL = 0.035


def _make_inputs(M: int, N: int, K: int, device: str, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = (torch.randn(N, K, device=device) * 0.02).to(torch.bfloat16)
    q, scale = quantize_block_fp8(w.cpu())
    return x, q.to(device), scale.to(device)


@gpu
@cutlass_sm120
@pytest.mark.parametrize(
    "M,N,K", [(256, 256, 512), (2048, 1536, 4096), (64, 28672, 5376)]
)
def test_cutlass_sm120_backend_matches_reference(M, N, K):
    x, q, scale = _make_inputs(M, N, K, "cuda")
    reference = fused_fp8_gemm_reference(x, q, scale)
    actual = fused_fp8_gemm(x, q, scale, backend="cutlass_sm120")

    assert actual.dtype == torch.bfloat16
    assert actual.shape == reference.shape
    rel = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert rel < BLOCKWISE_W8A8_REL_TOL, rel.item()


@gpu
@cutlass_sm120
def test_cutlass_sm120_backend_handles_m1_scale_layout():
    # M=1 regression: contiguous() no-ops on singleton dims, so a naive
    # .t().contiguous().t() hands the extension a non-M-major SFA stride
    # and its layout guard rejects the call.
    x, q, scale = _make_inputs(1, 256, 512, "cuda")
    reference = fused_fp8_gemm_reference(x, q, scale)
    actual = fused_fp8_gemm(x, q, scale, backend="cutlass_sm120")
    rel = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert rel < BLOCKWISE_W8A8_REL_TOL, rel.item()


@gpu
@cutlass_sm120
def test_cutlass_sm120_backend_supports_bias_3d_and_out():
    x, q, scale = _make_inputs(128, 256, 512, "cuda")
    x = x.reshape(4, 32, 512)
    bias = torch.randn(256, dtype=torch.bfloat16, device="cuda")
    reference = fused_fp8_gemm_reference(x, q, scale, bias)
    out = torch.empty_like(reference)

    returned = fused_fp8_gemm(
        x, q, scale, bias, out=out, backend="cutlass_sm120"
    )

    assert returned is out
    rel = (out.float() - reference.float()).norm() / reference.float().norm()
    assert rel < BLOCKWISE_W8A8_REL_TOL, rel.item()


def test_cutlass_sm120_backend_requires_cuda_device():
    x = torch.randn(8, 128, dtype=torch.bfloat16)
    weight = (torch.randn(128, 128) * 0.02).to(torch.bfloat16)
    q, scale = quantize_block_fp8(weight)

    with pytest.raises(RuntimeError, match="CUDA"):
        fused_fp8_gemm(x, q, scale, backend="cutlass_sm120")
