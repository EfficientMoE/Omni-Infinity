# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Rung A: ``torch._scaled_mm`` w8a8 backend for ``fused_fp8_gemm``.

P1+P4 plan (docs/superpowers/plans/2026-10-07-p1-p4-native-fp8-gemm.md),
rung A: a true-FP8-compute baseline on SM120's native FP8 tensor cores,
selected via ``fused_fp8_gemm(..., backend="scaled_mm")`` or the
``OMO_KERNELS_BACKEND=scaled_mm`` env var. The default (no backend) path
must stay byte-identical in behavior for existing callers.
"""

import pytest
import torch

from omni_infinity.kernels import fused_fp8_gemm, quantize_block_fp8
from omni_infinity.kernels._reference import fused_fp8_gemm_reference

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
gpu = pytest.mark.gpu

# Tolerance rationale: the scaled_mm backend quantizes BOTH operands to
# e4m3 (w8a8) with per-row scales, while the reference dequantizes a
# block-scaled (128x128) weight and runs the matmul in fp32. e4m3 has a
# 3-bit mantissa (~6% max relative step); dynamic per-token activation
# quantization adds its own ~2-3% RMS error on top of the weight
# requantization (block -> per-row collapses the K-axis scale
# granularity). The weight-only paths in tests/test_fp8_parity.py use
# rel < 0.05; the extra activation-side quantization justifies a modestly
# higher 0.08 relative-Frobenius bar. Measured values on sm120 are ~0.02
# to 0.04; 0.08 catches scale/transpose/dtype bugs (which produce
# rel > 0.5) without masking real regressions.
W8A8_REL_TOL = 0.08


def _make_inputs(M: int, N: int, K: int, dev: str, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    q, s = quantize_block_fp8(w.cpu())
    return x, q.to(dev), s.to(dev)


@gpu
@cuda
# torch._scaled_mm requires 16-aligned K (fp8 operands); shapes here are
# aligned on purpose -- the backend documents the constraint instead of
# padding (H3's real Linear shapes are all 16-aligned).
@pytest.mark.parametrize(
    "M,N,K", [(256, 256, 512), (2048, 1536, 4096), (64, 28672, 5376)]
)
def test_scaled_mm_backend_matches_reference(M, N, K):
    x, q, s = _make_inputs(M, N, K, "cuda")
    ref = fused_fp8_gemm_reference(x, q, s)
    got = fused_fp8_gemm(x, q, s, backend="scaled_mm")
    assert got.dtype == torch.bfloat16
    assert got.shape == ref.shape
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    assert rel < W8A8_REL_TOL, rel.item()


@gpu
@cuda
def test_scaled_mm_backend_applies_bias():
    x, q, s = _make_inputs(128, 256, 512, "cuda")
    bias = torch.randn(256, dtype=torch.bfloat16, device="cuda")
    ref = fused_fp8_gemm_reference(x, q, s, bias)
    got = fused_fp8_gemm(x, q, s, bias, backend="scaled_mm")
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    assert rel < W8A8_REL_TOL, rel.item()


@gpu
@cuda
def test_scaled_mm_backend_handles_3d_input_and_out():
    x3 = torch.randn(4, 32, 512, dtype=torch.bfloat16, device="cuda")
    _, q, s = _make_inputs(1, 256, 512, "cuda")
    got = fused_fp8_gemm(x3, q, s, backend="scaled_mm")
    assert got.shape == (4, 32, 256)
    out = torch.empty_like(got)
    returned = fused_fp8_gemm(x3, q, s, backend="scaled_mm", out=out)
    assert returned is out
    assert torch.equal(out, got)


@gpu
@cuda
def test_env_var_selects_scaled_mm_backend(monkeypatch):
    x, q, s = _make_inputs(128, 256, 512, "cuda")
    explicit = fused_fp8_gemm(x, q, s, backend="scaled_mm")
    monkeypatch.setenv("OMO_KERNELS_BACKEND", "scaled_mm")
    via_env = fused_fp8_gemm(x, q, s)
    assert torch.equal(explicit, via_env)


def test_unknown_backend_raises():
    x = torch.randn(8, 128, dtype=torch.bfloat16)
    w = (torch.randn(64, 128) * 0.02).to(torch.bfloat16)
    q, s = quantize_block_fp8(w)
    with pytest.raises(ValueError, match="backend"):
        fused_fp8_gemm(x, q, s, backend="nope")


def test_scaled_mm_backend_requires_cuda_device():
    x = torch.randn(8, 128, dtype=torch.bfloat16)
    w = (torch.randn(64, 128) * 0.02).to(torch.bfloat16)
    q, s = quantize_block_fp8(w)
    with pytest.raises(RuntimeError, match="CUDA"):
        fused_fp8_gemm(x, q, s, backend="scaled_mm")


def test_default_path_unchanged_on_cpu():
    # No backend kwarg + no env var == pre-existing behavior (reference
    # impl on CPU). Guards the "existing callers 100% unaffected" claim.
    x = torch.randn(8, 128, dtype=torch.bfloat16)
    w = (torch.randn(64, 128) * 0.02).to(torch.bfloat16)
    q, s = quantize_block_fp8(w)
    assert torch.equal(
        fused_fp8_gemm(x, q, s), fused_fp8_gemm_reference(x, q, s)
    )
