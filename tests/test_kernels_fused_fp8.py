# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
import pytest
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


cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)


@cuda
@pytest.mark.parametrize(
    "M,N,K", [(256, 256, 512), (2048, 1536, 4096), (17, 384, 200)]
)
def test_triton_matches_reference(M, N, K):
    from omni_infinity.kernels._fused_fp8_gemm import fused_fp8_gemm_triton

    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    b = torch.randn(N, dtype=torch.bfloat16, device=dev)
    q, s = quantize_block_fp8(w.cpu())
    q, s = q.to(dev), s.to(dev)
    got = fused_fp8_gemm_triton(x, q, s, b)
    ref = fused_fp8_gemm_reference(x, q, s, b)
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 2e-2, rel.item()
    assert got.shape == (M, N) and got.dtype == torch.bfloat16


@cuda
def test_triton_3d_input_and_out():
    from omni_infinity.kernels._fused_fp8_gemm import fused_fp8_gemm_triton

    dev = "cuda"
    x = torch.randn(2, 128, 512, dtype=torch.bfloat16, device=dev)
    w = (torch.randn(256, 512, device=dev) * 0.02).to(torch.bfloat16)
    q, s = quantize_block_fp8(w.cpu())
    q, s = q.to(dev), s.to(dev)
    y = fused_fp8_gemm_triton(x, q, s)
    assert tuple(y.shape) == (2, 128, 256)


def test_facade_exports():
    import omni_infinity.kernels as K

    assert hasattr(K, "fused_fp8_gemm") and hasattr(K, "quantize_block_fp8")
    assert K.WEIGHT_BLOCK == (128, 128)


def test_facade_reference_fallback_on_cpu():
    import omni_infinity.kernels as K

    x = torch.randn(4, 256, dtype=torch.bfloat16)
    w = (torch.randn(128, 256) * 0.02).to(torch.bfloat16)
    q, s = K.quantize_block_fp8(w)
    y = K.fused_fp8_gemm(x, q, s)  # cpu -> reference impl
    assert tuple(y.shape) == (4, 128) and y.dtype == torch.bfloat16


@cuda
def test_facade_env_forces_reference(monkeypatch):
    monkeypatch.setenv("OMO_KERNELS_REFERENCE", "1")
    import importlib

    import omni_infinity.kernels as K

    importlib.reload(K)
    x = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")
    w = (torch.randn(128, 256, device="cuda") * 0.02).to(torch.bfloat16)
    q, s = K.quantize_block_fp8(w.cpu())
    q, s = q.cuda(), s.cuda()
    y = K.fused_fp8_gemm(x, q, s)
    assert tuple(y.shape) == (8, 128)
    importlib.reload(K)  # restore
