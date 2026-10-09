# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from omni_infinity.kernels._quant import quantize_mxfp4
from omni_infinity.kernels._reference import (
    dequant_mxfp4,
    fused_mxfp4_gemm_reference,
)

_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def test_quantize_mxfp4_shapes():
    w = (torch.randn(256, 384) * 0.02).to(torch.bfloat16)
    packed, scales = quantize_mxfp4(w)
    assert packed.dtype == torch.uint8
    assert tuple(packed.shape) == (256, 192)
    assert scales.dtype == torch.uint8
    assert tuple(scales.shape) == (256, 12)


def test_quantize_mxfp4_roundtrip_error_bounded():
    torch.manual_seed(0)
    w = (torch.randn(512, 512) * 0.02).to(torch.bfloat16).to(torch.float32)
    packed, scales = quantize_mxfp4(w)
    deq = dequant_mxfp4(packed, scales)
    rel = (deq - w).norm() / w.norm()
    assert rel < 0.25, rel.item()
    amax = w.reshape(512, 16, 32).abs().amax(dim=2, keepdim=True)
    err = (deq - w).abs().reshape(512, 16, 32)
    assert (err <= amax / 3.0 + 1e-6).all()


def test_quantize_mxfp4_exact_for_representable_values():
    torch.manual_seed(1)
    lut = torch.tensor(_LUT + [-v for v in _LUT])
    idx = torch.randint(0, 16, (64, 128))
    w = lut[idx]
    w[:, ::32] = 6.0  # pins every block's amax so the exponent is 0
    packed, scales = quantize_mxfp4(w)
    assert (scales == 127).all()
    deq = dequant_mxfp4(packed, scales)
    assert torch.equal(deq, w + 0.0)  # +0.0 folds -0.0 into 0.0


def test_quantize_mxfp4_rejects_bad_k():
    with pytest.raises(AssertionError):
        quantize_mxfp4(torch.randn(128, 200))


def test_quantize_mxfp4_zero_weight():
    packed, scales = quantize_mxfp4(torch.zeros(64, 64))
    assert (packed == 0).all()
    assert (scales == 127).all()
    assert (dequant_mxfp4(packed, scales) == 0).all()


def test_reference_matches_flinear_of_dequant():
    torch.manual_seed(0)
    x = torch.randn(8, 512, dtype=torch.bfloat16)
    w = (torch.randn(256, 512) * 0.02).to(torch.bfloat16)
    b = torch.randn(256, dtype=torch.bfloat16)
    packed, scales = quantize_mxfp4(w)
    got = fused_mxfp4_gemm_reference(x, packed, scales, b).to(torch.float32)
    ref = torch.nn.functional.linear(
        x.to(torch.float32),
        dequant_mxfp4(packed, scales),
        b.to(torch.float32),
    )
    rel = (got - ref).norm() / ref.norm()
    assert rel < 1e-2, rel.item()


def test_reference_writes_the_provided_out_buffer():
    torch.manual_seed(1)
    x = torch.randn(8, 256, dtype=torch.bfloat16)
    w = (torch.randn(128, 256) * 0.02).to(torch.bfloat16)
    packed, scales = quantize_mxfp4(w)
    expected = fused_mxfp4_gemm_reference(x, packed, scales)
    out = torch.empty(8, 128, dtype=torch.bfloat16)
    returned = fused_mxfp4_gemm_reference(x, packed, scales, out=out)
    assert returned is out
    assert torch.equal(out, expected)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
gpu = pytest.mark.gpu


@gpu
@cuda
@pytest.mark.parametrize(
    "M,N,K", [(256, 256, 512), (2048, 1536, 4096), (17, 384, 224)]
)
def test_triton_matches_reference(M, N, K):
    from omni_infinity.kernels._fused_mxfp4_gemm import fused_mxfp4_gemm_triton

    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    b = torch.randn(N, dtype=torch.bfloat16, device=dev)
    packed, scales = quantize_mxfp4(w.cpu())
    packed, scales = packed.to(dev), scales.to(dev)
    got = fused_mxfp4_gemm_triton(x, packed, scales, b)
    ref = fused_mxfp4_gemm_reference(x, packed, scales, b)
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 2e-2, rel.item()
    assert got.shape == (M, N) and got.dtype == torch.bfloat16


@gpu
@cuda
def test_triton_no_bias_matches_reference():
    from omni_infinity.kernels._fused_mxfp4_gemm import fused_mxfp4_gemm_triton

    torch.manual_seed(2)
    dev = "cuda"
    x = torch.randn(64, 256, dtype=torch.bfloat16, device=dev)
    w = (torch.randn(128, 256, device=dev) * 0.02).to(torch.bfloat16)
    packed, scales = quantize_mxfp4(w.cpu())
    packed, scales = packed.to(dev), scales.to(dev)
    got = fused_mxfp4_gemm_triton(x, packed, scales)
    ref = fused_mxfp4_gemm_reference(x, packed, scales)
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 2e-2, rel.item()


@gpu
@cuda
def test_triton_3d_input_and_out():
    from omni_infinity.kernels._fused_mxfp4_gemm import fused_mxfp4_gemm_triton

    dev = "cuda"
    x = torch.randn(2, 128, 512, dtype=torch.bfloat16, device=dev)
    w = (torch.randn(256, 512, device=dev) * 0.02).to(torch.bfloat16)
    packed, scales = quantize_mxfp4(w.cpu())
    packed, scales = packed.to(dev), scales.to(dev)
    y = fused_mxfp4_gemm_triton(x, packed, scales)
    assert tuple(y.shape) == (2, 128, 256)
    out = torch.empty_like(y)
    returned = fused_mxfp4_gemm_triton(x, packed, scales, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert torch.equal(out, y)


def test_facade_exports():
    import omni_infinity.kernels as K

    assert hasattr(K, "fused_mxfp4_gemm") and hasattr(K, "quantize_mxfp4")
    assert K.MXFP4_BLOCK == 32


def test_facade_reference_fallback_on_cpu():
    import omni_infinity.kernels as K

    x = torch.randn(4, 256, dtype=torch.bfloat16)
    w = (torch.randn(128, 256) * 0.02).to(torch.bfloat16)
    packed, scales = K.quantize_mxfp4(w)
    y = K.fused_mxfp4_gemm(x, packed, scales)
    assert tuple(y.shape) == (4, 128) and y.dtype == torch.bfloat16


@gpu
@cuda
def test_facade_env_forces_reference(monkeypatch):
    monkeypatch.setenv("OMO_KERNELS_REFERENCE", "1")
    import importlib

    import omni_infinity.kernels as K

    importlib.reload(K)
    x = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")
    w = (torch.randn(128, 256, device="cuda") * 0.02).to(torch.bfloat16)
    packed, scales = K.quantize_mxfp4(w.cpu())
    packed, scales = packed.cuda(), scales.cuda()
    y = K.fused_mxfp4_gemm(x, packed, scales)
    assert tuple(y.shape) == (8, 128)
    importlib.reload(K)  # restore
