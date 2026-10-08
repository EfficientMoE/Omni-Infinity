# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Fused MXFP4 (E2M1) weight-only GEMM: C = A @ dequant(B).T + bias.

Ported/adapted from EfficientMoE/BatchGen (Apache-2.0) via MoE-Infinity
``moe_infinity/kernel/mxfp4_gemm.py``. Weights are packed two FP4 values
per uint8 (low nibble = even K index) with one uint8 E8M0 scale
(exponent = byte - 127) per 32 K elements. Activations stay bf16
(weight-only); the MMA runs on bf16 tensor cores.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# BLOCK_K is pinned to 32 so each K-tile consumes exactly one scale per
# N row (the MXFP4 scale granularity).
_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 32},
        num_stages=3,
        num_warps=4,
    )
    for bm, bn in (
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
        (32, 64),
        (64, 32),
    )
]


@triton.jit
def _fp4_lookup(idx):
    # E2M1 decode: sign in bit 3, magnitude index in bits 0-2 mapping to
    # {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    sign = tl.where(idx >= 8, -1.0, 1.0)
    mag_idx = idx & 0x07
    mag = tl.where(mag_idx == 0, 0.0, 0.0)
    mag = tl.where(mag_idx == 1, 0.5, mag)
    mag = tl.where(mag_idx == 2, 1.0, mag)
    mag = tl.where(mag_idx == 3, 1.5, mag)
    mag = tl.where(mag_idx == 4, 2.0, mag)
    mag = tl.where(mag_idx == 5, 3.0, mag)
    mag = tl.where(mag_idx == 6, 4.0, mag)
    mag = tl.where(mag_idx == 7, 6.0, mag)
    return (sign * mag).to(tl.float32)


@triton.jit
def _ldexp(mantissa, exponent):
    # mantissa * 2^exponent via IEEE-754 exponent-field bitcast.
    exp_clamped = tl.minimum(tl.maximum(exponent, -126), 127)
    exp_bits = (exp_clamped + 127).to(tl.int32) << 23
    power_of_2 = exp_bits.to(tl.float32, bitcast=True)
    return mantissa * power_of_2


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _mxfp4_gemm_kernel(
    lhs_ptr,
    rhs_packed_ptr,
    rhs_scales_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_lhs_m,
    stride_lhs_k,
    stride_rhs_n,
    stride_rhs_k,
    stride_scales_n,
    stride_scales_k,
    stride_out_m,
    stride_out_n,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_packed = K // 2
    BLOCK_K_HALF: tl.constexpr = BLOCK_K // 2

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_block * BLOCK_K

        scale_idx = k_start // 32
        scale_ptrs = (
            rhs_scales_ptr
            + offs_n * stride_scales_n
            + scale_idx * stride_scales_k
        )
        scales = tl.load(scale_ptrs, mask=offs_n < N, other=127)
        exponents = scales.to(tl.int32) - 127

        offs_k_packed = k_start // 2 + tl.arange(0, BLOCK_K_HALF)
        rhs_ptrs = (
            rhs_packed_ptr
            + offs_n[:, None] * stride_rhs_n
            + offs_k_packed[None, :] * stride_rhs_k
        )
        rhs_mask = (offs_n[:, None] < N) & (offs_k_packed[None, :] < K_packed)
        rhs_packed = tl.load(rhs_ptrs, mask=rhs_mask, other=0)

        idx_lo = (rhs_packed & 0x0F).to(tl.int32)
        idx_hi = ((rhs_packed >> 4) & 0x0F).to(tl.int32)

        val_lo = _fp4_lookup(idx_lo)
        val_hi = _fp4_lookup(idx_hi)

        exp_bc = exponents[:, None] + tl.zeros(
            (1, BLOCK_K_HALF), dtype=tl.int32
        )
        val_lo = _ldexp(val_lo, exp_bc).to(tl.bfloat16)
        val_hi = _ldexp(val_hi, exp_bc).to(tl.bfloat16)

        offs_k_even = k_start + tl.arange(0, BLOCK_K_HALF) * 2
        offs_k_odd = offs_k_even + 1

        lhs_even_ptrs = (
            lhs_ptr
            + offs_m[:, None] * stride_lhs_m
            + offs_k_even[None, :] * stride_lhs_k
        )
        lhs_odd_ptrs = (
            lhs_ptr
            + offs_m[:, None] * stride_lhs_m
            + offs_k_odd[None, :] * stride_lhs_k
        )

        lhs_even_mask = (offs_m[:, None] < M) & (offs_k_even[None, :] < K)
        lhs_odd_mask = (offs_m[:, None] < M) & (offs_k_odd[None, :] < K)

        lhs_even = tl.load(lhs_even_ptrs, mask=lhs_even_mask, other=0.0)
        lhs_odd = tl.load(lhs_odd_ptrs, mask=lhs_odd_mask, other=0.0)

        acc += tl.dot(
            lhs_even.to(tl.bfloat16), tl.trans(val_lo), allow_tf32=False
        )
        acc += tl.dot(
            lhs_odd.to(tl.bfloat16), tl.trans(val_hi), allow_tf32=False
        )

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias_vals[None, :].to(tl.float32)

    out_ptrs = (
        out_ptr
        + offs_m[:, None] * stride_out_m
        + offs_n[None, :] * stride_out_n
    )
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def fused_mxfp4_gemm_triton(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """C = a @ dequant(b_packed).T + bias. a:[..., K] bf16, b:[N, K//2] u8."""
    orig = a.shape
    a2d = a.reshape(-1, orig[-1]).to(torch.bfloat16).contiguous()
    M, K = a2d.shape
    b_packed = b_packed.contiguous()
    scale = scale.contiguous()
    N, half_k = b_packed.shape
    assert (
        K == half_k * 2
    ), f"K mismatch: a {a2d.shape} vs packed {b_packed.shape}"
    assert K % 32 == 0, f"K={K} must be a multiple of 32"
    assert b_packed.dtype == torch.uint8
    assert scale.dtype == torch.uint8
    c2d = (
        out.reshape(-1, N)
        if out is not None
        else torch.empty((M, N), dtype=torch.bfloat16, device=a2d.device)
    )

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    _mxfp4_gemm_kernel[grid](
        a2d,
        b_packed,
        scale,
        bias if bias is not None else a2d,
        c2d,
        M,
        N,
        K,
        a2d.stride(0),
        a2d.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        scale.stride(0),
        scale.stride(1),
        c2d.stride(0),
        c2d.stride(1),
        HAS_BIAS=bias is not None,
    )
    return c2d.reshape(*orig[:-1], N)
