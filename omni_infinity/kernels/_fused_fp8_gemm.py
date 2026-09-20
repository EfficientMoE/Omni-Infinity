# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Fused block-scaled FP8 (e4m3) weight-only GEMM: C = A @ dequant(B).T + bias.

Derived from EfficientMoE/BatchGen (Apache-2.0); see NOTICE. Hardened with
@triton.autotune, optional bias, and a corrected out=/3D return. Activations
stay bf16 (weight-only); the MMA runs on bf16 tensor cores.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# Scale indexing loads ONE scalar scale per tile, so BLOCK_SIZE_N must divide
# SCALE_BLOCK_M (128) and BLOCK_SIZE_K must divide SCALE_BLOCK_K (128).
_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": 8,
        },
        num_stages=ns, num_warps=nw,
    )
    for bm in (64, 128)
    for bn in (64, 128)
    for bk in (64, 128)
    for ns, nw in ((3, 4), (4, 8))
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _fgemm_kernel(
    a_ptr, b_ptr, c_ptr, scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    SCALE_BLOCK_M: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk
    )

    n_scale_k = tl.cdiv(K, SCALE_BLOCK_K)
    scale_row = (pid_n * BLOCK_SIZE_N) // SCALE_BLOCK_M

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_idx = k * BLOCK_SIZE_K
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] + k_idx < K)
        b_mask = (offs_bn[:, None] < N) & (offs_k[None, :] + k_idx < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_fp8 = tl.load(b_ptrs, mask=b_mask, other=0.0)
        scale = tl.load(
            scale_ptr + scale_row * n_scale_k + (k_idx // SCALE_BLOCK_K)
        )
        b = (b_fp8.to(tl.float32) * scale).to(tl.bfloat16)
        acc += tl.dot(a, tl.trans(b))
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        acc += bias[None, :].to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (
        offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def fused_fp8_gemm_triton(
    a: torch.Tensor,
    b_fp8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
    scale_block: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """C = a @ dequant(b_fp8).T + bias. a:[..., K] bf16, b_fp8:[N, K] e4m3."""
    orig = a.shape
    a2d = a.reshape(-1, orig[-1]).to(torch.bfloat16).contiguous()
    M, K = a2d.shape
    N, Kb = b_fp8.shape
    assert K == Kb, f"K mismatch: a {a2d.shape} vs b {b_fp8.shape}"
    b_fp8 = b_fp8.contiguous()
    n_blk = (N + scale_block[0] - 1) // scale_block[0]
    k_blk = (K + scale_block[1] - 1) // scale_block[1]
    scale = scale.reshape(n_blk, k_blk).to(torch.float32).contiguous()
    c2d = (
        out.reshape(-1, N)
        if out is not None
        else torch.empty((M, N), dtype=torch.bfloat16, device=a2d.device)
    )

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _fgemm_kernel[grid](
        a2d, b_fp8, c2d, scale, bias if bias is not None else a2d,
        M, N, K,
        a2d.stride(0), a2d.stride(1),
        b_fp8.stride(1), b_fp8.stride(0),  # swap => B.T
        c2d.stride(0), c2d.stride(1),
        SCALE_BLOCK_M=scale_block[0], SCALE_BLOCK_K=scale_block[1],
        HAS_BIAS=bias is not None,
    )
    return c2d.reshape(*orig[:-1], N)
