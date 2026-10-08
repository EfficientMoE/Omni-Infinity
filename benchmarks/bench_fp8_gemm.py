# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the fused block-scaled FP8 weight-only GEMM.

For each real H3 Linear shape and representative diffusion M values, measure
three implementations:

  (A) baseline_bf16matmul -- current per-row dequant->bf16 then ``F.linear``.
  (B) fused_w8a16         -- facade ``fused_fp8_gemm`` (block-wise, weight-only,
                             no bf16 weight materialization).
  (C) scaled_mm_w8a8      -- ``torch._scaled_mm`` with dummy unit scales
                             (mm-only absolute ceiling; not a real impl).
  (D) scaled_mm_backend   -- facade ``fused_fp8_gemm(backend="scaled_mm")``
                             (Rung A drop-in: per-call dequant->requant).
  (E) scaled_mm_prequant  -- per-row pre-quantized weight + dynamic
                             per-token activation quant (Rung A deploy).
  (F) cutlass_sm120       -- lazy-JIT CUTLASS w8a8 preserving 128x128
                             weight scales (Rung B deploy).

Reports per-op median CUDA-event latency, peak memory
(``torch.cuda.max_memory_allocated``), and streamed weight bytes (bf16 vs fp8).

Per Oracle: the fused path's real win is halved weight bytes + peak memory
(the bf16 weight temp disappears), NOT compute -- the MMA still runs on bf16
tensor cores, so latency may trail cuBLAS at large M. Do not claim a latency
win unless measured.

Run (idle GPU only; keep tensors within the free budget):
    CUDA_VISIBLE_DEVICES=5 python benchmarks/bench_fp8_gemm.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from omni_infinity.fp8 import quantize_per_row_fp8
from omni_infinity.kernels import fused_fp8_gemm, quantize_block_fp8
from omni_infinity.kernels._scaled_mm import _quantize_per_row


def _time(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def bench(M: int, N: int, K: int, dev: str = "cuda") -> None:
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    b = torch.randn(N, dtype=torch.bfloat16, device=dev)

    qb, sb = quantize_block_fp8(w.cpu())
    qb, sb = qb.to(dev), sb.to(dev)
    qr, sr = quantize_per_row_fp8(w)
    qr = qr.to(dev)
    sr = sr.to(dev)

    def baseline():
        wv = (qr.to(torch.float32) * sr).to(torch.bfloat16)
        return F.linear(x, wv, b)

    def fused():
        return fused_fp8_gemm(x, qb, sb, b)

    # torch._scaled_mm wants b column-major fp8; w.t() of a contiguous [N, K]
    # weight is exactly that. Per-row activation + per-col weight scales.
    xf = x.to(torch.float8_e4m3fn)
    wf = w.t().to(torch.float8_e4m3fn)
    sa = torch.ones(M, 1, device=dev)
    sw = torch.ones(1, N, device=dev)

    def w8a8():
        return torch._scaled_mm(
            xf, wf, scale_a=sa, scale_b=sw, out_dtype=torch.bfloat16
        )

    # Rung A facade path: block-scale input, per-call dequant->per-row
    # requant inside the backend (realistic drop-in cost).
    def scaled_mm_backend():
        return fused_fp8_gemm(x, qb, sb, b, backend="scaled_mm")

    # Rung A deploy path: weight pre-quantized per-row offline; only the
    # per-token activation quant stays in the hot loop. Closest honest
    # figure to the "true FP8 compute" ceiling with real scales (the
    # dummy-scale w8a8 row above remains the absolute mm-only ceiling).
    wt = qr.t()
    swt = sr.t()

    def scaled_mm_prequant():
        q_a, s_a = _quantize_per_row(x)
        c = torch._scaled_mm(
            q_a, wt, scale_a=s_a, scale_b=swt, out_dtype=torch.bfloat16
        )
        return c + b

    def cutlass_sm120():
        return fused_fp8_gemm(x, qb, sb, b, backend="cutlass_sm120")

    bf16_mb = 2 * N * K / 1e6
    fp8_mb = N * K / 1e6
    print(
        f"M={M} N={N} K={K} | weight bytes "
        f"bf16={bf16_mb:.1f}MB fp8={fp8_mb:.1f}MB"
    )
    impls = (
        ("baseline_bf16matmul", baseline),
        ("fused_w8a16", fused),
        ("scaled_mm_w8a8", w8a8),
        ("scaled_mm_backend", scaled_mm_backend),
        ("scaled_mm_prequant", scaled_mm_prequant),
        ("cutlass_sm120", cutlass_sm120),
    )
    for name, fn in impls:
        try:
            torch.cuda.reset_peak_memory_stats()
            fn()
            torch.cuda.synchronize()
            lat = _time(fn)
            mem = torch.cuda.max_memory_allocated() / 1e6
            print(f"  {name:22s} {lat:8.3f} ms  peak {mem:8.1f} MB")
        except Exception as exc:
            reason = str(exc).splitlines()[0][:100]
            print(f"  {name:22s} n/a ({reason})")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this benchmark")
    # Real H3 shapes: the P1 plan's M in {1k, 8k, 32k} x the dominant
    # transformer Linears (N=7168,K=5376), (N=5376,K=7168) and the
    # AdaLN-ish (N=28672,K=5376).
    shapes = [
        (1024, 7168, 5376),
        (8192, 7168, 5376),
        (32768, 7168, 5376),
        (1024, 5376, 7168),
        (8192, 5376, 7168),
        (32768, 5376, 7168),
        (1024, 28672, 5376),
        (8192, 28672, 5376),
        (32768, 28672, 5376),
    ]
    for M, N, K in shapes:
        bench(M, N, K)
        torch.cuda.empty_cache()
