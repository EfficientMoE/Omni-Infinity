# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the fused MXFP4 weight-only GEMM against the W4/W8 field.

P5 plan microbench: for the dominant H3 Linear shapes (incl. N=28672),
measure

  (A) baseline_bf16        -- ``F.linear`` with the true bf16 weight.
  (B) fused_w8a16          -- facade ``fused_fp8_gemm`` (inc-8 kernel).
  (C) fused_mxfp4          -- facade ``fused_mxfp4_gemm`` (this plan).
  (D) marlin_w4a16         -- MoE-Infinity's compiled Marlin kernel
                              (fp16 activations; fastest known W4 path).
  (E) native_fp4           -- BatchGen SM120 CUDA kernel, probed only:
                              grouped-MoE semantics; dense adaptation is
                              the plan's phase 2.

Reports per-op median CUDA-event latency, peak memory, and weight bytes
(bf16 / fp8 / fp4+scales). Accuracy is NOT asserted here (the golden
harness owns that); a loose rel-error probe guards against wiring bugs.

Run (idle GPU only):
    CUDA_VISIBLE_DEVICES=5 python benchmarks/bench_mxfp4_gemm.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from omni_infinity.kernels import (
    fused_fp8_gemm,
    fused_mxfp4_gemm,
    quantize_block_fp8,
    quantize_mxfp4,
)


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


def _marlin_impl(x, w, dev):
    from moe_infinity.kernel.marlin_gemm import (
        is_marlin_available,
        marlin_gemm,
        marlin_quantize,
        prepare_workspace,
    )

    if not is_marlin_available():
        raise RuntimeError("moe_infinity._marlin extension not built")
    N, K = w.shape
    xh = x.to(torch.float16)
    ref = F.linear(xh.float(), w.float()).half()
    # groupsize=-1 is deliberately NOT probed: on this SM120 install it
    # dies with cudaErrorIllegalInstruction, which poisons the CUDA
    # context for every later impl in the process. groupsize=128 fails
    # cleanly on unsupported shapes, so it is the only safe probe.
    last_exc: Exception | None = None
    for candidate in (w.t().contiguous(), w.contiguous()):
        try:
            packed, scales = marlin_quantize(
                candidate.to(torch.float16).to(dev), groupsize=128
            )
            workspace = prepare_workspace(N, dev)

            def run():
                return marlin_gemm(xh, packed, scales, workspace)

            got = run()
            if got.shape != (x.shape[0], N):
                raise RuntimeError(f"marlin output shape {got.shape}")
            rel = (got.float() - ref.float()).norm() / ref.float().norm()
            if rel > 0.2:
                raise RuntimeError(f"marlin orientation mismatch rel={rel:.3f}")
            return run
        except Exception as exc:  # try the other weight orientation
            last_exc = exc
    raise RuntimeError(f"marlin unusable: {last_exc}")


def bench(M: int, N: int, K: int, dev: str = "cuda") -> None:
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    b = torch.randn(N, dtype=torch.bfloat16, device=dev)

    q8, s8 = quantize_block_fp8(w.cpu())
    q8, s8 = q8.to(dev), s8.to(dev)
    q4, s4 = quantize_mxfp4(w.cpu())
    q4, s4 = q4.to(dev), s4.to(dev)

    def baseline():
        return F.linear(x, w, b)

    def fused_fp8():
        return fused_fp8_gemm(x, q8, s8, b)

    def fused_fp4():
        return fused_mxfp4_gemm(x, q4, s4, b)

    bf16_mb = 2 * N * K / 1e6
    fp8_mb = N * K / 1e6
    fp4_mb = (N * K / 2 + N * K / 32) / 1e6
    print(
        f"M={M} N={N} K={K} | weight bytes bf16={bf16_mb:.1f}MB "
        f"fp8={fp8_mb:.1f}MB fp4={fp4_mb:.1f}MB"
    )

    impls: list[tuple[str, object]] = [
        ("baseline_bf16", baseline),
        ("fused_w8a16", fused_fp8),
        ("fused_mxfp4", fused_fp4),
    ]
    try:
        impls.append(("marlin_w4a16", _marlin_impl(x, w, dev)))
    except Exception as exc:
        print(f"  marlin_w4a16           n/a ({str(exc)[:90]})")
    try:
        import batchgen_kernels.moe._C_mega_moe_sm120  # noqa: F401

        print(
            "  native_fp4             present but grouped-MoE semantics; "
            "dense adaptation is phase 2"
        )
    except Exception as exc:
        print(f"  native_fp4             n/a ({str(exc)[:90]})")

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
    shapes = [
        (4096, 7168, 5376),
        (8192, 7168, 5376),
        (4096, 5376, 7168),
        (8192, 5376, 7168),
        (4096, 28672, 5376),
        (8192, 28672, 5376),
    ]
    for M, N, K in shapes:
        bench(M, N, K)
        torch.cuda.empty_cache()
