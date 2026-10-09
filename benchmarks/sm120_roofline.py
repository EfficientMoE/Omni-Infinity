#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Measured sm120 roofline: achieved HBM bandwidth and BF16/FP8 GEMM TFLOPS.

Published-spec-only discipline: this does NOT cite third-party tensor-TFLOPS.
It measures the achieved ceiling on THIS card so downstream component-binding
labels (M2) rest on a measured ridge point, not a reverse-engineered figure.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path

import torch


def _time_cuda(fn: Callable[[], object], iters: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return statistics.median(times)


def measure_bandwidth(gib: float = 2.0, iters: int = 50) -> dict:
    n = int(gib * 1024**3 // 2)
    x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    z = torch.empty_like(x)
    nbytes = x.numel() * x.element_size()
    t_copy = _time_cuda(lambda x=x, y=y: y.copy_(x), iters)
    t_triad = _time_cuda(lambda x=x, y=y, z=z: torch.add(x, y, out=z), iters)
    bw_copy = 2 * nbytes / t_copy / 1e9
    bw_triad = 3 * nbytes / t_triad / 1e9
    del x, y, z
    torch.cuda.empty_cache()
    return {
        "copy_gbps": bw_copy,
        "triad_gbps": bw_triad,
        "achieved_bw_gbps": max(bw_copy, bw_triad),
        "tensor_gib": gib,
    }


def measure_bf16_gemm(sizes: list[int], iters: int = 30) -> dict:
    table: dict[int, float] = {}
    peak = 0.0
    for s in sizes:
        a = torch.randn(s, s, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(s, s, dtype=torch.bfloat16, device="cuda")
        t = _time_cuda(lambda a=a, b=b: torch.matmul(a, b), iters)
        tflops = 2 * s**3 / t / 1e12
        table[s] = tflops
        peak = max(peak, tflops)
        del a, b
        torch.cuda.empty_cache()
    return {"per_size_tflops": table, "bf16_peak_tflops": peak}


def measure_fp8_gemm(sizes: list[int], iters: int = 30) -> dict:
    try:
        fp8 = torch.float8_e4m3fn
        table: dict[int, float] = {}
        peak = 0.0
        scale = torch.tensor(1.0, device="cuda")
        for s in sizes:
            a = torch.randn(s, s, device="cuda").to(fp8)
            b = torch.randn(s, s, device="cuda").to(fp8).t()

            def run(a=a, b=b):
                return torch._scaled_mm(
                    a,
                    b,
                    scale_a=scale,
                    scale_b=scale,
                    out_dtype=torch.bfloat16,
                )

            run()
            t = _time_cuda(run, iters)
            table[s] = 2 * s**3 / t / 1e12
            peak = max(peak, table[s])
            del a, b
            torch.cuda.empty_cache()
        return {"per_size_tflops": table, "fp8_peak_tflops": peak}
    except Exception as exc:  # noqa: BLE001
        return {
            "per_size_tflops": {},
            "fp8_peak_tflops": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise SystemExit(
            "set CUDA_VISIBLE_DEVICES to exactly one idle GPU index"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("exactly one visible CUDA device is required")

    card = torch.cuda.get_device_name(0)
    cuda_cc = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
    sizes = [4096, 8192, 16384, 24576]
    bw = measure_bandwidth()
    bf16 = measure_bf16_gemm(sizes)
    fp8 = measure_fp8_gemm(sizes)

    ridge_bf16 = (bf16["bf16_peak_tflops"] * 1e12) / (
        bw["achieved_bw_gbps"] * 1e9
    )
    ridge_fp8 = (
        (fp8["fp8_peak_tflops"] * 1e12) / (bw["achieved_bw_gbps"] * 1e9)
        if fp8["fp8_peak_tflops"]
        else None
    )

    payload = {
        "card": card,
        "cuda_cc": cuda_cc,
        "achieved_bw_gbps": bw["achieved_bw_gbps"],
        "bandwidth_detail": bw,
        "bf16_peak_tflops": bf16["bf16_peak_tflops"],
        "bf16_per_size_tflops": bf16["per_size_tflops"],
        "fp8_peak_tflops": fp8["fp8_peak_tflops"],
        "fp8_detail": fp8,
        "ridge_ai_bf16": ridge_bf16,
        "ridge_ai_fp8": ridge_fp8,
        "note": (
            "measured on this card; third-party claims "
            "(BF16 ~271 / FP8 ~529 TFLOPS) NOT used"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
