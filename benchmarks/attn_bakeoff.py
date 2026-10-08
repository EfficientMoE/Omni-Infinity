# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Window-softmax backend bake-off for VDN on sm120 (#42 P3).

Standalone microbench: backends x seqlen x {window, dense} mask at H3
geometry (56 heads x 128 head-dim, chunk=5 radius=1 anchors=both),
CUDA-event timed, with max-abs / rms_rel error vs the fp32 eager
reference. Unavailable backends are recorded as rows, never crash the
matrix. Backend selection is env-gatable (OMNI_ATTN_BACKENDS, the
BatchGen v4_flashmla_adapter dispatch pattern) and --dump-probe writes
the resolution/availability record per run.

Run (flashinfer JIT needs the toolkit root):
  CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \\
    python benchmarks/attn_bakeoff.py --seqlens 8k 20k 37k
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "vdn-minimax-h3"))

from src.models.sequence_layout import SequenceLayout  # noqa: E402
from src.models.softmax_attention.window import (  # noqa: E402
    window_bounds,
    window_softmax_reference,
)

from omni_infinity.arch import vdn_attention as va  # noqa: E402

FIELDS = (
    "backend",
    "mask",
    "seq_tokens",
    "num_frames",
    "heads",
    "head_dim",
    "ms_median",
    "max_abs_err",
    "rms_rel",
    "status",
    "detail",
)
BACKENDS = ("decomposed", "flex", "cudnn", "fmha-v2", "fa4", "sage")


def _time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _window_fn(backend, layout, bounds, scale, anchors):
    if backend == "decomposed":
        from src.models.softmax_attention.decomposed import (
            window_softmax_decomposed,
        )

        return lambda q, k, v: window_softmax_decomposed(
            q, k, v, layout, bounds, scale, anchor_frames=anchors
        )
    if backend == "flex":
        from src.models.softmax_attention.flex_attention import (
            build_window_block_mask,
            window_softmax_flex,
        )

        mask = build_window_block_mask(
            layout, bounds, torch.device("cuda"), anchor_frames=anchors
        )
        return lambda q, k, v: window_softmax_flex(
            q, k, v, mask, scale, inference=True
        )
    fn = va.make_window_softmax(backend)
    return lambda q, k, v: fn(
        q, k, v, layout, bounds, scale, anchor_frames=anchors
    )


def _dense_fn(backend, scale):
    # One dense attention over all rows: each backend's dense-leg kernel.
    from torch.nn.attention import SDPBackend

    if backend in ("decomposed", "flex"):
        order = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
        ]
        return lambda q, k, v: va._sdpa_dense(q, k, v, scale, order)
    if backend == "cudnn":
        return lambda q, k, v: va._cudnn_dense(q, k, v, scale)
    if backend == "sage":
        return lambda q, k, v: va._sage_dense(q, k, v, scale)
    if backend == "fa4":

        def fa4(q, k, v):
            from flash_attn.cute.interface import flash_attn_func

            out = flash_attn_func(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                softmax_scale=scale,
                causal=False,
            )
            return (out[0] if isinstance(out, tuple) else out)[0]

        return fa4

    def fmha(q, k, v):
        import flashinfer

        return flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=False, sm_scale=scale
        )

    return fmha


def run_cell(backend, mask, layout, bounds, q, k, v, scale, args):
    row = {
        "backend": backend,
        "mask": mask,
        "seq_tokens": layout.seq_len,
        "num_frames": layout.num_frames,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "ms_median": "",
        "max_abs_err": "",
        "rms_rel": "",
        "status": "ok",
        "detail": "",
    }
    ok, detail = va.backend_available(backend)
    if not ok:
        row.update(status="unavailable", detail=detail)
        return row
    try:
        fn = (
            _window_fn(backend, layout, bounds, scale, args.anchors)
            if mask == "window"
            else _dense_fn(backend, scale)
        )
        out = fn(q, k, v)
        if args.check:
            if mask == "window":
                ref = window_softmax_reference(
                    q.float(),
                    k.float(),
                    v.float(),
                    layout,
                    bounds,
                    scale,
                    anchor_frames=args.anchors,
                )
            else:
                # MATH materialises the full score matrix (320 GB at 37k);
                # fp32 mem-efficient SDPA is the tractable oracle.
                ref = va._sdpa_dense(
                    q.float(),
                    k.float(),
                    v.float(),
                    scale,
                    [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION],
                )
            diff = out.float() - ref
            row["max_abs_err"] = f"{diff.abs().max().item():.4e}"
            rel = diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
            row["rms_rel"] = f"{rel.item():.4e}"
            del ref, diff
        del out
        ms = _time(lambda: fn(q, k, v), args.warmup, args.iters)
        row["ms_median"] = f"{ms:.3f}"
    except Exception as exc:  # noqa: BLE001 - matrix must record, not crash
        row.update(status="fail", detail=f"{type(exc).__name__}: {exc}"[:200])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seqlens", nargs="+", default=["8k", "20k", "37k"])
    p.add_argument("--backends", nargs="+", default=list(BACKENDS))
    p.add_argument("--heads", type=int, default=56)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--tokens-per-frame", type=int, default=1008)
    p.add_argument("--global-tokens", type=int, default=512)
    p.add_argument("--chunk", type=int, default=5)
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--anchors", default="both")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-check", dest="check", action="store_false")
    p.add_argument("--masks", nargs="+", default=["window", "dense"])
    p.add_argument(
        "--results-dir", type=Path, default=Path("results/attn_bakeoff")
    )
    p.add_argument("--dump-probe", action="store_true")
    args = p.parse_args()

    if env := os.environ.get("OMNI_ATTN_BACKENDS"):
        args.backends = [b for b in args.backends if b in env.split(",")]

    torch.manual_seed(0)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    probe = {
        "device": torch.cuda.get_device_name(0),
        "cc": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "backends": {
            b: dict(zip(("available", "detail"), va.backend_available(b)))
            for b in args.backends
        },
    }
    print(json.dumps(probe, indent=2))
    if args.dump_probe:
        (args.results_dir / "probe.json").write_text(
            json.dumps(probe, indent=2)
        )

    for spec in args.seqlens:
        raw = spec.rstrip("k")
        target = int(float(raw) * 1024) if "k" in spec else int(spec)
        frames = max(args.chunk, round(target / args.tokens_per_frame))
        layout = SequenceLayout(
            seq_len=args.global_tokens + frames * args.tokens_per_frame,
            video_start=args.global_tokens,
            num_frames=frames,
            tokens_per_frame=args.tokens_per_frame,
        )
        bounds = window_bounds(frames, radius=args.radius, chunk=args.chunk)
        scale = args.head_dim**-0.5
        shape = (layout.seq_len, args.heads, args.head_dim)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for mask in args.masks:
            for backend in args.backends:
                row = run_cell(
                    backend, mask, layout, bounds, q, k, v, scale, args
                )
                rows.append(row)
                print(
                    f"{backend:12s} {mask:7s} T={layout.seq_len:6d} "
                    f"ms={row['ms_median'] or '-':>9s} "
                    f"rms_rel={row['rms_rel'] or '-':>10s} "
                    f"[{row['status']}] {row['detail']}"
                )
        del q, k, v
        torch.cuda.empty_cache()
        va._PLAN_CACHE.clear()

    out = args.results_dir / "results.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")

    print("\n| backend | mask | tokens | ms | rms_rel | status |")
    print("|---|---|---:|---:|---:|---|")
    for r in rows:
        print(
            f"| {r['backend']} | {r['mask']} | {r['seq_tokens']} "
            f"| {r['ms_median'] or '-'} | {r['rms_rel'] or '-'} "
            f"| {r['status']} |"
        )


if __name__ == "__main__":
    main()
