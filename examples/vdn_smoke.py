# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""VDN-H3 T2VA smoke: render, record goldens, emit metrics JSON.

Examples:
    # record goldens (bf16 reference)
    python examples/vdn_smoke.py --prompt "a red ball bouncing" \
        --seed 0 --evals 8 --frames 120 \
        --record-goldens tests/fixtures/vdn_goldens

    # verify against goldens with block streaming on
    python examples/vdn_smoke.py --prompt "a red ball bouncing" \
        --seed 0 --evals 8 --frames 120 --offload --block-stream 1 \
        --goldens tests/fixtures/vdn_goldens/vdn_goldens.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch

from omni_infinity.arch.vdn import VdnRunner


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True)
    p.add_argument("--checkpoint", default="OpenVDN/vdn-minimax-h3")
    p.add_argument("--variant", default="8-step", choices=["8-step", "50-step"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evals", type=int, default=8)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp8", action="store_true")
    p.add_argument(
        "--softmax-backend",
        default=None,
        choices=[None, "flex", "decomposed"],
    )
    p.add_argument("--offload", action="store_true")
    p.add_argument(
        "--block-stream",
        type=int,
        default=0,
        help="blocks per streamed group; 0 disables",
    )
    p.add_argument("--stream-text-encoder", action="store_true")
    p.add_argument(
        "--record-goldens",
        default=None,
        help="directory to write vdn_goldens.pt into",
    )
    p.add_argument(
        "--goldens", default=None, help="goldens .pt to compare against"
    )
    p.add_argument(
        "--metrics-out", default=None, help="write a metrics JSON here"
    )
    return p.parse_args()


def main():
    args = _args()
    runner = VdnRunner.from_pretrained(
        args.checkpoint,
        variant=args.variant,
        device=args.device,
        fp8=args.fp8,
        softmax_backend=args.softmax_backend,
        offload=args.offload,
        block_stream_blocks_per_group=args.block_stream,
        stream_text_encoder=args.stream_text_encoder,
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = runner.generate(
        args.prompt,
        seed=args.seed,
        num_evaluations=args.evals,
        num_frames=args.frames,
    )
    elapsed = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"generated in {elapsed:.1f}s "
        f"({elapsed / args.evals:.2f}s/eval incl. overhead), "
        f"peak {peak_gib:.2f} GiB"
    )

    payload = {
        "latents": result.latents.cpu(),
        "audio_latents": result.audio_latents.cpu(),
        "prompt": args.prompt,
        "seed": args.seed,
        "evals": args.evals,
        "frames": args.frames,
        "variant": args.variant,
        "fp8": args.fp8,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
    }
    if args.record_goldens:
        out_dir = pathlib.Path(args.record_goldens)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_dir / "vdn_goldens.pt")
        print(f"goldens -> {out_dir / 'vdn_goldens.pt'}")

    rms_rel = None
    if args.goldens:
        golden = torch.load(args.goldens, weights_only=False)
        ref = golden["latents"].float()
        got = result.latents.float().to(ref.device)
        rms_rel = (
            ((got - ref) ** 2).mean().sqrt() / (ref**2).mean().sqrt()
        ).item()
        print(f"rms_rel={rms_rel:.4f}")

    if args.metrics_out:
        metrics = {
            "wall_s": elapsed,
            "s_per_eval_incl_overhead": elapsed / args.evals,
            "peak_gib": peak_gib,
            "rms_rel": rms_rel,
            "config": vars(args),
        }
        pathlib.Path(args.metrics_out).write_text(
            json.dumps(metrics, indent=2, default=str)
        )


if __name__ == "__main__":
    main()
