# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""P7 Phase 2 spike: degree-2 Ulysses sequence parallelism on the denoiser.

Splits one FL2VA denoise across two intra-socket GPUs with diffusers'
native context parallelism: ``MiniMaxH3Transformer3DModel`` ships a
``_cp_plan`` that shards the packed video+audio+text sequence at the
first transformer block and gathers it back at the output heads, and the
attention dispatch does the Ulysses all-to-all.  The token-refiner text
attention runs *before* the sequence is sharded, so its per-instance CP
config is cleared (otherwise it all-to-alls a replicated tensor).

This is the single-request *latency* path, orthogonal to the stage
pipeline (which is throughput).  Parity tier: not bitwise vs single-GPU
(CP changes head count 56->28 and reduction order), but the two ranks'
gathered outputs must be bitwise-identical and must match the committed
goldens within rms_rel <= 1e-3 (video and audio).

Run (one idle PIX pair on socket 0, e.g. GPUs 0,1):
  # 1. capture the post-encoder state once (single process):
  HF_HOME=... HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMNI_H3_CHECKPOINT=<snap> OMNI_H3_STORE=<store> CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=. python benchmarks/ulysses_spike.py --mode capture
  # 2. single-GPU baseline latency + goldens parity:
  ... CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python benchmarks/ulysses_spike.py --mode baseline --trials 5
  # 3. degree-2 Ulysses (two ranks, gloo+nccl):
  ... CUDA_VISIBLE_DEVICES=0,1 NCCL_MIN_NCHANNELS=8 NCCL_P2P_LEVEL=SYS \
  PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  benchmarks/ulysses_spike.py --mode cp --degree 2 --trials 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

GOLDENS = Path("tests/fixtures/goldens/fl2va_goldens.pt")
REFERENCE = Path("tests/fixtures/ref.png")
STATE = Path(
    os.environ.get("OMNI_USP_STATE", "/tmp/opencode/cp_fl2va_encoder_state.pt")
)


def _checkpoint() -> str:
    return os.environ["OMNI_H3_CHECKPOINT"]


def _store() -> str | None:
    return os.environ.get("OMNI_H3_STORE")


def _load_runner(
    components: tuple[str, ...], device: str, store: tuple[str, ...]
):
    from omni_infinity.runner import ReferenceRunner

    return ReferenceRunner.from_pretrained(
        _checkpoint(),
        device=device,
        offload=False,
        components=components,
        store_dir=_store(),
        store_components=store,
    )


def capture() -> None:
    from PIL import Image

    from omni_infinity.runner import resolve_resolution
    from omni_infinity.serve.roles import Role
    from omni_infinity.serve.split import prepare_state, run_stage

    payload = torch.load(GOLDENS, weights_only=False)
    height, width = resolve_resolution(payload["resolution"])
    runner = _load_runner(
        ("text_encoder", "tokenizer", "processor", "vae"), "cuda:0", ("vae",)
    )
    state = prepare_state(
        runner.pipeline,
        prompt=payload["prompt"],
        image=Image.open(REFERENCE).convert("RGB"),
        height=height,
        width=width,
        num_frames=payload["frames"],
        num_inference_steps=payload["steps"],
        generator=torch.Generator("cpu").manual_seed(payload["seed"]),
        output_type="np",
    )
    state = run_stage(runner.pipeline, Role.ENCODER, state)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, STATE)
    print(f"  saved post-encoder state -> {STATE}")


def _run_denoise(pipeline: Any, device: str) -> tuple[Any, float]:
    from omni_infinity.serve.roles import Role
    from omni_infinity.serve.split import move_state_tensors, run_stage

    state = torch.load(STATE, weights_only=False)
    move_state_tensors(state, device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    state = run_stage(pipeline, Role.DENOISER, state)
    torch.cuda.synchronize(device)
    return state, time.perf_counter() - start


def _outputs(state: Any) -> tuple[Any, Any]:
    video = state.get("latents")
    audio = state.get("audio_latents")
    video = video.detach().cpu() if torch.is_tensor(video) else video
    audio = audio.detach().cpu() if torch.is_tensor(audio) else audio
    return video, audio


def _rms_rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    got = got.float().flatten()
    ref = ref.float().flatten()
    denom = torch.linalg.vector_norm(ref).clamp_min(1e-12)
    return (torch.linalg.vector_norm(got - ref) / denom).item()


_PARITY_TOL = 1e-3


def _finite_within_tol(got: Any, ref: torch.Tensor) -> bool:
    if not (torch.is_tensor(got) and bool(torch.isfinite(got).all())):
        return False
    return _rms_rel(got, ref) <= _PARITY_TOL


def _parity_ok(video: Any, audio: Any, payload: dict[str, Any]) -> bool:
    if not _finite_within_tol(video, payload["latents"]):
        return False
    if payload["audio_latents"] is not None:
        return _finite_within_tol(audio, payload["audio_latents"])
    return True


def _report_parity(video: Any, audio: Any, payload: dict[str, Any]) -> None:
    vr = _rms_rel(video, payload["latents"])
    bitwise = torch.equal(video, payload["latents"])
    finite = bool(torch.isfinite(video).all())
    print(f"  video: rms_rel={vr:.3e} bitwise={bitwise} finite={finite}")
    if payload["audio_latents"] is not None and audio is not None:
        ar = _rms_rel(audio, payload["audio_latents"])
        print(f"  audio: rms_rel={ar:.3e}")


def baseline(trials: int) -> None:
    payload = torch.load(GOLDENS, weights_only=False)
    pipeline = _load_runner(
        ("transformer", "scheduler", "audio_scheduler"),
        "cuda:0",
        ("transformer",),
    ).pipeline
    pipeline.transformer.set_attention_backend("native")
    state, _ = _run_denoise(pipeline, "cuda:0")
    video, audio = _outputs(state)
    print("baseline parity vs goldens (1 GPU, native):")
    _report_parity(video, audio, payload)
    ok = _parity_ok(video, audio, payload)
    times = [_run_denoise(pipeline, "cuda:0")[1] for _ in range(trials)]
    print(
        f"  baseline denoise latency: {statistics.median(times):.3f}s median "
        f"over {trials}"
    )
    if not ok:
        raise SystemExit("baseline parity gate FAILED vs goldens")


def _cross_rank_equal(tensor: Any, device: str) -> bool:
    import torch.distributed as dist

    if not torch.is_tensor(tensor):
        return True
    local = tensor.to(device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return all(torch.equal(gathered[0], other) for other in gathered)


def cp(trials: int, degree: int) -> None:
    import torch.distributed as dist
    from diffusers import ContextParallelConfig

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method="env://")
    gate_passed = False
    try:
        payload = torch.load(GOLDENS, weights_only=False)
        pipeline = _load_runner(
            ("transformer", "scheduler", "audio_scheduler"),
            device,
            ("transformer",),
        ).pipeline
        transformer = pipeline.transformer
        transformer.set_attention_backend("native")
        transformer.enable_parallelism(
            config=ContextParallelConfig(
                ring_degree=1, ulysses_degree=degree, ulysses_anything=True
            )
        )
        # The token refiner runs before the packed sequence is sharded, so
        # its replicated text attention must not do the Ulysses all-to-all.
        for block in transformer.token_refiner.refiner_blocks:
            block.attn.processor._parallel_config = None
        assert all(
            block.attn.processor._parallel_config is None
            for block in transformer.token_refiner.refiner_blocks
        )

        dist.barrier()
        state, _ = _run_denoise(pipeline, device)
        video, audio = _outputs(state)
        video_equal = _cross_rank_equal(video, device)
        audio_equal = _cross_rank_equal(audio, device)
        gate_passed = (
            video_equal and audio_equal and _parity_ok(video, audio, payload)
        )
        if rank == 0:
            ranks = dist.get_world_size()
            print(f"cp parity (degree {degree} Ulysses, {ranks} ranks):")
            print(
                f"  cross_rank_bitwise: video={video_equal} audio={audio_equal}"
            )
            _report_parity(video, audio, payload)
            print(f"  parity gate (rms_rel <= {_PARITY_TOL:g}): {gate_passed}")

        times = []
        for _ in range(trials):
            dist.barrier()
            _, elapsed = _run_denoise(pipeline, device)
            scalar = torch.tensor([elapsed], device=device)
            dist.all_reduce(scalar, op=dist.ReduceOp.MAX)
            times.append(scalar.item())
        if rank == 0:
            print(
                f"  cp denoise latency: {statistics.median(times):.3f}s median "
                f"over {trials}"
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
    if not gate_passed:
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("capture", "baseline", "cp")
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--degree", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "capture":
        capture()
    elif args.mode == "baseline":
        baseline(args.trials)
    else:
        cp(args.trials, args.degree)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
