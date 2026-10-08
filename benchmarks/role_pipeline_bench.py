# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""P7 bench grid: role-pipeline throughput across GPU topologies.

Measures jobs/hour for the cross-process encoder/denoiser/decoder role
pipeline at 2/3/4/6 GPUs against the single-GPU streamed baseline, over
a VidProM multi-prompt trace. Each run gates a golden-prompt job bitwise
against the committed goldens (role arms gate a dedicated parity job
before the timed trace; the baseline gates its first job) — the handoff
is data-movement-only, so a fully resident role run must match the
single-GPU latents exactly.

Topologies (the denoiser is the throughput bottleneck, so the scaling
unit isolates it; replicas are independent — no cross-replica
collectives — so N replicas deliver ~N x the per-replica rate):

  2  : one replica  {enc+dec share cuda:a, denoiser cuda:b}  (PIX pair)
  3  : one replica  {enc cuda:0, denoiser cuda:1, dec cuda:2}
  4  : two replicas of the 2-GPU unit on PIX pairs (0,1) + (2,3)
  6  : three replicas of the 2-GPU unit on (0,1) + (2,3) + (4,5)

Role arms are fully resident (offload=False); they need only
OMNI_H3_CHECKPOINT + OMNI_H3_STORE. The baseline uses the streamed
offload recipe (the ~144 GB bf16 FL2VA set does not fit one 96 GB GPU)
— same env knobs as tests/test_split_parity.py.

Run the role grid (all six GPUs visible; devices pinned per replica):
  HF_HOME=... HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMNI_H3_CHECKPOINT=<snapshot> OMNI_H3_STORE=<store> \
  python benchmarks/role_pipeline_bench.py --topology 6 --jobs 24 \
      --json /tmp/opencode/p7-bench-6gpu.json

Run the single-GPU baseline (pin one idle GPU):
  CUDA_VISIBLE_DEVICES=<idle> HF_HOME=... HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 OMNI_H3_CHECKPOINT=<snapshot> \
  OMNI_H3_STORE=<store> OMNI_H3_ADALN_CACHE=1 OMNI_H3_BLOCK_STREAM=1 \
  OMNI_H3_STREAM_TEXT_ENCODER=1 \
  python benchmarks/role_pipeline_bench.py --topology 1 --jobs 4 \
      --json /tmp/opencode/p7-bench-1gpu.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import torch

GOLDENS = Path("tests/fixtures/goldens/fl2va_goldens.pt")
REFERENCE = Path("tests/fixtures/ref.png")

_PAIR_TOPOLOGIES: dict[str, list[tuple[int, int]]] = {
    "2": [(0, 1)],
    "4": [(0, 1), (2, 3)],
    "6": [(0, 1), (2, 3), (4, 5)],
}


def _role_map(enc: str, den: str, dec: str) -> dict[Any, str]:
    from omni_infinity.serve.roles import Role

    return {Role.ENCODER: enc, Role.DENOISER: den, Role.DECODER: dec}


def topology_device_maps(name: str) -> list[dict[Any, str]]:
    """Return the per-replica device maps for a topology name."""

    if name == "3":
        return [_role_map("cuda:0", "cuda:1", "cuda:2")]
    if name in _PAIR_TOPOLOGIES:
        return [
            _role_map(f"cuda:{a}", f"cuda:{b}", f"cuda:{a}")
            for a, b in _PAIR_TOPOLOGIES[name]
        ]
    raise ValueError(f"unknown topology {name!r}; expected 1|2|3|4|6")


def trace_prompts(pool: str, count: int) -> list[str]:
    """``count`` prompts cycled from the committed VidProM/VBench pool.

    The denoiser cost is prompt-independent (fixed NFE on fixed-shape
    latents), so cycling the pool keeps the trace denoiser-bound while
    giving the encoder realistic prompt variety.
    """

    from benchmarks.caches.workload import load_prompts

    prompts = list(dict.fromkeys(load_prompts("fixture", pool=pool)))
    if not prompts:
        raise ValueError(f"prompt pool {pool!r} is empty")
    return [prompts[index % len(prompts)] for index in range(count)]


def _request(
    payload: dict[str, Any],
    image: Any,
    prompt: str,
    height: int,
    width: int,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "image": image,
        "height": height,
        "width": width,
        "num_frames": payload["frames"],
        "num_inference_steps": payload["steps"],
        "generator": torch.Generator("cpu").manual_seed(payload["seed"]),
        "output_type": "np",
    }


def _audio_ok(audio_latents: Any, payload: dict[str, Any]) -> bool:
    if payload["audio_latents"] is None:
        return audio_latents is None
    if audio_latents is None:
        return False
    tensor = (
        audio_latents.cpu() if torch.is_tensor(audio_latents) else audio_latents
    )
    return torch.equal(tensor, payload["audio_latents"])


def _video_ok(latents: Any, payload: dict[str, Any]) -> bool:
    tensor = latents.cpu() if torch.is_tensor(latents) else latents
    return torch.equal(tensor, payload["latents"])


def _steady_spacing(times: list[float], start: float) -> float:
    """Median inter-completion delta over the last half of a replica."""

    deltas = []
    previous = start
    for finished in times:
        deltas.append(finished - previous)
        previous = finished
    tail = deltas[len(deltas) // 2 :] or deltas
    return statistics.median(tail) if tail else float("nan")


def run_baseline(
    jobs: int,
    payload: dict[str, Any],
    image: Any,
) -> dict[str, Any]:
    """Serial single-GPU streamed baseline; parity-reports job 0."""

    import os

    from omni_infinity.runner import ReferenceRunner

    runner = ReferenceRunner.from_pretrained(
        os.environ.get("OMNI_H3_CHECKPOINT", "MiniMaxAI/MiniMax-H3"),
        offload=os.environ.get("OMNI_H3_OFFLOAD", "1") == "1",
        store_dir=os.environ.get("OMNI_H3_STORE"),
        store_components=tuple(
            os.environ.get(
                "OMNI_H3_STORE_COMPONENTS", "transformer,vae,audio_vae"
            ).split(",")
        ),
        adaln_host_cache=os.environ.get("OMNI_H3_ADALN_CACHE", "0") == "1",
        block_stream_blocks_per_group=int(
            os.environ.get("OMNI_H3_BLOCK_STREAM", "0")
        ),
        stream_text_encoder=os.environ.get("OMNI_H3_STREAM_TEXT_ENCODER", "0")
        == "1",
    )
    prompts = [payload["prompt"], *trace_prompts("vidprom", max(jobs - 1, 0))]
    latencies: list[float] = []
    parity = {"video": False, "audio": False}
    for index, prompt in enumerate(prompts):
        start = time.perf_counter()
        result = runner.generate(
            prompt,
            image=image,
            seed=payload["seed"],
            num_inference_steps=payload["steps"],
            resolution=payload["resolution"],
            num_frames=payload["frames"],
        )
        latencies.append(time.perf_counter() - start)
        if index == 0:
            parity["video"] = _video_ok(result.latents, payload)
            parity["audio"] = _audio_ok(result.audio_latents, payload)
        print(f"  baseline job{index}: {latencies[-1]:7.2f}s")
    print(
        f"  baseline job0 parity vs goldens: video={parity['video']} "
        f"audio={parity['audio']}"
    )
    if not (parity["video"] and parity["audio"]):
        raise SystemExit("baseline parity gate FAILED vs goldens")
    # The cold first job carries the one-time text-encoder warmup; use
    # the warm jobs (if any) for the serial rate.
    warm = latencies[1:] or latencies
    per_job = statistics.median(warm)
    summary = {
        "topology": "1",
        "gpus": 1,
        "replicas": 1,
        "jobs": len(prompts),
        "per_job_s": per_job,
        "jobs_per_hour": 3600.0 / per_job,
        "parity_video_bitwise": parity["video"],
        "parity_audio_bitwise": parity["audio"],
    }
    print(
        f"  baseline: {per_job:6.2f}s/job warm -> "
        f"{summary['jobs_per_hour']:7.1f} jobs/hour (1 GPU, streamed)"
    )
    return summary


def _drive_replica(
    pipeline: Any,
    specs: list[tuple[str, dict[str, Any]]],
    order: list[tuple[str, float]],
    errors: list[BaseException],
    barrier: threading.Barrier,
) -> None:
    """Submit one replica's slice while a thread drains its results.

    The role queues cap in-flight jobs, so ``submit`` blocks past the
    cap; a dedicated collector drains concurrently (submitting the
    whole slice before collecting would deadlock — see RolePipeline).
    """

    local_error: list[BaseException] = []

    def collector() -> None:
        try:
            for _ in range(len(specs)):
                result = pipeline.collect(timeout=3600)
                order.append((result.job_id, time.perf_counter()))
        except BaseException as error:  # noqa: BLE001
            local_error.append(error)

    thread = threading.Thread(target=collector)
    barrier.wait()
    thread.start()
    try:
        for job_id, request in specs:
            pipeline.submit(job_id, request)
        thread.join(timeout=3600 * 2)
        if thread.is_alive():
            errors.append(RuntimeError("replica collector did not finish"))
        elif local_error:
            errors.append(local_error[0])
    except BaseException as error:  # noqa: BLE001
        errors.append(error)


def run_grid(
    topology: str,
    jobs: int,
    payload: dict[str, Any],
    image: Any,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Warm up K replicas, gate parity, then time a VidProM trace."""

    import os

    from omni_infinity.serve.role_pipeline import RolePipeline

    maps = topology_device_maps(topology)
    checkpoint = os.environ["OMNI_H3_CHECKPOINT"]
    store = os.environ.get("OMNI_H3_STORE")
    pipelines: list[Any] = []
    try:
        spawn0 = time.perf_counter()
        for device_map in maps:
            pipelines.append(
                RolePipeline(checkpoint, device_map, store_dir=store)
            )
        warm_threads = [
            threading.Thread(target=pipeline.warmup) for pipeline in pipelines
        ]
        for thread in warm_threads:
            thread.start()
        for thread in warm_threads:
            thread.join()
        print(
            f"  {len(pipelines)} replica(s) warm in "
            f"{time.perf_counter() - spawn0:6.1f}s"
        )

        pipelines[0].submit(
            "parity",
            _request(payload, image, payload["prompt"], height, width),
        )
        parity_result = pipelines[0].collect(timeout=3600)
        outputs = parity_result.meta["outputs"]
        video_ok = _video_ok(outputs["latents"], payload)
        audio_ok = _audio_ok(outputs["audio_latents"], payload)
        print(f"  parity vs goldens: video={video_ok} audio={audio_ok}")
        if not (video_ok and audio_ok):
            raise SystemExit("role-pipeline parity gate FAILED vs goldens")

        prompts = trace_prompts("vidprom", jobs)
        per_replica: list[list[tuple[str, dict[str, Any]]]] = [
            [] for _ in pipelines
        ]
        for index, prompt in enumerate(prompts):
            request = _request(payload, image, prompt, height, width)
            per_replica[index % len(pipelines)].append((f"job{index}", request))
        orders: list[list[tuple[str, float]]] = [[] for _ in pipelines]
        errors: list[BaseException] = []
        barrier = threading.Barrier(len(pipelines))
        start = time.perf_counter()
        threads = [
            threading.Thread(
                target=_drive_replica,
                args=(
                    pipeline,
                    per_replica[index],
                    orders[index],
                    errors,
                    barrier,
                ),
            )
            for index, pipeline in enumerate(pipelines)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]

        finishes = [ts for order in orders for _, ts in order]
        if len(finishes) != jobs:
            raise SystemExit(f"collected {len(finishes)} of {jobs} jobs")
        wall = max(finishes) - start
        wall_rate = jobs / wall * 3600.0
        spacings = [
            _steady_spacing([ts for _, ts in order], start)
            for order in orders
            if order
        ]
        steady_rate = sum(3600.0 / spacing for spacing in spacings)
        gpus = len({dev for device_map in maps for dev in device_map.values()})
        summary = {
            "topology": topology,
            "gpus": gpus,
            "replicas": len(pipelines),
            "jobs": jobs,
            "wall_s": wall,
            "wall_jobs_per_hour": wall_rate,
            "per_replica_spacing_s": spacings,
            "steady_jobs_per_hour": steady_rate,
            "parity_video_bitwise": video_ok,
            "parity_audio_bitwise": audio_ok,
        }
        print(
            f"  {jobs} jobs in {wall:6.2f}s -> wall {wall_rate:7.1f} "
            f"jobs/hour; steady {steady_rate:7.1f} jobs/hour "
            f"({len(pipelines)} replica(s), {gpus} GPU(s))"
        )
        return summary
    finally:
        for pipeline in pipelines:
            try:
                pipeline.shutdown()
            except BaseException as error:  # noqa: BLE001
                print(f"  shutdown warning: {error}")


def main() -> int:
    from PIL import Image

    from omni_infinity.runner import resolve_resolution

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology",
        default="3",
        choices=("1", "2", "3", "4", "6"),
        help="1 = single-GPU streamed baseline; 2/3/4/6 = role replicas",
    )
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    payload = torch.load(GOLDENS, weights_only=False)
    assert payload["torch_version"] == torch.__version__, "stack mismatch"
    height, width = resolve_resolution(payload["resolution"])
    image = Image.open(REFERENCE).convert("RGB")

    print(f"topology={args.topology} jobs={args.jobs}")
    if args.topology == "1":
        summary = run_baseline(args.jobs, payload, image)
    else:
        summary = run_grid(
            args.topology, args.jobs, payload, image, height, width
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
