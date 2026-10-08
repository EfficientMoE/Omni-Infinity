# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""P7 bench: cross-process role pipeline throughput + goldens parity.

Spawns the encoder/denoiser/decoder role processes (one GPU each, fully
resident), submits N identical jobs back-to-back so stages overlap
across jobs, verifies job 0 bitwise against the committed goldens, and
reports per-job walltime and jobs/hour vs the serialized sum.

Run:
  CUDA_VISIBLE_DEVICES=0,1,2 HF_HOME=... HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 OMNI_H3_CHECKPOINT=<snapshot> \
  OMNI_H3_STORE=<store> python benchmarks/role_pipeline_bench.py [N]
"""

import os
import sys
import threading
import time
from pathlib import Path

import torch

GOLDENS = Path("tests/fixtures/goldens/fl2va_goldens.pt")
REFERENCE = Path("tests/fixtures/ref.png")


def main():
    from PIL import Image

    from omni_infinity.runner import resolve_resolution
    from omni_infinity.serve.role_pipeline import RolePipeline
    from omni_infinity.serve.roles import Role

    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    payload = torch.load(GOLDENS, weights_only=False)
    assert payload["torch_version"] == torch.__version__, "stack mismatch"
    height, width = resolve_resolution(payload["resolution"])
    image = Image.open(REFERENCE).convert("RGB")

    device_map = {
        Role.ENCODER: os.environ.get("OMNI_ENC_DEVICE", "cuda:0"),
        Role.DENOISER: os.environ.get("OMNI_DEN_DEVICE", "cuda:1"),
        Role.DECODER: os.environ.get("OMNI_DEC_DEVICE", "cuda:2"),
    }
    print("spawning role processes (parallel component loads)...")
    t0 = time.perf_counter()
    pipeline = RolePipeline(
        os.environ["OMNI_H3_CHECKPOINT"],
        device_map,
        store_dir=os.environ.get("OMNI_H3_STORE"),
    )
    try:
        results = {}
        completion_order = []

        collector_error = []

        def collector():
            try:
                for _ in range(jobs):
                    result = pipeline.collect(timeout=3600)
                    results[result.job_id] = result.meta["outputs"]
                    completion_order.append(
                        (result.job_id, time.perf_counter())
                    )
            except BaseException as error:
                collector_error.append(error)

        submit_t = time.perf_counter()
        collect_thread = threading.Thread(target=collector)
        collect_thread.start()
        for index in range(jobs):
            pipeline.submit(
                f"job{index}",
                {
                    "prompt": payload["prompt"],
                    "image": image,
                    "height": height,
                    "width": width,
                    "num_frames": payload["frames"],
                    "num_inference_steps": payload["steps"],
                    "generator": torch.Generator("cpu").manual_seed(
                        payload["seed"]
                    ),
                    "output_type": "np",
                },
            )
        collect_thread.join(timeout=3600 * 2)
        if collect_thread.is_alive():
            raise SystemExit("collector did not finish")
        if collector_error:
            raise collector_error[0]
        if len(completion_order) != jobs:
            raise SystemExit(
                f"collected {len(completion_order)} of {jobs} jobs"
            )
        total = time.perf_counter() - submit_t
        print(
            f"  spawn-to-submit={submit_t - t0:7.1f}s; {jobs} jobs in "
            f"{total:6.2f}s cold (includes lazy worker model loads)"
        )
        previous = submit_t
        deltas = []
        for job_id, finished in completion_order:
            delta = finished - previous
            deltas.append(delta)
            print(f"  {job_id}: +{delta:6.2f}s")
            previous = finished
        tail = deltas[len(deltas) // 2 :]
        if tail:
            spacing = sum(tail) / len(tail)
            print(
                f"  steady-state spacing (last {len(tail)} jobs): "
                f"{spacing:5.2f}s/job ({3600 / spacing:7.1f} jobs/hour)"
            )
    finally:
        pipeline.shutdown()

    outputs = results["job0"]
    bitwise = torch.equal(outputs["latents"], payload["latents"])
    if payload["audio_latents"] is None:
        audio_ok = outputs["audio_latents"] is None
    else:
        audio_ok = outputs["audio_latents"] is not None and torch.equal(
            outputs["audio_latents"], payload["audio_latents"]
        )
    print(f"  job0 video latents bitwise vs goldens: {bitwise}")
    print(f"  job0 audio latents bitwise vs goldens: {audio_ok}")
    if not (bitwise and audio_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
