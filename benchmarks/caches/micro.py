# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmarks for the isolated cost of each issue #24 cache level.

Everything runs on synthetic tensors, without checkpoints or network access.
The serving code paths for C1 keys and tiers, C3 keys, and the C5 decision
wrapper are timed on CPU in CI and on CUDA when available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from benchmarks.caches.contract import MICRO_REPS, WARMUP
from omni_infinity.caches._tensor_tree import update_hash_for_value
from omni_infinity.caches.condition import ConditionCache, condition_key
from omni_infinity.caches.denoise import DenoiseCacheConfig, denoise_step_cache

_EMBED_DIM = 5120


def _c3_call_key(args: tuple, kwargs: dict) -> bytes:
    """Apply the same key recipe as the C3 vision forward wrapper."""
    hasher = hashlib.sha256()
    update_hash_for_value(hasher, "omni-vision-v1")
    update_hash_for_value(hasher, args)
    for key in sorted(kwargs):
        update_hash_for_value(hasher, key)
        update_hash_for_value(hasher, kwargs[key])
    return hasher.digest()


def _time_ms(fn: Callable[[], Any], reps: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples = []
    use_cuda = torch.cuda.is_available()
    for _ in range(reps):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _row(bench: str, samples: list[float], reps: int, **extra) -> dict:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "suite": "micro",
        "bench": bench,
        "reps": reps,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "mean_ms": statistics.fmean(ordered),
        "extra": extra,
    }


def run_micro(
    *,
    out_path: Path,
    reps: int = MICRO_REPS,
    warmup: int = WARMUP,
    embed_rows: int = 2048,
    image_bytes: int = 1 << 20,
) -> list[dict]:
    """Run all synthetic cache microbenchmarks and write their JSON rows."""
    rows: list[dict] = []
    prompt = "p" * 512
    blob = bytes(image_bytes)

    def make_condition_key():
        return condition_key(
            "ns",
            prompt,
            (blob,),
            height=256,
            width=256,
            num_frames=120,
        )

    rows.append(
        _row(
            "c1_condition_key",
            _time_ms(make_condition_key, reps, warmup),
            reps,
            image_bytes=image_bytes,
        )
    )

    entry = {
        "prompt_embeds": torch.randn(
            embed_rows, _EMBED_DIM, dtype=torch.bfloat16
        )
    }
    disk_dir = out_path.parent / "c1-disk"
    memory_cache = ConditionCache(max_entries=4)
    key = make_condition_key()
    rows.append(
        _row(
            "c1_mem_put",
            _time_ms(lambda: memory_cache.put(key, dict(entry)), reps, warmup),
            reps,
            embed_rows=embed_rows,
        )
    )
    rows.append(
        _row(
            "c1_mem_get",
            _time_ms(lambda: memory_cache.get(key), reps, warmup),
            reps,
            embed_rows=embed_rows,
        )
    )

    disk_cache = ConditionCache(max_entries=4, cache_dir=disk_dir)
    disk_cache.put(key, dict(entry))

    def cold_get():
        cold = ConditionCache(max_entries=4, cache_dir=disk_dir)
        return cold.get(key)

    rows.append(
        _row(
            "c1_disk_get_cold",
            _time_ms(cold_get, reps, warmup),
            reps,
            embed_rows=embed_rows,
        )
    )

    patches = torch.randn(1024, 1176)
    rows.append(
        _row(
            "c3_call_key",
            _time_ms(lambda: _c3_call_key((patches,), {}), reps, warmup),
            reps,
            patch_rows=1024,
        )
    )

    tiny = _TinyTransformer()
    config = DenoiseCacheConfig(
        coefficients=(1.0,), threshold=1e6, mode="output"
    )
    steps = 4
    hidden_states = torch.randn(8, 16)

    def one_generation():
        with denoise_step_cache(tiny, config, steps) as stats:
            for _ in range(steps):
                tiny(hidden_states=hidden_states)
        return stats

    stats = one_generation()
    rows.append(
        _row(
            "c5_decision_overhead",
            _time_ms(one_generation, reps, warmup),
            reps,
            computed=stats.computed,
            skipped=stats.skipped,
            steps=steps,
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    return rows


class _TinyTransformer(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2.0


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/cache-bench")
    )
    parser.add_argument("--reps", type=int, default=MICRO_REPS)
    parser.add_argument("--embed-rows", type=int, default=2048)
    parser.add_argument("--image-bytes", type=int, default=1 << 20)
    args = parser.parse_args()
    rows = run_micro(
        out_path=args.results_dir / "micro.json",
        reps=args.reps,
        embed_rows=args.embed_rows,
        image_bytes=args.image_bytes,
    )
    for row in rows:
        print(
            f"{row['bench']:24s} p50={row['p50_ms']:.3f} ms "
            f"p95={row['p95_ms']:.3f} ms [{row['device']}]"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
