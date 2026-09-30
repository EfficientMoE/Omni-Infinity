# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmarks: the isolated cost of each issue #24 cache level.

Everything here runs on synthetic tensors — no checkpoint, no network —
so the same code paths that serve real generations (``condition_key``,
``ConditionCache`` tiers, the vision call key, the C5 decision wrapper)
are timed on CPU in CI and on GPU when available.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from benchmarks.caches.contract import MICRO_REPS, WARMUP
from omni_infinity.caches.condition import ConditionCache, condition_key
from omni_infinity.caches.denoise import (
    DenoiseCacheConfig,
    denoise_step_cache,
)
from omni_infinity.caches.vision import _call_key

_EMBED_DIM = 5120  # layer-50 prompt_embeds width (docs/caches.md)


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
    rows: list[dict] = []
    prompt = "p" * 512
    blob = bytes(image_bytes)

    # C1: whole-condition key derivation (prompt + one image slot).
    rows.append(
        _row(
            "c1_condition_key",
            _time_ms(
                lambda: condition_key("ns", prompt, (blob,)), reps, warmup
            ),
            reps,
            image_bytes=image_bytes,
        )
    )

    # C1: memory tier put/get, then a disk-tier cold read.
    entry = {
        "prompt_embeds": torch.randn(
            embed_rows, _EMBED_DIM, dtype=torch.bfloat16
        )
    }
    disk_dir = out_path.parent / "c1-disk"
    cache = ConditionCache(max_entries=4, cache_dir=disk_dir)
    key = condition_key("ns", prompt, (blob,))
    rows.append(
        _row(
            "c1_mem_put",
            _time_ms(lambda: cache.put(key, dict(entry)), reps, warmup),
            reps,
            embed_rows=embed_rows,
        )
    )
    rows.append(
        _row(
            "c1_mem_get",
            _time_ms(lambda: cache.get(key), reps, warmup),
            reps,
            embed_rows=embed_rows,
        )
    )

    def _cold_get():
        cold = ConditionCache(max_entries=4, cache_dir=disk_dir)
        return cold.get(key)

    rows.append(
        _row(
            "c1_disk_get_cold",
            _time_ms(_cold_get, reps, warmup),
            reps,
            embed_rows=embed_rows,
        )
    )

    # C3: content hash of a vision-tower call (pixel patches tensor).
    patches = torch.randn(1024, 1176)
    rows.append(
        _row(
            "c3_call_key",
            _time_ms(lambda: _call_key((patches,), {}), reps, warmup),
            reps,
            patch_rows=1024,
        )
    )

    # C5: decision overhead of the wrapped forward on a tiny module.
    tiny = _TinyTransformer()
    config = DenoiseCacheConfig(
        coefficients=(1.0,), threshold=1e6, mode="output"
    )
    steps = 4
    x = torch.randn(8, 16)

    def _one_generation():
        with denoise_step_cache(tiny, config, steps) as stats:
            for _ in range(steps):
                tiny(hidden_states=x)
        return stats

    stats = _one_generation()  # counters from one representative run
    rows.append(
        _row(
            "c5_decision_overhead",
            _time_ms(_one_generation, reps, warmup),
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
