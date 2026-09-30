# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end serving trace for the condition cache (issue #24 C1).

Replays a seeded VidProM-derived trace against a running job server.
Start the server with the condition cache in its immutable profile —
requests must repeat the exact ordered optimization list or the server
answers HTTP 409::

    OMNI_MODEL_ARCH=h3-dense \
    OMNI_OPTIMIZATIONS=adaln-host-cache,block-stream,\
    text-encoder-stream,condition-cache \
    OMNI_CONDITION_CACHE_DIR=./cache-c1 \
    OMNI_CHECKPOINT=... python -m omni_infinity.serve

Repeated prompts are the exact-hit condition, so repeated-request
latency directly measures C1's cross-request contribution; restart the
server between two runs of this script against the same cache dir to
measure the disk tier.

The server is not modified and exposes no cache counters; expected hits
are derived from the trace (a repeat of an earlier prompt in the same
run is an expected hit).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from benchmarks.caches.contract import (
    FIELDS,
    FRAMES,
    RESOLUTION,
    SEED,
    STEPS,
    verdict_for,
)
from benchmarks.caches.workload import TraceItem, build_trace


class JobClient:  # pragma: no cover — exercised against a live server
    """Minimal polling client for the /v1/jobs API."""

    def __init__(
        self,
        base_url: str,
        optimizations: list[str],
        arch: str,
        timeout_s: float = 1800.0,
    ):
        import httpx  # serve extra

        self._client = httpx.Client(base_url=base_url, timeout=60.0)
        self._optimizations = optimizations
        self._arch = arch
        self._timeout_s = timeout_s

    def submit_and_wait(self, prompt: str) -> float:
        started = time.perf_counter()
        response = self._client.post(
            "/v1/jobs",
            json={
                "type": "fl2va",
                "prompt": prompt,
                "model_arch": self._arch,
                "optimizations": self._optimizations,
                "seed": SEED,
                "num_inference_steps": STEPS,
                "resolution": RESOLUTION,
                "num_frames": FRAMES,
            },
        )
        response.raise_for_status()
        job_id = response.json()["id"]
        deadline = started + self._timeout_s
        while time.perf_counter() < deadline:
            status = self._client.get(f"/v1/jobs/{job_id}").json()
            if status["status"] in ("succeeded", "failed", "cancelled"):
                if status["status"] != "succeeded":
                    raise RuntimeError(f"job {job_id}: {status['status']}")
                return (time.perf_counter() - started) * 1000.0
            time.sleep(1.0)
        raise TimeoutError(f"job {job_id} did not finish")


def replay(
    trace: list[TraceItem],
    client,
    *,
    arch: str,
    cache_config: str,
    repeat_ratio: float,
) -> list[dict]:
    rows = []
    for item in trace:
        elapsed_ms = client.submit_and_wait(item.prompt)
        row = {field: "" for field in FIELDS}
        row.update(
            suite="serve-trace",
            arch=arch,
            cache_config=cache_config,
            dataset="vidprom",
            trace_len=len(trace),
            repeat_ratio=repeat_ratio,
            rep=item.index,
            phase="warm" if item.expected_hit else "cold",
            e2e_ms=elapsed_ms,
            expected_hits=1 if item.expected_hit else 0,
            notes="",
        )
        row["verdict"] = verdict_for(row)
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    repeat = [r["e2e_ms"] for r in rows if r["expected_hits"] == 1]
    unique = [r["e2e_ms"] for r in rows if r["expected_hits"] == 0]
    return {
        "requests": len(rows),
        "expected_hits": len(repeat),
        "repeat_p50_ms": statistics.median(repeat) if repeat else None,
        "unique_p50_ms": statistics.median(unique) if unique else None,
    }


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--arch", default="h3-dense")
    parser.add_argument(
        "--optimizations",
        default="adaln-host-cache,block-stream,"
        "text-encoder-stream,condition-cache",
        help="must repeat the server's exact ordered list",
    )
    parser.add_argument("--trace-len", type=int, default=16)
    parser.add_argument("--repeat-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/cache-bench")
    )
    args = parser.parse_args()
    trace = build_trace(
        seed=args.seed,
        length=args.trace_len,
        repeat_ratio=args.repeat_ratio,
    )
    client = JobClient(
        args.base_url, args.optimizations.split(","), args.arch
    )
    rows = replay(
        trace,
        client,
        arch=args.arch,
        cache_config="c1",
        repeat_ratio=args.repeat_ratio,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(
        args.results_dir / "serve_trace.csv", "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summarize(rows), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
