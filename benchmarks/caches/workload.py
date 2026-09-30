# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Seeded prompt traces for the issue #24 cache benchmarks.

The committed fixture (``fixtures/prompts.json``) is the only source the
tests and default benchmark runs read — CI never touches the network.
``--refresh`` regenerates the fixture from the SOTA HuggingFace prompt
suites (VidProM real-user prompts; the VBench standard suite) and
requires the ``bench`` extra (``pip install -e '.[bench]'``).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from benchmarks.caches.contract import PROMPT_FIXTURE

VIDPROM_HF_ID = "WenhaoWang/VidProM"
VBENCH_PROMPT_URL = (
    "https://raw.githubusercontent.com/Vchitect/VBench/master/"
    "prompts/all_dimension.txt"
)


@dataclass(frozen=True)
class TraceItem:
    """One request in a serving trace."""

    index: int
    prompt: str
    expected_hit: bool


def load_prompts(source: str = "fixture", pool: str = "vidprom") -> list[str]:
    """Read prompts from one pool in the committed offline fixture."""
    if source == "fixture":
        payload = json.loads(PROMPT_FIXTURE.read_text())
        return list(payload[pool])
    if source in ("vidprom", "vbench"):
        payload = json.loads(PROMPT_FIXTURE.read_text())
        return list(payload[source])
    raise ValueError(
        f"unknown prompt source {source!r}; expected fixture|vidprom|vbench"
    )


def build_trace(
    *, seed: int, length: int, repeat_ratio: float, pool: str = "vidprom"
) -> list[TraceItem]:
    """Build a deterministic trace with a controlled repeat fraction."""
    if not 0.0 <= repeat_ratio <= 1.0:
        raise ValueError("repeat_ratio must be within [0, 1]")
    prompts = list(dict.fromkeys(load_prompts("fixture", pool=pool)))
    repeats = int(length * repeat_ratio)
    uniques = length - repeats
    if uniques > len(prompts):
        raise ValueError(
            f"trace needs {uniques} unique prompts; pool has {len(prompts)}"
        )
    rng = random.Random(seed)
    unique_prompts = rng.sample(prompts, uniques)
    positions = list(range(1, length))
    rng.shuffle(positions)
    repeat_positions = sorted(positions[:repeats])
    trace: list[TraceItem] = []
    unique_iter = iter(unique_prompts)
    for index in range(length):
        if index in repeat_positions and trace:
            source = rng.choice(trace)
            trace.append(TraceItem(index, source.prompt, True))
        else:
            trace.append(TraceItem(index, next(unique_iter), False))
    return trace


def _refresh(source: str, count: int, seed: int) -> None:  # pragma: no cover
    """Regenerate fixture pools from HuggingFace/GitHub."""
    payload = json.loads(PROMPT_FIXTURE.read_text())
    rng = random.Random(seed)
    if source in ("vidprom", "all"):
        from datasets import load_dataset

        stream = load_dataset(VIDPROM_HF_ID, split="train", streaming=True)
        reservoir: list[str] = []
        for row_index, row in enumerate(stream):
            if row_index >= 20_000:
                break
            prompt = (row.get("prompt") or "").strip()
            if not 10 <= len(prompt) <= 300:
                continue
            if len(reservoir) < count:
                reservoir.append(prompt)
            else:
                slot = rng.randint(0, row_index)
                if slot < count:
                    reservoir[slot] = prompt
        payload["vidprom"] = reservoir
    if source in ("vbench", "all"):
        from urllib.request import urlopen

        lines = urlopen(VBENCH_PROMPT_URL, timeout=30).read().decode()
        prompts = [line.strip() for line in lines.splitlines() if line.strip()]
        payload["vbench"] = rng.sample(prompts, min(16, len(prompts)))
    PROMPT_FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--source", choices=("vidprom", "vbench", "all"), default="all"
    )
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.refresh:
        _refresh(args.source, args.count, args.seed)
        return 0
    parser.error("nothing to do; pass --refresh")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
