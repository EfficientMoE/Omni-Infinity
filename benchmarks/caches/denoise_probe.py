# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""C5 calibration probe (docs/opt_in_caches.md Phase 4 gate).

Wraps the transformer ``forward`` for one generation and records, per
call: the relative-L1 distance between consecutive step inputs (the
TeaCache decision signal) and between consecutive step outputs (the
truth the polynomial must predict). Fit coefficients offline (e.g.
``numpy.polyfit`` on input vs output distances) and pass them to
``--c5-coefficients`` in ``benchmarks.caches.ablation``.

GPU usage:
    OMNI_CHECKPOINT=... python -m benchmarks.caches.denoise_probe \
        --results-dir results/cache-bench
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path


def _rel_l1(current, previous) -> float | None:
    if previous is None:
        return None
    scale = previous.abs().mean()
    if float(scale) == 0.0:
        return None
    return float((current.to(previous.dtype) - previous).abs().mean() / scale)


@contextlib.contextmanager
def probe_forward(module, signal_name: str = "hidden_states"):
    """Yield a list that fills with one record per forward call."""
    records: list[dict] = []
    state = {"input": None, "output": None}
    original = module.forward

    def probed(*args, **kwargs):
        signal = kwargs.get(signal_name, args[0] if args else None)
        output = original(*args, **kwargs)
        from omni_infinity.caches._tensor_tree import tree_tensors

        leaf = tree_tensors(output)[0]
        records.append(
            {
                "call": len(records),
                "input_rel_l1": _rel_l1(signal, state["input"]),
                "output_rel_l1": _rel_l1(leaf, state["output"]),
            }
        )
        state["input"] = signal.detach()
        state["output"] = leaf.detach()
        return output

    module.forward = probed
    try:
        yield records
    finally:
        del module.__dict__["forward"]


def main() -> int:  # pragma: no cover — needs weights
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/cache-bench")
    )
    args = parser.parse_args()
    if not os.environ.get("OMNI_CHECKPOINT"):
        print(json.dumps({"skip": "weights-absent"}))
        return 0
    from benchmarks.caches.contract import (
        FRAMES,
        PROMPT,
        RESOLUTION,
        SEED,
        STEPS,
    )
    from omni_infinity.runner import ReferenceRunner

    runner = ReferenceRunner.from_pretrained(
        os.environ["OMNI_CHECKPOINT"],
        workflow="fl2va",
        offload=True,
        store_dir=os.environ.get("OMNI_STORE_DIR"),
        block_stream_blocks_per_group=1,
        stream_text_encoder=True,
    )
    transformer = runner.pipeline.components[runner.transformer_component]
    with probe_forward(transformer) as records:
        runner.generate(
            PROMPT,
            seed=SEED,
            num_inference_steps=STEPS,
            resolution=RESOLUTION,
            num_frames=FRAMES,
        )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "denoise_probe.json"
    out.write_text(json.dumps({"records": records}, indent=2) + "\n")
    print(f"wrote {out} ({len(records)} calls)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
