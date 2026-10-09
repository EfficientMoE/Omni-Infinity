# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""C5 calibration probe for the deferred model-specific calibration gate.

Wraps transformer ``forward`` for one generation and records consecutive
input relative-L1 distances (the decision signal) and output relative-L1
distances (the truth to fit). Pass fitted coefficients to
``benchmarks.caches.ablation --c5-coefficients``.

GPU usage::

    OMNI_CHECKPOINT=... python -m benchmarks.caches.denoise_probe \
        --results-dir results/cache-bench
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from omni_infinity.caches.denoise import _fbcache_signal, _teacache_signal


@dataclass(frozen=True)
class _ProbeSignalConfig:
    signal_name: str
    io_names: tuple[str, ...]


_SIGNAL_CONFIG = _ProbeSignalConfig(
    signal_name="hidden_states",
    io_names=(
        "hidden_states",
        "audio_hidden_states",
        "encoder_hidden_states",
        "timestep",
        "timestep_indices",
        "token_tags",
        "position_ids",
        "video_indices",
        "audio_indices",
        "text_indices",
    ),
)


def _rel_l1(current, previous) -> float | None:
    if previous is None:
        return None
    scale = previous.abs().mean()
    if float(scale) == 0.0:
        return 0.0 if float(current.abs().mean()) == 0.0 else float("inf")
    return float((current.to(previous.dtype) - previous).abs().mean() / scale)


@contextlib.contextmanager
def probe_forward(module, signal_name: str = "hidden_states"):
    """Yield a list that fills with one record per forward call.

    Records through ``register_forward_hook`` rather than replacing
    ``forward``: the H3 offload machinery rebinds the transformer's
    instance ``forward`` on the first call, which silently evicts a
    replacement wrapper, while ``__call__`` hooks keep firing.
    """
    records: list[dict] = []
    state = {
        "input": None,
        "output": None,
        "teacache": None,
        "fbcache": None,
    }

    def indicator(name, function, hooked, args, kwargs):
        try:
            signal = function(hooked, args, kwargs, _SIGNAL_CONFIG)
            distance = _rel_l1(signal, state[name])
            state[name] = signal.detach().clone()
            return distance, None
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            return None, reason

    def record(hooked, args, kwargs, output):
        signal = kwargs.get(signal_name, args[0] if args else None)
        from omni_infinity.caches._tensor_tree import tree_tensors

        leaves = tree_tensors(output)
        leaf = leaves[0] if leaves else None
        teacache_rel_l1, teacache_error = indicator(
            "teacache", _teacache_signal, hooked, args, kwargs
        )
        fbcache_rel_l1, fbcache_error = indicator(
            "fbcache", _fbcache_signal, hooked, args, kwargs
        )
        records.append(
            {
                "call": len(records),
                "input_rel_l1": _rel_l1(signal, state["input"]),
                "output_rel_l1": (
                    _rel_l1(leaf, state["output"]) if leaf is not None else None
                ),
                "teacache_rel_l1": teacache_rel_l1,
                "fbcache_rel_l1": fbcache_rel_l1,
                "teacache_error": teacache_error,
                "fbcache_error": fbcache_error,
            }
        )
        state["input"] = signal.detach().clone()
        if leaf is not None:
            state["output"] = leaf.detach().clone()
        return output

    handle = module.register_forward_hook(record, with_kwargs=True)
    try:
        yield records
    finally:
        handle.remove()


def main() -> int:  # pragma: no cover
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
    from omni_infinity.runner import ReferenceRunner, _transformer_component

    store_dir = os.environ.get("OMNI_STORE_DIR")
    store_components = tuple(
        filter(
            None,
            os.environ.get("OMNI_STORE_COMPONENTS", "").split(","),
        )
    )
    runner = ReferenceRunner.from_pretrained(
        os.environ["OMNI_CHECKPOINT"],
        workflow="fl2va",
        offload=True,
        store_dir=store_dir,
        store_components=store_components or None,
        adaln_host_cache=bool(store_dir),
        block_stream_blocks_per_group=1,
        stream_text_encoder=True,
    )
    transformer = _transformer_component(
        runner.pipeline, runner.transformer_component
    )
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
