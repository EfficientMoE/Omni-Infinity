# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""C5 — denoise-step feature cache (issue #24). Approximate, opt-in.

The TeaCache decision rule, implemented as a wrapper on the transformer
``forward`` (the hook pattern vLLM-Omni and SGLang Diffusion use), kept
out of ``third_party/``:

- per step, the relative L1 distance between consecutive step inputs is
  rescaled by a model-specific polynomial and accumulated;
- while the accumulator stays under the threshold the evaluation is
  skipped and the cached result replayed; a computed step resets it;
- the first and last steps always compute.

Two replay modes, because what can be cached depends on the model's
forward contract:

- ``mode="residual"`` is TeaCache-faithful: cache ``output - input`` per
  leaf and replay ``input + residual``. It requires output leaves that
  are shape-compatible with the paired inputs (``io_names``).
- ``mode="output"`` replays the previous prediction unchanged
  (fixed-decision step reuse in the FORA family). This is the only mode
  applicable to MiniMax-H3 *from outside the forward*: H3 returns
  per-modality projections (``sample``/``audio_sample``) index-selected
  from the packed rows, so no input-shaped residual exists at this
  boundary — and the packed text rows live in the residual stream, which
  is exactly why issue #24 forbids publishing H3 text KV from a denoise
  step as a cross-request prefix.

Calibration is not optional. Neither TeaCache nor cache-dit publishes
MiniMax-H3 coefficients, and SGLang's own uncalibrated fallback
(Wan2.2) silently no-ops. This module therefore refuses to run without
explicit ``coefficients`` and ``threshold`` — there is no default that
could quietly regress quality. Any run with this cache enabled is off
the bitwise parity gates and needs its own quality gate (CLIP/SSIM/PSNR
style, as vLLM-Omni gates its diffusion caches — never ``rms_rel=0``).

Both serving stacks ship the `cache-dit`_ library as the alternative
integration surface; :func:`enable_cache_dit` delegates to it when it is
installed, rather than growing a third implementation here.

.. _cache-dit: https://cache-dit.readthedocs.io/
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any

import torch

from omni_infinity.caches._tensor_tree import tree_map, tree_tensors

_ACTIVE_FLAG = "_omni_denoise_cache_active"


@dataclasses.dataclass(frozen=True)
class DenoiseCacheConfig:
    """Calibrated parameters for the step cache.

    ``coefficients`` is the polynomial (highest degree first, as
    ``numpy.poly1d`` and the TeaCache releases order it) that rescales
    the raw relative-L1 input distance; ``threshold`` is the accumulated
    rescaled distance under which a step is skipped. Both must come from
    an H3-specific calibration sweep — no published table covers H3.
    """

    coefficients: tuple[float, ...]
    threshold: float
    mode: str = "output"
    signal_name: str = "hidden_states"
    io_names: tuple[str, ...] = ("hidden_states",)
    calls_per_step: int = 1
    warmup_steps: int = 1
    final_steps: int = 1

    def __post_init__(self):
        if not self.coefficients:
            raise ValueError(
                "DenoiseCacheConfig requires calibrated rescaling "
                "coefficients: no published TeaCache/cache-dit table "
                "covers MiniMax-H3, and shipping a default would either "
                "no-op (SGLang's uncalibrated Wan2.2 behavior) or "
                "silently regress quality (issue #24)"
            )
        if not self.threshold > 0:
            raise ValueError("threshold must be > 0")
        if self.mode not in ("output", "residual"):
            raise ValueError("mode must be 'output' or 'residual'")
        if self.calls_per_step < 1:
            raise ValueError("calls_per_step must be >= 1")
        if self.warmup_steps < 1 or self.final_steps < 1:
            raise ValueError(
                "the first and last steps always compute: warmup_steps "
                "and final_steps must be >= 1"
            )


@dataclasses.dataclass
class DenoiseCacheStats:
    computed: int = 0
    skipped: int = 0


def _poly(coefficients: tuple[float, ...], x: float) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * x + coefficient
    return result


class _Slot:
    """Per-CFG-branch state (SGLang keeps separate positive/negative
    residual slots; ``calls_per_step`` maps each intra-step call to its
    own slot)."""

    __slots__ = ("previous_signal", "cached", "accumulated")

    def __init__(self):
        self.previous_signal: torch.Tensor | None = None
        self.cached: Any = None
        self.accumulated = 0.0


def _resolve_input(name: str, position: int, args, kwargs):
    if name in kwargs:
        return kwargs[name]
    if position < len(args):
        return args[position]
    raise KeyError(
        f"transformer forward received no '{name}' input; adjust "
        "DenoiseCacheConfig.signal_name/io_names to the model's "
        "forward signature"
    )


def _replace_leaves(structure, leaves):
    """Rebuild *structure* with its tensor leaves swapped, in order."""
    iterator = iter(leaves)
    return tree_map(lambda _leaf: next(iterator), structure)


@contextlib.contextmanager
def denoise_step_cache(
    transformer,
    config: DenoiseCacheConfig,
    total_steps: int,
):
    """Wrap ``transformer.forward`` for one generation.

    State (accumulators, cached results) is per-generation, so the
    context must enclose exactly one pipeline call. Yields a
    :class:`DenoiseCacheStats` the caller can inspect afterwards.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if getattr(transformer, _ACTIVE_FLAG, False):
        raise RuntimeError("a denoise step cache is already enabled")

    stats = DenoiseCacheStats()
    slots = [_Slot() for _ in range(config.calls_per_step)]
    calls = 0
    signal_position = (
        config.io_names.index(config.signal_name)
        if config.signal_name in config.io_names
        else 0
    )
    had_instance_forward = "forward" in transformer.__dict__
    previous_forward = transformer.__dict__.get("forward")
    original = transformer.forward

    def cached_forward(*args, **kwargs):
        nonlocal calls
        step = calls // config.calls_per_step
        slot = slots[calls % config.calls_per_step]
        calls += 1

        signal = _resolve_input(
            config.signal_name, signal_position, args, kwargs
        )
        boundary = (
            step < config.warmup_steps
            or step >= total_steps - config.final_steps
        )
        skip = False
        if not boundary and slot.cached is not None:
            previous = slot.previous_signal
            scale = previous.abs().mean()
            if float(scale) != 0.0:
                distance = float(
                    (signal.to(previous.dtype) - previous).abs().mean() / scale
                )
                slot.accumulated += abs(_poly(config.coefficients, distance))
                skip = slot.accumulated < config.threshold
        slot.previous_signal = signal.detach()

        if skip:
            stats.skipped += 1
            if config.mode == "output":
                return slot.cached
            inputs = [
                _resolve_input(name, position, args, kwargs)
                for position, name in enumerate(config.io_names)
            ]
            return _replace_leaves(
                slot.cached,
                [
                    tensor + residual
                    for tensor, residual in zip(
                        inputs, tree_tensors(slot.cached)
                    )
                ],
            )

        output = original(*args, **kwargs)
        stats.computed += 1
        slot.accumulated = 0.0
        if config.mode == "output":
            slot.cached = output
        else:
            inputs = [
                _resolve_input(name, position, args, kwargs)
                for position, name in enumerate(config.io_names)
            ]
            leaves = tree_tensors(output)
            if len(leaves) != len(config.io_names):
                raise RuntimeError(
                    f"residual mode pairs {len(config.io_names)} "
                    f"io_names with {len(leaves)} output tensors; for "
                    "outputs that are not input-shaped (MiniMax-H3's "
                    "per-modality projections) use mode='output'"
                )
            for name, leaf, tensor in zip(config.io_names, leaves, inputs):
                if leaf.shape != tensor.shape:
                    raise RuntimeError(
                        f"residual for '{name}' needs matching shapes; "
                        f"got {tuple(leaf.shape)} vs "
                        f"{tuple(tensor.shape)}; use mode='output'"
                    )
            slot.cached = _replace_leaves(
                output,
                [
                    leaf.detach() - tensor
                    for leaf, tensor in zip(leaves, inputs)
                ],
            )
        return output

    transformer.forward = cached_forward
    setattr(transformer, _ACTIVE_FLAG, True)
    try:
        yield stats
    finally:
        if had_instance_forward:
            transformer.forward = previous_forward
        else:
            del transformer.__dict__["forward"]
        setattr(transformer, _ACTIVE_FLAG, False)


def enable_cache_dit(target, **kwargs):
    """Delegate to the `cache-dit` library (DBCache/TaylorSeer/SCM) —
    the integration surface SGLang and vLLM-Omni already use — instead
    of maintaining a third block-cache implementation here. The same
    calibration caveat applies: cache-dit publishes no MiniMax-H3
    profile, so quality-gate any configuration before serving with it.
    """
    try:
        import cache_dit
    except ImportError as exc:
        raise ImportError(
            "cache-dit is not installed; `pip install cache-dit` "
            "(https://cache-dit.readthedocs.io/)"
        ) from exc
    return cache_dit.enable_cache(target, **kwargs)
