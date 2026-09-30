# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Generation-local denoise-step caching."""

from __future__ import annotations

import importlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType

import torch

from omni_infinity.caches._tensor_tree import tree_map, tree_tensors


@dataclass(frozen=True)
class DenoiseCacheConfig:
    """Configuration requiring model-specific calibrated coefficients."""

    coefficients: tuple[float, ...]
    threshold: float
    mode: str = "output"
    signal_name: str = "hidden_states"
    io_names: tuple[str, ...] = ("hidden_states",)
    calls_per_step: int = 1
    warmup_steps: int = 1
    final_steps: int = 1

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError(
                "calibrated coefficients are required for MiniMax-H3"
            )
        if not all(math.isfinite(value) for value in self.coefficients):
            raise ValueError("calibrated coefficients must be finite")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.mode not in {"output", "residual"}:
            raise ValueError("mode must be output or residual")
        if self.warmup_steps < 1 or self.final_steps < 1:
            raise ValueError("the first and last steps must always compute")
        if self.calls_per_step < 1:
            raise ValueError("calls_per_step must be positive")


@dataclass
class DenoiseCacheStats:
    """Counts transformer evaluations and cached replays."""

    computed: int = 0
    skipped: int = 0


@dataclass
class _Slot:
    signal: object | None = None
    output: object | None = None
    inputs: object | None = None
    residual: object | None = None


def _snapshot(value):
    return tree_map(lambda tensor: tensor.detach().clone(), value)


def _relative_l1(current, previous) -> float:
    current_tensors = tree_tensors(current)
    previous_tensors = tree_tensors(previous)
    if len(current_tensors) != len(previous_tensors):
        return float("inf")
    if not current_tensors:
        return 0.0

    numerator = 0.0
    denominator = 0.0
    for current_tensor, previous_tensor in zip(
        current_tensors, previous_tensors, strict=True
    ):
        if current_tensor.shape != previous_tensor.shape:
            return float("inf")
        numerator += (current_tensor - previous_tensor).abs().mean().item()
        denominator += previous_tensor.abs().mean().item()
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return numerator / denominator


def _polynomial(coefficients: tuple[float, ...], value: float) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


def _resolve_input(name: str, args: tuple, kwargs: dict, io_names: tuple):
    if name in kwargs:
        return kwargs[name]
    try:
        index = io_names.index(name)
    except ValueError:
        index = 0
    if index < len(args):
        return args[index]
    if args:
        return args[0]
    raise ValueError(f"could not resolve input {name!r}")


def _residual_inputs(args: tuple, kwargs: dict, io_names: tuple):
    values = tuple(
        _resolve_input(name, args, kwargs, io_names) for name in io_names
    )
    return values[0] if len(values) == 1 else values


def _tree_binary(left, right, operation):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            raise ValueError(
                f"residual shape mismatch: {left.shape} != {right.shape}"
            )
        return operation(left, right)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("residual shape mismatch")
        return [
            _tree_binary(a, b, operation)
            for a, b in zip(left, right, strict=True)
        ]
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise ValueError("residual shape mismatch")
        return tuple(
            _tree_binary(a, b, operation)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError("residual shape mismatch")
        return {
            key: _tree_binary(left[key], right[key], operation) for key in left
        }
    raise ValueError("residual shape mismatch")


def _subtract(left, right):
    return _tree_binary(left, right, lambda a, b: a - b)


def _add(left, right):
    return _tree_binary(left, right, lambda a, b: a + b)


@contextmanager
def denoise_step_cache(transformer, config, total_steps: int):
    """Cache denoise outputs for the duration of one generation."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if getattr(transformer, "_omni_denoise_cache_active", False):
        raise RuntimeError("denoise-step cache is already enabled")

    had_instance_forward = "forward" in transformer.__dict__
    instance_forward = transformer.__dict__.get("forward")
    original_forward = transformer.forward
    stats = DenoiseCacheStats()
    slots = [_Slot() for _ in range(config.calls_per_step)]
    calls = 0

    def cached_forward(_module, *args, **kwargs):
        nonlocal calls
        step = calls // config.calls_per_step
        slot_index = calls % config.calls_per_step
        calls += 1
        slot = slots[slot_index]
        signal = _resolve_input(
            config.signal_name, args, kwargs, config.io_names
        )
        boundary = step < config.warmup_steps or (
            total_steps - config.final_steps <= step < total_steps
        )
        distance = (
            float("inf")
            if slot.signal is None
            else _relative_l1(signal, slot.signal)
        )
        compute = (
            boundary
            or slot.output is None
            or not math.isfinite(distance)
            or (_polynomial(config.coefficients, distance) > config.threshold)
        )
        slot.signal = _snapshot(signal)

        if compute:
            output = original_forward(*args, **kwargs)
            stats.computed += 1
            slot.output = output
            if config.mode == "residual":
                slot.inputs = _residual_inputs(args, kwargs, config.io_names)
                try:
                    slot.residual = _subtract(output, slot.inputs)
                except ValueError:
                    slot.residual = None
            return output

        stats.skipped += 1
        if config.mode == "output":
            return slot.output
        current_inputs = _residual_inputs(args, kwargs, config.io_names)
        if slot.residual is None:
            slot.residual = _subtract(slot.output, slot.inputs)
        return _add(current_inputs, slot.residual)

    transformer._omni_denoise_cache_active = True
    transformer.forward = MethodType(cached_forward, transformer)
    try:
        yield stats
    finally:
        if had_instance_forward:
            transformer.forward = instance_forward
        else:
            delattr(transformer, "forward")
        delattr(transformer, "_omni_denoise_cache_active")


def enable_cache_dit(target, **kwargs):
    """Enable the optional cache-dit integration or fail closed."""
    try:
        cache_dit = importlib.import_module("cache_dit")
        enable_cache = cache_dit.enable_cache
    except ModuleNotFoundError as exc:
        if exc.name != "cache_dit":
            raise
        raise ImportError("cache-dit is not installed") from None
    except AttributeError:
        raise ImportError("cache-dit is not installed") from None
    return enable_cache(target, **kwargs)
