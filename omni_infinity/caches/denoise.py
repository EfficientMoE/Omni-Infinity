# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Generation-local denoise-step caching."""

from __future__ import annotations

import importlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

import torch

from omni_infinity.caches._tensor_tree import tree_map, tree_tensors


@dataclass(frozen=True)
class DenoiseCacheConfig:
    """Configuration requiring model-specific calibrated coefficients.

    ``indicator="raw"`` preserves the original hidden-state comparison.
    ``indicator="teacache"`` compares the first H3 block's timestep-modulated
    video input. ``indicator="fbcache"`` faithfully runs the first block over
    the packed projected inputs and compares its output-minus-input residual.
    ``accumulate`` adds polynomial-rescaled distances until the threshold is
    reached and resets the total whenever the model computes. FBCache normally
    uses no accumulator; with ``accumulate=False`` and identity coefficients
    ``(1, 0)``, this is its plain per-step threshold rule. ``approximator``
    selects reuse or first-order finite-difference extrapolation without
    changing the compute/skip decision.
    """

    coefficients: tuple[float, ...]
    threshold: float
    mode: str = "output"
    approximator: str = "reuse"
    indicator: str = "raw"
    accumulate: bool = False
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
        if self.approximator not in {"reuse", "taylor1"}:
            raise ValueError("approximator must be reuse or taylor1")
        if self.indicator not in {"raw", "teacache", "fbcache"}:
            raise ValueError("indicator must be raw, teacache, or fbcache")
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
    accumulated: float = 0.0
    calls: int = 0
    computed_call: int | None = None
    value: object | None = None
    derivative: object | None = None


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


def _disable_compile(function):
    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None)
    return function if disable is None else disable(function)


@_disable_compile
def _cache_decision(
    *,
    boundary: bool,
    has_output: bool,
    distance: float,
    coefficients: tuple[float, ...],
    threshold: float,
    accumulate: bool,
    accumulated: float,
) -> tuple[bool, float]:
    """Make the skip decision in plain Python, outside compiled model graphs."""
    if boundary or not has_output or not math.isfinite(distance):
        return True, 0.0
    rescaled = _polynomial(coefficients, distance)
    if not accumulate:
        return rescaled > threshold, 0.0
    accumulated += rescaled
    compute = not accumulated < threshold
    return compute, 0.0 if compute else accumulated


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


def _module_target(module, fallback_device, fallback_dtype):
    """Device and dtype of a module's weights, or the activation's own.

    Block streaming leaves transformer submodules host-resident between
    group onloads, so the indicator must move activations to wherever each
    submodule's weights actually live instead of assuming one device.
    """
    parameter = next(module.parameters(), None)
    if parameter is None:
        return fallback_device, fallback_dtype
    return parameter.device, parameter.dtype


def _module_device(module, fallback_device):
    if not isinstance(module, torch.nn.Module):
        return fallback_device
    tensor = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    return fallback_device if tensor is None else tensor.device


def _teacache_signal(transformer, args: tuple, kwargs: dict, config):
    """Return the first H3 block's AdaLN-modulated noisy video input."""
    hidden_states = _resolve_input(
        config.signal_name, args, kwargs, config.io_names
    )
    timestep = _resolve_input("timestep", args, kwargs, config.io_names)
    timestep_indices = _resolve_input(
        "timestep_indices", args, kwargs, config.io_names
    )
    token_tags = _resolve_input("token_tags", args, kwargs, config.io_names)
    video_indices = _resolve_input(
        "video_indices", args, kwargs, config.io_names
    )

    device, dtype = _module_target(
        transformer.proj_in, hidden_states.device, hidden_states.dtype
    )
    projected = transformer.proj_in(
        hidden_states.to(device=device, dtype=dtype)
    )
    block = transformer.transformer_blocks[0]
    device, dtype = _module_target(
        block.norm1, projected.device, projected.dtype
    )
    projected = projected.to(device=device, dtype=dtype)
    device, _ = _module_target(
        transformer.time_proj, timestep.device, timestep.dtype
    )
    temb = transformer.time_proj(timestep.to(device=device))
    device, dtype = _module_target(
        transformer.time_embedder, temb.device, temb.dtype
    )
    temb = transformer.time_embedder(temb.to(device=device, dtype=dtype))
    shift_msa, scale_msa, *_ = block.adaln_proj(
        temb.to(device=_module_device(block.adaln_proj, temb.device))
    )
    norm_hidden_states = block.norm1(projected)
    device = norm_hidden_states.device
    dtype = norm_hidden_states.dtype
    adaln_indices = (
        (timestep_indices * 3 + token_tags)
        .index_select(0, video_indices)
        .to(device)
    )
    scale_msa = scale_msa.to(device=device, dtype=dtype).index_select(
        0, adaln_indices
    )
    shift_msa = shift_msa.to(device=device, dtype=dtype).index_select(
        0, adaln_indices
    )
    return norm_hidden_states * (1.0 + scale_msa) + shift_msa


def _project_to_module(module, value):
    device, dtype = _module_target(module, value.device, value.dtype)
    return module(value.to(device=device, dtype=dtype))


def _fbcache_signal(transformer, args: tuple, kwargs: dict, config):
    """Return the real first H3 block output-minus-input residual."""
    hidden_states = _resolve_input(
        config.signal_name, args, kwargs, config.io_names
    )
    audio_hidden_states = _resolve_input(
        "audio_hidden_states", args, kwargs, config.io_names
    )
    encoder_hidden_states = _resolve_input(
        "encoder_hidden_states", args, kwargs, config.io_names
    )
    timestep = _resolve_input("timestep", args, kwargs, config.io_names)
    timestep_indices = _resolve_input(
        "timestep_indices", args, kwargs, config.io_names
    )
    token_tags = _resolve_input("token_tags", args, kwargs, config.io_names)
    position_ids = _resolve_input("position_ids", args, kwargs, config.io_names)
    video_indices = _resolve_input(
        "video_indices", args, kwargs, config.io_names
    )
    audio_indices = _resolve_input(
        "audio_indices", args, kwargs, config.io_names
    )
    text_indices = _resolve_input("text_indices", args, kwargs, config.io_names)

    video_embeds = _project_to_module(transformer.proj_in, hidden_states)
    audio_embeds = _project_to_module(
        transformer.audio_proj_in, audio_hidden_states
    )
    text_embeds = _project_to_module(
        transformer.context_embedder, encoder_hidden_states
    )
    text_embeds = _project_to_module(transformer.token_refiner, text_embeds)
    packed = text_embeds.new_zeros(
        (text_embeds.shape[0], position_ids.shape[0], text_embeds.shape[-1])
    )
    packed = packed.index_copy(1, text_indices.to(packed.device), text_embeds)
    packed = packed.index_copy(
        1,
        video_indices.to(packed.device),
        video_embeds.to(device=packed.device, dtype=packed.dtype),
    )
    packed = packed.index_copy(
        1,
        audio_indices.to(packed.device),
        audio_embeds.to(device=packed.device, dtype=packed.dtype),
    )

    temb = _project_to_module(transformer.time_proj, timestep)
    temb = _project_to_module(transformer.time_embedder, temb)
    block = transformer.transformer_blocks[0]
    device, dtype = _module_target(block.norm1, packed.device, packed.dtype)
    packed = packed.to(device=device, dtype=dtype)
    temb = temb.to(device=_module_device(block.adaln_proj, temb.device))
    adaln_indices = (timestep_indices * 3 + token_tags).to(packed.device)
    rotary_emb = transformer.rope(
        position_ids.to(_module_device(transformer.rope, packed.device))
    )
    rotary_emb = tree_map(lambda tensor: tensor.to(packed.device), rotary_emb)
    block_output = block(packed, temb, adaln_indices, rotary_emb)
    packed = packed.to(device=block_output.device, dtype=block_output.dtype)
    return _subtract(block_output, packed)


def _indicator_signal(transformer, args: tuple, kwargs: dict, config):
    if config.indicator == "teacache":
        return _teacache_signal(transformer, args, kwargs, config)
    if config.indicator == "fbcache":
        return _fbcache_signal(transformer, args, kwargs, config)
    return _resolve_input(config.signal_name, args, kwargs, config.io_names)


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


def _scale(value, factor):
    return tree_map(lambda tensor: tensor * factor, value)


def _update_taylor(slot, value, current_call: int) -> None:
    derivative = None
    if slot.value is not None and slot.computed_call is not None:
        window = current_call - slot.computed_call
        try:
            derivative = _scale(_subtract(value, slot.value), 1.0 / window)
        except ValueError:
            pass
    slot.value = _snapshot(value)
    slot.derivative = None if derivative is None else _snapshot(derivative)
    slot.computed_call = current_call


def _taylor_value(slot, current_call: int):
    if slot.derivative is None:
        return slot.value
    elapsed = current_call - slot.computed_call
    return _add(slot.value, _scale(slot.derivative, elapsed))


@contextmanager
def denoise_step_cache(transformer, config, total_steps: int):
    """Cache denoise outputs for one generation.

    The Python skip decision is compiler-disabled, and cached calls bypass the
    transformer's forward entirely rather than entering a compiled graph.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if getattr(transformer, "_omni_denoise_cache_active", False):
        raise RuntimeError("denoise-step cache is already enabled")

    had_instance_forward = "forward" in transformer.__dict__
    instance_forward = transformer.__dict__.get("forward")
    original_forward = transformer.forward
    forward_delegate = [original_forward]
    stats = DenoiseCacheStats()
    slots = [_Slot() for _ in range(config.calls_per_step)]
    calls = 0

    @wraps(original_forward)
    def cached_forward(*args, **kwargs):
        nonlocal calls
        step = calls // config.calls_per_step
        slot_index = calls % config.calls_per_step
        calls += 1
        slot = slots[slot_index]
        current_slot_call = slot.calls
        slot.calls += 1
        signal = _indicator_signal(transformer, args, kwargs, config)
        boundary = step < config.warmup_steps or (
            total_steps - config.final_steps <= step < total_steps
        )
        distance = (
            float("inf")
            if slot.signal is None
            else _relative_l1(signal, slot.signal)
        )
        compute, slot.accumulated = _cache_decision(
            boundary=boundary,
            has_output=slot.output is not None,
            distance=distance,
            coefficients=config.coefficients,
            threshold=config.threshold,
            accumulate=config.accumulate,
            accumulated=slot.accumulated,
        )
        slot.signal = _snapshot(signal)

        if compute:
            slot.accumulated = 0.0
            try:
                output = forward_delegate[0](*args, **kwargs)
            finally:
                preserve_wrapper(transformer, ())
            stats.computed += 1
            slot.output = output
            if config.mode == "residual":
                slot.inputs = _residual_inputs(args, kwargs, config.io_names)
                try:
                    slot.residual = _subtract(output, slot.inputs)
                except ValueError:
                    slot.residual = None
            if config.approximator == "taylor1":
                value = output if config.mode == "output" else slot.residual
                if value is None:
                    slot.value = None
                    slot.derivative = None
                    slot.computed_call = current_slot_call
                else:
                    _update_taylor(slot, value, current_slot_call)
            return output

        stats.skipped += 1
        if config.mode == "output":
            if config.approximator == "taylor1":
                return _taylor_value(slot, current_slot_call)
            return slot.output
        current_inputs = _residual_inputs(args, kwargs, config.io_names)
        if slot.residual is None:
            slot.residual = _subtract(slot.output, slot.inputs)
        residual = slot.residual
        if config.approximator == "taylor1":
            residual = _taylor_value(slot, current_slot_call)
        return _add(current_inputs, residual)

    def preserve_wrapper(current_module, _args):
        if current_module.forward is not cached_forward:
            forward_delegate[0] = current_module.forward
            current_module.forward = cached_forward

    transformer._omni_denoise_cache_active = True
    transformer.forward = cached_forward
    guardian_handle = transformer.register_forward_pre_hook(preserve_wrapper)
    try:
        yield stats
    finally:
        guardian_handle.remove()
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
