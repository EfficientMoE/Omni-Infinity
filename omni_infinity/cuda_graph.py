# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Resident-profile CUDA graph capture and replay.

The decision, quarantine, generation, and statistics pattern is adapted from
``MoE-Infinity/moe_infinity/serving/cuda_graph.py``. Shared graph pools,
static input/output buffers, warmup-before-capture, and bucket eviction are
adapted from ``BatchGen/batchgen/cuda_graph/graph_manager.py``. Both donors
are sibling EfficientMoE projects. This adaptation keys graphs by H3 video
shape and keeps CUDA primitives behind an injectable backend so lifecycle
logic remains CPU-testable.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)

FALLBACK_REASONS = (
    "shape_bucket_miss",
    "warmup_not_done",
    "capture_failed",
    "quarantined",
    "cache_skip",
    "kwarg_mismatch",
    "generation_invalidated",
    "replay_failed",
    "disabled",
)


@dataclass(frozen=True, order=True)
class GraphKey:
    """A fixed H3 video-shape bucket (batch size is always one)."""

    height: int
    width: int
    frames: int

    def __post_init__(self) -> None:
        if min(self.height, self.width, self.frames) <= 0:
            raise ValueError("CUDA graph bucket dimensions must be positive")


@dataclass(frozen=True)
class GraphDecision:
    eligible: bool
    reason: str
    key: GraphKey | None = None


@dataclass
class GraphExecutionStats:
    captures: int = 0
    replays: int = 0
    capture_failures: int = 0
    graph_pool_bytes: int = 0
    capture_time_ms: float = 0.0
    warmup_calls: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)


@dataclass
class CapturedExecution:
    """Backend-owned graph plus its stable output tree."""

    graph: object
    outputs: object
    pool_bytes: int = 0


class GraphBackend(Protocol):
    def make_pool(self) -> object: ...

    def capture(self, function, pool: object) -> CapturedExecution: ...

    def replay(self, captured: CapturedExecution) -> None: ...

    def drop(self, captured: CapturedExecution) -> None: ...


class TorchCudaGraphBackend:
    """Small indirection over ``torch.cuda`` used by the production runner."""

    def __init__(self, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)

    def make_pool(self) -> object:
        return torch.cuda.graph_pool_handle()

    def capture(self, function, pool: object) -> CapturedExecution:
        caller_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller_stream)
        before = torch.cuda.memory_allocated(self.device)
        with torch.cuda.stream(self.stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                outputs = function()
        caller_stream.wait_stream(self.stream)
        pool_bytes = max(
            0, int(torch.cuda.memory_allocated(self.device) - before)
        )
        return CapturedExecution(graph, outputs, pool_bytes=pool_bytes)

    def replay(self, captured: CapturedExecution) -> None:
        captured.graph.replay()

    def drop(self, captured: CapturedExecution) -> None:
        del captured


@dataclass
class _StaticCall:
    function: object
    args: tuple
    kwargs: dict[str, Any]


@dataclass
class _WarmupState:
    function: object
    args: tuple
    kwargs: dict[str, Any]
    calls: int = 0


@dataclass
class _GraphState:
    execution: CapturedExecution
    call: _StaticCall
    generation: int


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {
            item.name: _clone_tree(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
        return dataclasses.replace(value, **changes)
    if isinstance(value, tuple):
        values = tuple(_clone_tree(item) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*values)
        return values
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    return value


def _compatibility(static, runtime) -> str | None:
    if isinstance(static, torch.Tensor) or isinstance(runtime, torch.Tensor):
        if not isinstance(static, torch.Tensor) or not isinstance(
            runtime, torch.Tensor
        ):
            return "kwarg_mismatch"
        if (
            static.shape != runtime.shape
            or static.dtype != runtime.dtype
            or static.device != runtime.device
            or static.layout != runtime.layout
        ):
            return "shape_bucket_miss"
        return None
    if type(static) is not type(runtime):
        return "kwarg_mismatch"
    if isinstance(static, (tuple, list)):
        if len(static) != len(runtime):
            return "kwarg_mismatch"
        for left, right in zip(static, runtime, strict=True):
            reason = _compatibility(left, right)
            if reason is not None:
                return reason
        return None
    if isinstance(static, dict):
        if static.keys() != runtime.keys():
            return "kwarg_mismatch"
        for key in static:
            reason = _compatibility(static[key], runtime[key])
            if reason is not None:
                return reason
        return None
    try:
        matches = static == runtime
    except Exception:
        matches = static is runtime
    if not isinstance(matches, bool):
        return "kwarg_mismatch"
    return None if matches else "kwarg_mismatch"


def _copy_tree(static, runtime) -> None:
    if isinstance(static, torch.Tensor):
        static.copy_(runtime, non_blocking=True)
        return
    if isinstance(static, (tuple, list)):
        for left, right in zip(static, runtime, strict=True):
            _copy_tree(left, right)
        return
    if isinstance(static, dict):
        for key in static:
            _copy_tree(static[key], runtime[key])


def _call_compatibility(
    static_args: tuple,
    static_kwargs: dict[str, Any],
    args: tuple,
    kwargs: dict[str, Any],
) -> str | None:
    reason = _compatibility(static_args, args)
    if reason is not None:
        return reason
    return _compatibility(static_kwargs, kwargs)


class CudaGraphManager:
    """Warm, capture, replay, and evict resident transformer graphs."""

    WARMUP_ITERATIONS = 2

    def __init__(
        self,
        *,
        backend: GraphBackend | None = None,
        enabled: bool = True,
        warmup_iterations: int = WARMUP_ITERATIONS,
        max_buckets: int = 2,
        device: torch.device | str = "cuda",
    ) -> None:
        if warmup_iterations < 1:
            raise ValueError("warmup_iterations must be at least one")
        if max_buckets < 1:
            raise ValueError("max_buckets must be at least one")
        self.enabled = bool(enabled)
        self.warmup_iterations = warmup_iterations
        self.max_buckets = max_buckets
        self.backend = backend or TorchCudaGraphBackend(device)
        self._pool = self.backend.make_pool()
        self._graphs: OrderedDict[GraphKey, _GraphState] = OrderedDict()
        self._warmups: dict[GraphKey, _WarmupState] = {}
        self._quarantined: dict[GraphKey, str] = {}
        self._invalidated: set[GraphKey] = set()
        self._stats = GraphExecutionStats()
        self._lock = threading.RLock()
        self.generation = 0
        self.current_bucket: GraphKey | None = None

    @contextmanager
    def bucket(self, key: GraphKey):
        previous = self.current_bucket
        self.current_bucket = key
        try:
            yield
        finally:
            self.current_bucket = previous

    def check_eligibility(
        self,
        key: GraphKey | None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> GraphDecision:
        kwargs = kwargs or {}
        if not self.enabled:
            return GraphDecision(False, "disabled", key)
        if key is None:
            return GraphDecision(False, "shape_bucket_miss")
        if key in self._quarantined:
            return GraphDecision(False, "quarantined", key)
        if key in self._invalidated:
            return GraphDecision(False, "generation_invalidated", key)
        state = self._graphs.get(key)
        if state is not None:
            reason = _call_compatibility(
                state.call.args, state.call.kwargs, args, kwargs
            )
            return GraphDecision(reason is None, reason or "eligible", key)
        warmup = self._warmups.get(key)
        if warmup is not None:
            reason = _call_compatibility(
                warmup.args, warmup.kwargs, args, kwargs
            )
            if reason is not None:
                return GraphDecision(False, reason, key)
            if warmup.calls >= self.warmup_iterations:
                return GraphDecision(True, "eligible", key)
            return GraphDecision(False, "warmup_not_done", key)
        if self._graphs or self._warmups:
            return GraphDecision(False, "shape_bucket_miss", key)
        return GraphDecision(False, "warmup_not_done", key)

    def try_execute(self, key: GraphKey | None, function, *args, **kwargs):
        with self._lock:
            decision = self.check_eligibility(key, args, kwargs)
            if key is None or decision.reason == "disabled":
                self._record_fallback(decision.reason)
                return None
            if decision.reason == "quarantined":
                self._record_fallback("quarantined")
                return None
            if decision.reason == "generation_invalidated":
                self._invalidated.discard(key)
                self._record_fallback("generation_invalidated")
                return None

            state = self._graphs.get(key)
            if state is not None:
                if not decision.eligible:
                    self._record_fallback(decision.reason)
                    return None
                _copy_tree(state.call.args, args)
                _copy_tree(state.call.kwargs, kwargs)
                try:
                    self.backend.replay(state.execution)
                except Exception as exc:
                    logger.warning(
                        "CUDA graph replay failed for %s: %s", key, exc
                    )
                    self._quarantined[key] = type(exc).__name__
                    self._graphs.pop(key, None)
                    self._stats.graph_pool_bytes -= state.execution.pool_bytes
                    self.backend.drop(state.execution)
                    self._record_fallback("replay_failed")
                    return None
                self._graphs.move_to_end(key)
                self._stats.replays += 1
                return _clone_tree(state.execution.outputs)

            warmup = self._warmups.get(key)
            if warmup is None:
                reason = decision.reason
                warmup = _WarmupState(
                    function=function,
                    args=_clone_tree(args),
                    kwargs=_clone_tree(kwargs),
                )
                self._warmups[key] = warmup
            else:
                reason = _call_compatibility(
                    warmup.args, warmup.kwargs, args, kwargs
                )
                if reason is not None:
                    self._warmups[key] = _WarmupState(
                        function=function,
                        args=_clone_tree(args),
                        kwargs=_clone_tree(kwargs),
                        calls=1,
                    )
                    self._stats.warmup_calls += 1
                    self._record_fallback(reason)
                    return None
                reason = "warmup_not_done"

            if warmup.calls < self.warmup_iterations:
                warmup.calls += 1
                self._stats.warmup_calls += 1
                self._record_fallback(reason)
                return None

            static_call = _StaticCall(
                function=function,
                args=_clone_tree(args),
                kwargs=_clone_tree(kwargs),
            )

            def captured_forward():
                return static_call.function(
                    *static_call.args, **static_call.kwargs
                )

            started = time.perf_counter()
            try:
                execution = self.backend.capture(captured_forward, self._pool)
                # CUDA stream capture records the work but does not provide a
                # scheduler-consumable result for this denoise step. Replay
                # once with the just-copied inputs before returning outputs.
                # This launch is capture cost, not a steady-state replay stat.
                self.backend.replay(execution)
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1e3
                self._stats.capture_time_ms += elapsed
                self._stats.capture_failures += 1
                self._quarantined[key] = type(exc).__name__
                self._warmups.pop(key, None)
                self._record_fallback("capture_failed")
                logger.warning("CUDA graph capture failed for %s: %s", key, exc)
                return None
            elapsed = (time.perf_counter() - started) * 1e3
            self._stats.capture_time_ms += elapsed
            self._warmups.pop(key, None)
            while len(self._graphs) >= self.max_buckets:
                oldest = next(iter(self._graphs))
                self.drop_bucket(oldest)
            self._graphs[key] = _GraphState(
                execution=execution,
                call=static_call,
                generation=self.generation,
            )
            self._stats.captures += 1
            self._stats.graph_pool_bytes += execution.pool_bytes
            return _clone_tree(execution.outputs)

    def record_cache_skip(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("cache skip count must be non-negative")
        if count:
            self._stats.fallback_reasons["cache_skip"] += count

    def drop_bucket(self, key: GraphKey) -> bool:
        state = self._graphs.pop(key, None)
        self._warmups.pop(key, None)
        self._quarantined.pop(key, None)
        if state is None:
            return False
        self._stats.graph_pool_bytes -= state.execution.pool_bytes
        self.backend.drop(state.execution)
        return True

    def has_bucket(self, key: GraphKey) -> bool:
        return key in self._graphs

    def invalidate(self, reason: str) -> None:
        del reason
        with self._lock:
            keys = (
                set(self._graphs) | set(self._warmups) | set(self._quarantined)
            )
            for state in self._graphs.values():
                self.backend.drop(state.execution)
            self._graphs.clear()
            self._warmups.clear()
            self._quarantined.clear()
            self._invalidated.update(keys)
            self._stats.graph_pool_bytes = 0
            self.generation += 1

    def stats_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "captures": self._stats.captures,
                "replays": self._stats.replays,
                "capture_failures": self._stats.capture_failures,
                "graph_pool_bytes": self._stats.graph_pool_bytes,
                "capture_time_ms": self._stats.capture_time_ms,
                "warmup_calls": self._stats.warmup_calls,
                "graphs": len(self._graphs),
                "generation": self.generation,
                "fallback_reasons": dict(self._stats.fallback_reasons),
            }

    def _record_fallback(self, reason: str) -> None:
        if reason not in FALLBACK_REASONS:
            raise ValueError(f"unknown CUDA graph fallback reason {reason!r}")
        self._stats.fallback_reasons[reason] += 1


__all__ = [
    "CapturedExecution",
    "CudaGraphManager",
    "FALLBACK_REASONS",
    "GraphDecision",
    "GraphExecutionStats",
    "GraphKey",
    "TorchCudaGraphBackend",
]
