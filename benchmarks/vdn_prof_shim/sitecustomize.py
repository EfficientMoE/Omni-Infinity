# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Source-free CUDA range instrumentation for upstream VDN inference.

Python imports this module automatically when its directory is on
``PYTHONPATH``.  Without ``VDN_PROF_OUT`` it deliberately does nothing, which
keeps CPU imports and unrelated commands safe.
"""

from __future__ import annotations

import atexit
import builtins
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

UPSTREAM_PATCH_TARGETS = (
    "src.models.hybrid_attention.HybridAttention._window_softmax",
    "src.models.softmax_attention.dense_processor.FlexFA4Processor.__call__",
)


class RangeAccounting:
    """Accumulate event pairs into one exclusive component map per NFE."""

    def __init__(
        self,
        event_factory: Callable[[], Any],
        synchronize: Callable[[], None],
        range_push: Callable[[str], None] | None = None,
        range_pop: Callable[[], None] | None = None,
        on_step: Callable[[list[dict[str, dict[str, float | int]]]], None]
        | None = None,
    ) -> None:
        self._event_factory = event_factory
        self._synchronize = synchronize
        self._range_push = range_push
        self._range_pop = range_pop
        self._on_step = on_step
        self._pending: list[tuple[str, Any, Any, str | None]] = []
        self._stack: list[str] = []
        self._linear_calls = 0
        self.steps: list[dict[str, dict[str, float | int]]] = []

    def count_linear(self) -> None:
        if self._stack:
            self._linear_calls += 1

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = self._event_factory()
        stop = self._event_factory()
        parent = self._stack[-1] if self._stack else None
        start.record()
        self._stack.append(name)
        if self._range_push is not None:
            self._range_push(name)
        try:
            yield
        finally:
            if self._range_pop is not None:
                self._range_pop()
            self._stack.pop()
            stop.record()
            self._pending.append((name, start, stop, parent))
            if name == "step_total":
                self._finish_step()

    def _finish_step(self) -> None:
        self._synchronize()
        elapsed = [
            (name, float(start.elapsed_time(stop)), parent)
            for name, start, stop, parent in self._pending
        ]
        totals: dict[str, dict[str, float | int]] = {}
        nested_gate_ms = sum(
            milliseconds
            for name, milliseconds, parent in elapsed
            if name == "gates" and parent == "linear_branch"
        )
        for name, milliseconds, _parent in elapsed:
            bucket = totals.setdefault(name, {"total_ms": 0.0, "calls": 0})
            bucket["total_ms"] = float(bucket["total_ms"]) + milliseconds
            bucket["calls"] = int(bucket["calls"]) + 1
        if "linear_branch" in totals:
            branch = totals["linear_branch"]
            branch["total_ms"] = float(branch["total_ms"]) - nested_gate_ms
        if self._linear_calls:
            totals["linear_calls"] = {
                "total_ms": 0.0,
                "calls": self._linear_calls,
            }
        self.steps.append(totals)
        self._pending.clear()
        self._linear_calls = 0
        if self._on_step is not None:
            self._on_step(self.steps)


_OUTPUT = os.environ.get("VDN_PROF_OUT")
_ACCOUNTING: RangeAccounting | None = None


def _write_steps(steps: list[dict[str, dict[str, float | int]]]) -> None:
    if not _OUTPUT:
        return
    path = Path(_OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"nfes": steps}, indent=2) + "\n")
    temporary.replace(path)


def _timed_method(method: Callable[..., Any], range_name: str):
    if getattr(method, "_vdn_prof_wrapped", False):
        return method

    def wrapped(*args: Any, **kwargs: Any):
        assert _ACCOUNTING is not None
        with _ACCOUNTING.measure(range_name):
            return method(*args, **kwargs)

    wrapped._vdn_prof_wrapped = True
    return wrapped


def _patch_loaded_modules() -> None:
    hybrid = sys.modules.get("src.models.hybrid_attention")
    if hybrid is not None and hasattr(hybrid, "HybridAttention"):
        cls = hybrid.HybridAttention
        cls._window_softmax = _timed_method(
            cls._window_softmax,
            "window_softmax",
        )
    dense = sys.modules.get("src.models.softmax_attention.dense_processor")
    if dense is not None and hasattr(dense, "FlexFA4Processor"):
        cls = dense.FlexFA4Processor
        cls.__call__ = _timed_method(cls.__call__, "dense_attn")


def _install() -> None:
    global _ACCOUNTING

    import torch
    import torch.nn.functional as functional

    _ACCOUNTING = RangeAccounting(
        event_factory=lambda: torch.cuda.Event(enable_timing=True),
        synchronize=torch.cuda.synchronize,
        range_push=torch.cuda.nvtx.range_push,
        range_pop=torch.cuda.nvtx.range_pop,
        on_step=_write_steps,
    )
    original_module_call = torch.nn.Module.__call__

    def module_call(module: Any, *args: Any, **kwargs: Any):
        class_name = type(module).__name__
        ranges = {
            "MiniMaxH3Transformer3DModel": "step_total",
            "VDNMiniMaxH3Transformer3DModel": "step_total",
            "BidirectionalLinearBranch": "linear_branch",
            "OutputGate": "gates",
        }
        range_name = ranges.get(class_name)
        if range_name is None:
            return original_module_call(module, *args, **kwargs)
        assert _ACCOUNTING is not None
        with _ACCOUNTING.measure(range_name):
            return original_module_call(module, *args, **kwargs)

    torch.nn.Module.__call__ = module_call

    original_linear = functional.linear

    def counted_linear(*args: Any, **kwargs: Any):
        assert _ACCOUNTING is not None
        _ACCOUNTING.count_linear()
        return original_linear(*args, **kwargs)

    functional.linear = counted_linear

    original_import = builtins.__import__

    def importing(*args: Any, **kwargs: Any):
        module = original_import(*args, **kwargs)
        _patch_loaded_modules()
        return module

    builtins.__import__ = importing
    _patch_loaded_modules()
    atexit.register(lambda: _write_steps(_ACCOUNTING.steps))


if _OUTPUT:
    _install()
