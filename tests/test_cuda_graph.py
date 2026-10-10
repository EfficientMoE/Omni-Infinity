# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import omni_infinity.cuda_graph as cuda_graph_mod
from omni_infinity.cuda_graph import (
    CapturedExecution,
    CudaGraphManager,
    GraphKey,
)


def _copy_tree(destination, source):
    if isinstance(destination, torch.Tensor):
        destination.copy_(source)
        return
    for target, value in zip(destination, source, strict=True):
        _copy_tree(target, value)


def _zero_tree(value):
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    return tuple(_zero_tree(item) for item in value)


class _FakeGraph:
    def __init__(self, function):
        self.function = function
        self.outputs = _zero_tree(function())


class FakeBackend:
    def __init__(self, *, fail_capture: bool = False):
        self.fail_capture = fail_capture
        self.capture_calls = 0
        self.replay_calls = 0
        self.dropped = []
        self.pools = []
        self.capture_warmups = []

    def make_pool(self):
        pool = object()
        self.pools.append(pool)
        return pool

    def capture(self, function, pool, *, warmup_iterations):
        assert pool in self.pools
        self.capture_calls += 1
        self.capture_warmups.append(warmup_iterations)
        if self.fail_capture:
            raise RuntimeError("synthetic capture failure")
        for _ in range(warmup_iterations):
            function()
        graph = _FakeGraph(function)
        return CapturedExecution(graph, graph.outputs, pool_bytes=64)

    def replay(self, captured):
        self.replay_calls += 1
        _copy_tree(captured.outputs, captured.graph.function())

    def drop(self, captured):
        self.dropped.append(captured)


def _capture(manager, key, value=3.0, **kwargs):
    tensor = torch.tensor([value])
    assert manager.try_execute(key, torch.mul, tensor, **kwargs) is None
    return manager.try_execute(key, torch.mul, tensor, **kwargs)


def test_bucket_warms_captures_and_replays_with_static_input_copy():
    backend = FakeBackend()
    manager = CudaGraphManager(
        backend=backend, warmup_iterations=2, max_buckets=2
    )
    key = GraphKey(height=256, width=256, frames=124)

    def function(value, *, offset):
        return value + offset, value * 2

    assert (
        manager.try_execute(
            key, function, torch.tensor([1.0]), offset=torch.tensor([2.0])
        )
        is None
    )
    assert (
        manager.try_execute(
            key, function, torch.tensor([2.0]), offset=torch.tensor([3.0])
        )
        is None
    )
    captured = manager.try_execute(
        key, function, torch.tensor([3.0]), offset=torch.tensor([4.0])
    )
    replayed = manager.try_execute(
        key, function, torch.tensor([5.0]), offset=torch.tensor([6.0])
    )

    assert tuple(tensor.item() for tensor in captured) == (7.0, 6.0)
    assert tuple(tensor.item() for tensor in replayed) == (11.0, 10.0)
    assert captured[0].data_ptr() != replayed[0].data_ptr()
    assert backend.capture_calls == 1
    assert backend.replay_calls == 2
    assert backend.capture_warmups == [2]
    assert manager.stats_snapshot()["fallback_reasons"] == {
        "warmup_not_done": 2
    }


def test_new_bucket_mid_flight_records_shape_bucket_miss():
    manager = CudaGraphManager(
        backend=FakeBackend(), warmup_iterations=1, max_buckets=2
    )
    first = GraphKey(256, 256, 124)
    second = GraphKey(512, 512, 124)
    _capture(manager, first, other=2.0)

    result = manager.try_execute(
        second, torch.mul, torch.tensor([2.0]), other=3.0
    )

    assert result is None
    assert (
        manager.stats_snapshot()["fallback_reasons"]["shape_bucket_miss"] == 1
    )


def test_capture_failure_quarantines_bucket_without_retry_storm():
    backend = FakeBackend(fail_capture=True)
    manager = CudaGraphManager(
        backend=backend, warmup_iterations=1, max_buckets=1
    )
    key = GraphKey(256, 256, 124)
    tensor = torch.tensor([2.0])

    assert manager.try_execute(key, torch.neg, tensor) is None
    assert manager.try_execute(key, torch.neg, tensor) is None
    assert manager.check_eligibility(key, (tensor,), {}).reason == "quarantined"
    assert manager.try_execute(key, torch.neg, tensor) is None

    snapshot = manager.stats_snapshot()
    assert backend.capture_calls == 1
    assert snapshot["capture_failures"] == 1
    assert snapshot["fallback_reasons"] == {
        "warmup_not_done": 1,
        "capture_failed": 1,
        "quarantined": 1,
    }


def test_invalidate_advances_generation_and_reports_stale_bucket_once():
    manager = CudaGraphManager(
        backend=FakeBackend(), warmup_iterations=1, max_buckets=1
    )
    key = GraphKey(256, 256, 124)
    _capture(manager, key, other=2.0)

    manager.invalidate("model changed")

    decision = manager.check_eligibility(
        key, (torch.tensor([1.0]),), {"other": 2.0}
    )
    assert decision.reason == "generation_invalidated"
    assert (
        manager.try_execute(key, torch.mul, torch.tensor([1.0]), other=2.0)
        is None
    )
    assert manager.generation == 1
    assert manager.stats_snapshot()["fallback_reasons"] == {
        "warmup_not_done": 1,
        "generation_invalidated": 1,
    }


def test_lru_eviction_drops_least_recently_replayed_bucket():
    backend = FakeBackend()
    manager = CudaGraphManager(
        backend=backend, warmup_iterations=1, max_buckets=2
    )
    first = GraphKey(256, 256, 124)
    second = GraphKey(512, 512, 124)
    third = GraphKey(768, 768, 124)
    _capture(manager, first, other=2.0)
    _capture(manager, second, other=2.0)
    assert (
        manager.try_execute(first, torch.mul, torch.tensor([4.0]), other=2.0)
        is not None
    )
    _capture(manager, third, other=2.0)

    assert manager.has_bucket(first)
    assert not manager.has_bucket(second)
    assert manager.has_bucket(third)
    assert len(backend.dropped) == 1
    assert len({id(pool) for pool in backend.pools}) == 3


def test_non_tensor_kwarg_change_falls_back_without_replay():
    backend = FakeBackend()
    manager = CudaGraphManager(
        backend=backend, warmup_iterations=1, max_buckets=1
    )
    key = GraphKey(256, 256, 124)

    def function(value, *, return_dict):
        return value + int(return_dict)

    assert (
        manager.try_execute(key, function, torch.tensor([1]), return_dict=False)
        is None
    )
    assert (
        manager.try_execute(key, function, torch.tensor([1]), return_dict=False)
        is not None
    )
    assert (
        manager.try_execute(key, function, torch.tensor([1]), return_dict=True)
        is None
    )

    assert backend.replay_calls == 1
    assert manager.stats_snapshot()["replays"] == 0
    assert manager.stats_snapshot()["fallback_reasons"]["kwarg_mismatch"] == 1


def test_stats_snapshot_and_cache_skip_hook_are_plain_dicts():
    manager = CudaGraphManager(
        backend=FakeBackend(), warmup_iterations=1, max_buckets=1
    )
    manager.record_cache_skip(3)
    snapshot = manager.stats_snapshot()

    assert snapshot == {
        "captures": 0,
        "replays": 0,
        "capture_failures": 0,
        "graph_pool_bytes": 0,
        "capture_time_ms": 0.0,
        "warmup_calls": 0,
        "capture_warmup_calls": 0,
        "graphs": 0,
        "generation": 0,
        "arena_bytes": 0,
        "pinned_host_bytes": 0,
        "fallback_reasons": {"cache_skip": 3},
    }
    assert isinstance(snapshot["fallback_reasons"], dict)


def test_disabled_manager_falls_back_without_touching_backend():
    backend = FakeBackend()
    manager = CudaGraphManager(backend=backend, enabled=False)
    key = GraphKey(256, 256, 124)

    assert manager.try_execute(key, torch.neg, torch.tensor([1.0])) is None
    assert backend.capture_calls == 0
    assert manager.stats_snapshot()["fallback_reasons"] == {"disabled": 1}


def test_disabled_default_manager_does_not_initialize_cuda(monkeypatch):
    def fail_backend(device):
        raise AssertionError(f"CUDA backend constructed for {device}")

    monkeypatch.setattr(cuda_graph_mod, "TorchCudaGraphBackend", fail_backend)

    manager = CudaGraphManager(enabled=False)

    assert manager.try_execute(None, lambda: None) is None
    assert manager.stats_snapshot()["fallback_reasons"] == {"disabled": 1}
