# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

import contextlib
import dataclasses

import pytest
import torch
import torch.nn as nn

HIDDEN = 11520
BOTTLENECK = 2912


@dataclasses.dataclass
class _CopyRecord:
    group_index: int
    source_step: int
    start: torch.cuda.Event
    end: torch.cuda.Event
    destination_step: int | None = None
    overlaps_last_group_compute: bool = False


@dataclasses.dataclass
class _ComputeRecord:
    group_index: int
    step: int
    start: torch.cuda.Event
    end: torch.cuda.Event


class _Timeline:
    def __init__(self):
        self.step = 0
        self.copies: list[_CopyRecord] = []
        self.compute: list[_ComputeRecord] = []


class _Block(nn.Module):
    def __init__(
        self,
        index: int,
        input_features: int,
        output_features: int,
        timeline: _Timeline,
    ):
        super().__init__()
        self.index = index
        self.timeline = timeline
        self.proj = nn.Linear(
            input_features,
            output_features,
            bias=False,
            dtype=torch.bfloat16,
        )

    def forward(self, hidden_states):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        hidden_states = self.proj(hidden_states)
        if hasattr(torch.cuda, "_sleep"):
            torch.cuda._sleep(300_000_000)
        end.record()
        self.timeline.compute.append(
            _ComputeRecord(self.index, self.timeline.step, start, end)
        )
        return hidden_states


class _SyntheticTransformer(nn.Module):
    def __init__(self, timeline: _Timeline, hidden: int = HIDDEN):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [
                _Block(0, hidden, BOTTLENECK, timeline),
                _Block(1, BOTTLENECK, hidden, timeline),
                _Block(2, hidden, hidden, timeline),
                _Block(3, hidden, hidden, timeline),
            ]
        )
        # Forces Diffusers to install its top-level lazy-prefetch hook.
        self.unmatched = nn.Identity()

    def forward(self, hidden_states):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


@contextlib.contextmanager
def _observe_onloads(timeline: _Timeline, groups: list):
    from diffusers.hooks.group_offloading import ModuleGroup

    original = ModuleGroup.onload_
    group_indices = {id(group): index for index, group in enumerate(groups)}

    def observed(group):
        index = group_indices.get(id(group))
        if index is None:
            return original(group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(group.stream)
        result = original(group)
        end.record(group.stream)
        timeline.copies.append(
            _CopyRecord(index, timeline.step, start, end)
        )
        return result

    ModuleGroup.onload_ = observed
    try:
        yield
    finally:
        ModuleGroup.onload_ = original


def _block_groups(transformer) -> list:
    from diffusers.hooks.group_offloading import _GROUP_OFFLOADING

    groups = []
    for block in transformer.transformer_blocks:
        hook = block._diffusers_hook.get_hook(_GROUP_OFFLOADING)
        groups.append(hook.group)
    return groups


def _interval(origin, record) -> tuple[float, float]:
    return origin.elapsed_time(record.start), origin.elapsed_time(record.end)


def _intersection_ms(
    interval: tuple[float, float], others: list[tuple[float, float]]
) -> float:
    start, end = interval
    intersections = [
        (max(start, other_start), min(end, other_end))
        for other_start, other_end in others
        if max(start, other_start) < min(end, other_end)
    ]
    if not intersections:
        return 0.0
    intersections.sort()
    merged = [list(intersections[0])]
    for current_start, current_end in intersections[1:]:
        if current_start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current_end)
        else:
            merged.append([current_start, current_end])
    return sum(end - start for start, end in merged)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_first_group_prefetch_overlaps_previous_step_tail():
    from diffusers.hooks import apply_group_offloading

    timeline = _Timeline()
    transformer = _SyntheticTransformer(timeline)
    apply_group_offloading(
        transformer,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=True,
        record_stream=False,
        low_cpu_mem_usage=False,
    )
    groups = _block_groups(transformer)
    streams = {group.stream for group in groups}
    assert None not in streams

    origin = torch.cuda.Event(enable_timing=True)
    origin.record()
    for stream in streams:
        stream.wait_event(origin)

    hidden_states = torch.zeros(1, HIDDEN, device="cpu", dtype=torch.bfloat16)
    with _observe_onloads(timeline, groups):
        for step in range(1, 4):
            timeline.step = step
            hidden_states = transformer(hidden_states)
    torch.cuda.synchronize()

    compute_intervals = {
        step: [
            _interval(origin, record)
            for record in timeline.compute
            if record.step == step
        ]
        for step in range(1, 4)
    }
    last_compute = {
        step: _interval(
            origin,
            next(
                record
                for record in timeline.compute
                if record.step == step and record.group_index == 3
            ),
        )
        for step in range(1, 4)
    }

    for copy in timeline.copies:
        copy_interval = _interval(origin, copy)
        tail_interval = last_compute[copy.source_step]
        copy.overlaps_last_group_compute = (
            _intersection_ms(copy_interval, [tail_interval]) > 0
        )
        copy.destination_step = (
            copy.source_step + 1
            if copy.group_index == 0
            and copy.overlaps_last_group_compute
            else copy.source_step
        )

    for step, intervals in compute_intervals.items():
        print(f"step {step} compute intervals: {intervals}")
    for copy in timeline.copies:
        print(
            f"copy group={copy.group_index} source={copy.source_step} "
            f"destination={copy.destination_step} "
            f"interval={_interval(origin, copy)} "
            f"tail_overlap={copy.overlaps_last_group_compute}"
        )

    measured_steps = [2, 3]
    assert 1 not in measured_steps
    for destination_step in measured_steps:
        copies = [
            copy
            for copy in timeline.copies
            if copy.destination_step == destination_step
        ]
        copy_intervals = [_interval(origin, copy) for copy in copies]
        total_h2d_ms = sum(end - start for start, end in copy_intervals)
        overlapped_h2d_ms = sum(
            _intersection_ms(interval, compute_intervals[destination_step])
            for interval in copy_intervals
        )
        ratio = overlapped_h2d_ms / total_h2d_ms
        print(
            f"step {destination_step}: overlapped_h2d_ms="
            f"{overlapped_h2d_ms:.3f} total_h2d_ms={total_h2d_ms:.3f} "
            f"ratio={ratio:.3f}"
        )
        first_group_copy = next(
            copy for copy in copies if copy.group_index == 0
        )
        assert ratio > 0.50
        assert first_group_copy.source_step == destination_step - 1
        assert first_group_copy.overlaps_last_group_compute
