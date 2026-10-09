# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from omni_infinity.graph_arena import (
    ArenaBlockStreamer,
    discover_block_groups,
    plan_group_footprints,
)


class FakeArenaBackend:
    """CPU stand-in: real tensors, no-op streams/events, an op log."""

    def __init__(self):
        self.log = []

    def allocate_slot(self, numel, dtype):
        return torch.empty(numel, dtype=dtype)

    def allocate_pinned(self, numel, dtype):
        return torch.empty(numel, dtype=dtype)

    def make_stream(self):
        return object()

    def make_event(self):
        return object()

    def record_event(self, event):
        self.log.append(("record_current", event))

    def record_event_on(self, event, stream):
        self.log.append(("record_on_copy", event))

    def stream_wait_event(self, stream, event):
        self.log.append(("stream_wait", event))

    def current_stream_wait_event(self, event):
        self.log.append(("current_wait", event))

    def stream_scope(self, stream):
        return nullcontext()

    def move_resident(self, tensor):
        self.log.append(("move_resident", tensor.shape))
        return tensor


class ToyBlock(torch.nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


class ToyTransformer(torch.nn.Module):
    def __init__(self, n_blocks=4, hidden=8):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            ToyBlock(hidden) for _ in range(n_blocks)
        )
        self.norm_out = torch.nn.LayerNorm(hidden)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


def _seeded(n_blocks=4, hidden=8):
    torch.manual_seed(0)
    return ToyTransformer(n_blocks=n_blocks, hidden=hidden)


def _install(model, **kwargs):
    backend = FakeArenaBackend()
    streamer = ArenaBlockStreamer(model, backend=backend, **kwargs)
    return streamer, backend


def test_discovery_matches_diffusers_block_level_chunking():
    model = _seeded(n_blocks=5)

    singles = discover_block_groups(model, 1)
    assert [group.group_id for group in singles] == [
        f"blocks_{i}_{i}" for i in range(5)
    ]
    assert [group.modules for group in singles] == [
        [block] for block in model.blocks
    ]

    pairs = discover_block_groups(model, 2)
    assert [group.group_id for group in pairs] == [
        "blocks_0_1",
        "blocks_2_3",
        "blocks_4_4",
    ]
    assert pairs[0].modules == [model.blocks[0], model.blocks[1]]
    assert pairs[2].modules == [model.blocks[4]]


def test_discovery_rejects_bad_group_size_and_blockless_modules():
    with pytest.raises(ValueError, match="at least one"):
        discover_block_groups(_seeded(), 0)
    with pytest.raises(ValueError, match="no direct-child"):
        discover_block_groups(torch.nn.Linear(4, 4), 1)


def test_footprint_planning_sizes_slots_to_the_max_group():
    model = _seeded()
    model.blocks.append(ToyBlock(hidden=8))
    big = torch.nn.Linear(32, 32)
    model.blocks[-1].fc1 = big

    groups = discover_block_groups(model, 1)
    slot_numels = plan_group_footprints(groups)

    small = sum(p.numel() for p in ToyBlock(hidden=8).parameters())
    largest = sum(p.numel() for p in model.blocks[-1].parameters())
    assert largest > small
    assert slot_numels == {torch.float32: largest}
    assert groups[0].numel_by_dtype == {torch.float32: small}


def test_slot_assignment_alternates_between_two_slots():
    streamer, _backend = _install(_seeded(n_blocks=5))
    assert [group.slot for group in streamer.groups] == [0, 1, 0, 1, 0]


def test_rebinding_is_pointer_stable_across_forwards():
    model = _seeded()
    streamer, _backend = _install(model)

    slot_ptr_ranges = {
        dtype: [
            (slot.data_ptr(), slot.data_ptr() + slot.nbytes) for slot in slots
        ]
        for dtype, slots in streamer._slots.items()
    }
    before = {
        name: param.data_ptr() for name, param in model.named_parameters()
    }
    for name, param in model.blocks.named_parameters():
        group_index = int(name.split(".")[0])
        slot = streamer.groups[group_index].slot
        low, high = slot_ptr_ranges[param.dtype][slot]
        assert low <= param.data_ptr() < high, name

    x = torch.randn(2, 8)
    model(x)
    model(x)

    after = {name: param.data_ptr() for name, param in model.named_parameters()}
    assert before == after


def test_streamed_forward_matches_eager_reference():
    reference = _seeded()
    model = _seeded()
    streamer, _backend = _install(model)

    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = reference(x)
        actual = model(x)
    assert torch.equal(actual, expected)

    for pinned in streamer._pinned:
        for flat in pinned.values():
            flat.add_(0.25)
    with torch.no_grad():
        mutated = model(x)
    assert not torch.equal(mutated, expected)


def test_hook_schedule_reproduces_probe_event_ordering():
    model = _seeded(n_blocks=4)
    streamer, backend = _install(model)
    backend.log.clear()

    with torch.no_grad():
        model(torch.randn(2, 8))

    def copy_ops(index):
        ops = []
        if index >= streamer.num_slots:
            ops.append(("stream_wait", streamer.compute_done[index - 2]))
        ops.append(("record_on_copy", streamer.copy_done[index]))
        return ops

    expected = [
        ("record_current", streamer.fork),
        ("stream_wait", streamer.fork),
        *copy_ops(0),
        *copy_ops(1),
        ("current_wait", streamer.copy_done[0]),
        ("record_current", streamer.compute_done[0]),
        *copy_ops(2),
        ("current_wait", streamer.copy_done[1]),
        ("record_current", streamer.compute_done[1]),
        *copy_ops(3),
        ("current_wait", streamer.copy_done[2]),
        ("record_current", streamer.compute_done[2]),
        ("current_wait", streamer.copy_done[3]),
        ("record_current", streamer.compute_done[3]),
    ]
    assert backend.log == expected


def test_byte_accounting_counts_slots_and_pinned_buffers():
    model = _seeded(n_blocks=4)
    streamer, _backend = _install(model)

    per_block = sum(p.numel() for p in ToyBlock(hidden=8).parameters())
    element = torch.empty(0, dtype=torch.float32).element_size()
    assert streamer.arena_bytes == 2 * per_block * element
    assert streamer.pinned_host_bytes == 4 * per_block * element
    assert streamer.num_groups == 4


def test_non_block_tensors_are_moved_resident_not_streamed():
    model = _seeded()
    streamer, backend = _install(model)

    streamed = {id(view) for pairs in streamer._views for view, _host in pairs}
    assert id(model.norm_out.weight.data) not in streamed
    moved = [shape for op, shape in backend.log if op == "move_resident"]
    assert sorted(moved) == sorted(
        [model.norm_out.weight.shape, model.norm_out.bias.shape]
    )


def test_to_guard_blocks_componentsmanager_style_moves():
    model = _seeded()
    _streamer, _backend = _install(model)

    before = {
        name: param.data_ptr() for name, param in model.named_parameters()
    }
    assert model.to("cpu") is model
    after = {name: param.data_ptr() for name, param in model.named_parameters()}
    assert before == after


def test_double_install_and_bad_slot_count_are_rejected():
    model = _seeded()
    _streamer, _backend = _install(model)
    with pytest.raises(RuntimeError, match="already installed"):
        ArenaBlockStreamer(model, backend=FakeArenaBackend())

    with pytest.raises(ValueError, match="at least two slots"):
        ArenaBlockStreamer(_seeded(), num_slots=1, backend=FakeArenaBackend())


def test_remove_restores_hooks_and_to_method():
    model = _seeded()
    streamer, backend = _install(model)
    streamer.remove()

    backend.log.clear()
    with torch.no_grad():
        model(torch.randn(2, 8))
    assert backend.log == []
    assert not hasattr(model, "_omni_arena_streamer")

    ArenaBlockStreamer(model, backend=FakeArenaBackend())
