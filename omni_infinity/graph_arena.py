# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Pointer-stable arena block streaming for CUDA-graph capture.

Implements the design proven by ``benchmarks/probe_blockstream_graph.py``
(P2 Phase 3 / Phase 0, verdict FEASIBLE): the stock diffusers group-offload
hooks CPU-sync inside the capture region (``stream.synchronize()`` in
``ModuleGroup._onload_from_memory`` and ``GroupOffloadingHook.pre_forward``)
and rebind ``param.data`` to fresh allocations per onload, so they can never
be captured. This module replaces them for the ``cuda-graph + block-stream``
profile only:

- transformer block groups are discovered exactly like diffusers
  ``block_level``: direct-child ``nn.ModuleList`` / ``nn.Sequential``
  entries, chunked into groups of ``num_blocks_per_group`` blocks;
- two device arena slots are allocated per dtype, sized to the maximum
  group footprint; group ``g`` owns slot ``g % num_slots``;
- every group tensor's storage is rebound ONCE to a view inside its slot,
  so the pointers a captured graph reads never change; per-step H2D refills
  the slots from fixed pinned host buffers on a dedicated copy stream;
- hazards are enforced with events only (no CPU sync, no allocation in the
  hook path): compute(g) waits copy_done(g); copy(g) waits
  compute_done(g - num_slots); a fork event recorded on the current stream
  at forward entry orders the copy stream behind the previous step;
- non-block ("unmatched") transformer tensors are moved resident once --
  they are small next to the streamed blocks.

Two contracts callers must respect: the pinned host buffers are the sole
source of truth for block weights -- groups ``g`` and ``g + num_slots``
alias the same slot, so reading ``param.data`` (``state_dict()``, parity
dumps) outside the hook schedule returns whatever last refilled the slot;
and the forward pass must execute the groups in discovery order (the
hazard events assume it), which holds for H3's single sequential block
list exactly as it does for diffusers block_level.

CUDA primitives stay behind the injectable :class:`ArenaBackend` so the
grouping, sizing, slot-assignment, rebinding, and accounting logic is
CPU-testable (mirrors the ``cuda_graph.py`` backend pattern).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MethodType
from typing import Protocol

import torch

logger = logging.getLogger(__name__)

_ARENA_ATTR = "_omni_arena_streamer"


class ArenaBackend(Protocol):
    """CUDA primitives used by the streamer; fakeable for CPU tests."""

    def allocate_slot(self, numel: int, dtype: torch.dtype) -> torch.Tensor: ...

    def allocate_pinned(
        self, numel: int, dtype: torch.dtype
    ) -> torch.Tensor: ...

    def make_stream(self) -> object: ...

    def make_event(self) -> object: ...

    def record_event(self, event: object) -> None: ...

    def record_event_on(self, event: object, stream: object) -> None: ...

    def stream_wait_event(self, stream: object, event: object) -> None: ...

    def current_stream_wait_event(self, event: object) -> None: ...

    def stream_scope(self, stream: object): ...

    def move_resident(self, tensor: torch.Tensor) -> torch.Tensor: ...


class TorchArenaBackend:
    """Production backend over ``torch.cuda`` streams and events."""

    def __init__(self, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)

    def allocate_slot(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(numel, device=self.device, dtype=dtype)

    def allocate_pinned(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(numel, dtype=dtype).pin_memory()

    def make_stream(self) -> torch.cuda.Stream:
        return torch.cuda.Stream(device=self.device)

    def make_event(self) -> torch.cuda.Event:
        return torch.cuda.Event()

    def record_event(self, event) -> None:
        event.record(torch.cuda.current_stream(self.device))

    def record_event_on(self, event, stream) -> None:
        event.record(stream)

    def stream_wait_event(self, stream, event) -> None:
        stream.wait_event(event)

    def current_stream_wait_event(self, event) -> None:
        torch.cuda.current_stream(self.device).wait_event(event)

    def stream_scope(self, stream):
        return torch.cuda.stream(stream)

    def move_resident(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)


@dataclass
class BlockGroup:
    """One streamed group: consecutive blocks from a direct-child list."""

    index: int
    group_id: str
    modules: list[torch.nn.Module]
    slot: int = -1
    numel_by_dtype: dict[torch.dtype, int] = field(default_factory=dict)


def _named_tensor_entries(module: torch.nn.Module):
    """Yield ``(owner, name, tensor, is_param)`` in deterministic order."""
    for submodule in module.modules():
        for name, param in submodule._parameters.items():
            if param is not None:
                yield submodule, name, param, True
        for name, buffer in submodule._buffers.items():
            if buffer is not None:
                yield submodule, name, buffer, False


def discover_block_groups(
    transformer: torch.nn.Module, num_blocks_per_group: int
) -> list[BlockGroup]:
    """Chunk direct-child ModuleList/Sequential blocks into groups.

    Mirrors diffusers ``_apply_group_offloading_block_level`` discovery so
    the arena path streams exactly the modules the stock profile streams.
    """
    if num_blocks_per_group < 1:
        raise ValueError("num_blocks_per_group must be at least one")
    groups: list[BlockGroup] = []
    for name, child in transformer.named_children():
        if not isinstance(child, (torch.nn.ModuleList, torch.nn.Sequential)):
            continue
        for start in range(0, len(child), num_blocks_per_group):
            modules = list(child[start : start + num_blocks_per_group])
            if not modules:
                continue
            stop = start + len(modules) - 1
            groups.append(
                BlockGroup(
                    index=len(groups),
                    group_id=f"{name}_{start}_{stop}",
                    modules=modules,
                )
            )
    if not groups:
        raise ValueError(
            "arena streaming found no direct-child ModuleList/Sequential "
            "blocks on the transformer"
        )
    return groups


def plan_group_footprints(groups: list[BlockGroup]) -> dict[torch.dtype, int]:
    """Fill per-group dtype footprints; return max slot numel per dtype."""
    slot_numels: dict[torch.dtype, int] = {}
    for group in groups:
        numels: dict[torch.dtype, int] = {}
        for module in group.modules:
            for _owner, _name, tensor, _is_param in _named_tensor_entries(
                module
            ):
                numels[tensor.dtype] = (
                    numels.get(tensor.dtype, 0) + tensor.numel()
                )
        group.numel_by_dtype = numels
        for dtype, numel in numels.items():
            slot_numels[dtype] = max(slot_numels.get(dtype, 0), numel)
    if not slot_numels:
        raise ValueError("arena streaming found no tensors in block groups")
    return slot_numels


def _guarded_to(module, *args, **kwargs):
    """``.to()`` no-op guard for the arena-streamed transformer.

    Arena-bound ``param.data`` views must never be rebound by a device
    move (the captured graph reads those pointers). Diffusers group
    offloading has the same contract: a group-offloaded module refuses
    ``.to()``. The ComponentsManager auto-offload hook calls ``.to()`` at
    attach time and on offload decisions; both become no-ops here.
    """
    del args, kwargs
    logger.debug("ignoring .to() on the arena-streamed transformer")
    return module


class ArenaBlockStreamer:
    """Double-buffered pointer-stable weight arenas with hook scheduling.

    Installs forward pre-hooks (first module of each group) and forward
    hooks (last module) that reproduce the probe's event schedule on a
    dedicated copy stream. The hook path performs no CPU sync, calls no
    ``.item()``, and allocates nothing, so a whole-forward CUDA-graph
    capture records the H2D refills as graph memcpy nodes that re-read the
    pinned host buffers every replay.
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        *,
        num_blocks_per_group: int = 1,
        num_slots: int = 2,
        backend: ArenaBackend | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        if num_slots < 2:
            raise ValueError("arena streaming needs at least two slots")
        if getattr(transformer, _ARENA_ATTR, None) is not None:
            raise RuntimeError(
                "arena streaming is already installed on this transformer"
            )
        self.transformer = transformer
        self.num_slots = num_slots
        self.backend: ArenaBackend = (
            backend if backend is not None else TorchArenaBackend(device)
        )
        self.groups = discover_block_groups(transformer, num_blocks_per_group)
        slot_numels = plan_group_footprints(self.groups)
        for group in self.groups:
            group.slot = group.index % num_slots

        self._slots: dict[torch.dtype, list[torch.Tensor]] = {
            dtype: [
                self.backend.allocate_slot(numel, dtype)
                for _ in range(num_slots)
            ]
            for dtype, numel in slot_numels.items()
        }
        self._pinned: list[dict[torch.dtype, torch.Tensor]] = []
        # Per group: (arena_view, host_view) pairs the copy hook refills.
        self._views: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        handled: set[int] = set()
        for group in self.groups:
            self._bind_group(group, handled)
        self._move_residents(handled)

        self.copy_stream = self.backend.make_stream()
        self.fork = self.backend.make_event()
        self.copy_done = [self.backend.make_event() for _ in self.groups]
        self.compute_done = [self.backend.make_event() for _ in self.groups]
        self._handles: list[object] = []
        for group in self.groups:
            self._handles.append(
                group.modules[0].register_forward_pre_hook(
                    self._make_pre_hook(group.index)
                )
            )
            self._handles.append(
                group.modules[-1].register_forward_hook(
                    self._make_post_hook(group.index)
                )
            )
        self._original_to = transformer.to
        transformer.to = MethodType(_guarded_to, transformer)
        setattr(transformer, _ARENA_ATTR, self)
        logger.info(
            "arena streaming installed: %d groups, %d slots, "
            "%.2f GiB device arenas, %.2f GiB pinned host",
            len(self.groups),
            num_slots,
            self.arena_bytes / 1024**3,
            self.pinned_host_bytes / 1024**3,
        )

    def _bind_group(self, group: BlockGroup, handled: set[int]) -> None:
        """Pin the group's host bytes and rebind its tensors once."""
        pinned = {
            dtype: self.backend.allocate_pinned(numel, dtype)
            for dtype, numel in group.numel_by_dtype.items()
        }
        offsets = dict.fromkeys(group.numel_by_dtype, 0)
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for module in group.modules:
            for owner, name, tensor, is_param in _named_tensor_entries(module):
                handled.add(id(tensor))
                dtype = tensor.dtype
                offset = offsets[dtype]
                numel = tensor.numel()
                host_view = pinned[dtype][offset : offset + numel].view(
                    tensor.shape
                )
                arena = self._slots[dtype][group.slot]
                arena_view = arena[offset : offset + numel].view(tensor.shape)
                host_view.copy_(tensor.detach())
                if is_param:
                    tensor.data = arena_view
                else:
                    owner._buffers[name] = arena_view
                pairs.append((arena_view, host_view))
                offsets[dtype] = offset + numel
        self._pinned.append(pinned)
        self._views.append(pairs)

    def _move_residents(self, handled: set[int]) -> None:
        """Move non-block transformer tensors resident, once."""
        for owner, name, tensor, is_param in _named_tensor_entries(
            self.transformer
        ):
            if id(tensor) in handled:
                continue
            # Group tensors were rebound to arena views already; their
            # new .data objects must not be moved either.
            if any(
                tensor is view for pairs in self._views for view, _host in pairs
            ):
                continue
            if is_param:
                tensor.data = self.backend.move_resident(tensor.data)
            else:
                owner._buffers[name] = self.backend.move_resident(tensor)

    def _issue_copy(self, index: int) -> None:
        """H2D-refill group ``index`` on the copy stream (event-guarded)."""
        if index >= self.num_slots:
            self.backend.stream_wait_event(
                self.copy_stream, self.compute_done[index - self.num_slots]
            )
        with self.backend.stream_scope(self.copy_stream):
            for arena_view, host_view in self._views[index]:
                arena_view.copy_(host_view, non_blocking=True)
        self.backend.record_event_on(self.copy_done[index], self.copy_stream)

    def _make_pre_hook(self, index: int):
        def pre_hook(module, args):
            del module, args
            if index == 0:
                # Fork recorded on the CURRENT stream at forward entry:
                # orders the copy stream behind the previous step both in
                # eager mode and inside a capture (replays serialize on
                # the launch stream).
                self.backend.record_event(self.fork)
                self.backend.stream_wait_event(self.copy_stream, self.fork)
                self._issue_copy(0)
            nxt = index + 1
            if nxt < len(self.groups):
                self._issue_copy(nxt)
            self.backend.current_stream_wait_event(self.copy_done[index])

        return pre_hook

    def _make_post_hook(self, index: int):
        def post_hook(module, args, output):
            del module, args
            self.backend.record_event(self.compute_done[index])
            return output

        return post_hook

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def arena_bytes(self) -> int:
        return sum(
            slot.numel() * slot.element_size()
            for slots in self._slots.values()
            for slot in slots
        )

    @property
    def pinned_host_bytes(self) -> int:
        return sum(
            flat.numel() * flat.element_size()
            for pinned in self._pinned
            for flat in pinned.values()
        )

    def remove(self) -> None:
        """Detach hooks and the ``.to()`` guard.

        Weights stay arena-bound: restoring the original storage layout is
        out of scope (tear down the transformer instead). Intended for
        tests and controlled teardown only.
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.transformer.to = self._original_to
        if getattr(self.transformer, _ARENA_ATTR, None) is self:
            delattr(self.transformer, _ARENA_ATTR)


__all__ = [
    "ArenaBackend",
    "ArenaBlockStreamer",
    "BlockGroup",
    "TorchArenaBackend",
    "discover_block_groups",
    "plan_group_footprints",
]
