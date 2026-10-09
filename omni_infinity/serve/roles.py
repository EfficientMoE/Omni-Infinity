# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU role abstraction for stage-pipelined serving (P7 Phase 1).

The serving profile gains ``OMNI_ROLE={all,encoder,denoiser,decoder}``
plus an optional ``OMNI_DEVICE_MAP``.  ``role="all"`` is the existing
single-process, single-device contract and stays the default.  Split
roles run as separate processes (one device each) connected by
:class:`HandoffQueue`, which moves stage payloads device-to-device with
copy-engine copies on a dedicated CUDA stream — never NCCL collectives
(see docs/multigpu_platform.md for the platform measurements behind
that rule).
"""

from __future__ import annotations

import enum
import queue
from dataclasses import dataclass, field
from typing import Any

import torch


class Role(str, enum.Enum):
    """Serving role.

    ``ALL`` is the single-process default; ``PIPELINE`` is the
    single-host coordinator that spawns the three stage roles as local
    processes (contract addition over the P7 plan's enum, noted in
    #42); the stage values are reserved for future per-role servers.
    """

    ALL = "all"
    PIPELINE = "pipeline"
    ENCODER = "encoder"
    DENOISER = "denoiser"
    DECODER = "decoder"


_SPLIT_ROLES = (Role.ENCODER, Role.DENOISER, Role.DECODER)


def parse_role(value: str) -> Role:
    """Parse ``OMNI_ROLE``; raise ``ValueError`` on unknown roles."""

    normalized = value.strip().lower()
    try:
        return Role(normalized)
    except ValueError:
        allowed = ", ".join(role.value for role in Role)
        raise ValueError(
            f"OMNI_ROLE={value!r} is not valid; expected one of: {allowed}"
        ) from None


def parse_device_map(
    value: str | None,
    *,
    default_device: str = "cuda",
) -> dict[Role, str]:
    """Parse ``OMNI_DEVICE_MAP`` ("encoder=cuda:0,denoiser=cuda:1,...").

    Unlisted split roles fall back to ``default_device``.  Raises
    ``ValueError`` for malformed entries or unknown role names.
    """

    mapping = {role: default_device for role in _SPLIT_ROLES}
    if not value or not value.strip():
        return mapping
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, separator, device = entry.partition("=")
        if not separator or not device.strip():
            raise ValueError(
                f"OMNI_DEVICE_MAP entry {entry!r} is not of the form "
                "role=device"
            )
        role = parse_role(name)
        if role not in _SPLIT_ROLES:
            raise ValueError(
                "OMNI_DEVICE_MAP may only map encoder/denoiser/decoder, "
                f"not {role.value!r}"
            )
        mapping[role] = device.strip()
    return mapping


@dataclass
class StagePayload:
    """Tensors crossing one stage boundary, plus job bookkeeping.

    ``encoder -> denoiser`` carries the condition embeddings;
    ``denoiser -> decoder`` carries video/audio latents.  ``meta``
    carries small picklable call state (seed, resolution, step counts)
    so the downstream role can resume the modular pipeline.
    """

    job_id: str
    stage: Role
    tensors: dict[str, torch.Tensor]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_device(
        self,
        device: str | torch.device,
        *,
        non_blocking: bool = False,
    ) -> StagePayload:
        """Return a copy of the payload with tensors on ``device``."""

        moved = {
            name: tensor.to(device, non_blocking=non_blocking)
            for name, tensor in self.tensors.items()
        }
        return StagePayload(
            job_id=self.job_id,
            stage=self.stage,
            tensors=moved,
            meta=dict(self.meta),
        )


class HandoffQueue:
    """Bounded intra-process stage-handoff queue with D2D placement.

    This is the single-process handoff primitive; the cross-process
    transport (``torch.multiprocessing`` CUDA-IPC queues between role
    processes) composes on top of it in a later P7 increment.

    ``put`` moves the payload to ``dest_device`` on a copy stream that
    lives on the *source* device (where PyTorch issues peer copies),
    ordered after the producer's current stream, so producer compute
    is not serialized behind the copy.  Source tensors are protected
    with ``record_stream`` and a CUDA event is recorded for the
    consumer, whose current stream waits on it in ``get``.  On
    CPU-only hosts (unit tests) the queue degrades to plain
    ``Tensor.to`` moves.
    """

    def __init__(self, dest_device: str, maxsize: int = 2) -> None:
        self.dest_device = torch.device(dest_device)
        self._queue: queue.Queue[tuple[StagePayload, Any]] = queue.Queue(
            maxsize=maxsize
        )
        self._use_cuda = (
            self.dest_device.type == "cuda" and torch.cuda.is_available()
        )
        self._copy_streams: dict[int, torch.cuda.Stream] = {}

    def _copy_stream_for(self, device: torch.device) -> torch.cuda.Stream:
        stream = self._copy_streams.get(device.index)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._copy_streams[device.index] = stream
        return stream

    def put(
        self,
        payload: StagePayload,
        *,
        timeout: float | None = None,
    ) -> None:
        """Copy ``payload`` to the destination device and enqueue it."""

        source = next(
            (
                tensor.device
                for tensor in payload.tensors.values()
                if tensor.is_cuda
            ),
            None,
        )
        if self._use_cuda and source is not None:
            copy_stream = self._copy_stream_for(source)
            copy_stream.wait_stream(torch.cuda.current_stream(source))
            with torch.cuda.stream(copy_stream):
                for tensor in payload.tensors.values():
                    if tensor.is_cuda:
                        tensor.record_stream(copy_stream)
                moved = payload.to_device(self.dest_device, non_blocking=True)
                event = torch.cuda.Event()
                event.record(copy_stream)
        else:
            moved = payload.to_device(self.dest_device)
            event = None
        self._queue.put((moved, event), timeout=timeout)

    def get(self, *, timeout: float | None = None) -> StagePayload:
        """Dequeue a payload; the caller's stream waits on the copy."""

        payload, event = self._queue.get(timeout=timeout)
        if event is not None:
            torch.cuda.current_stream(self.dest_device).wait_event(event)
        return payload

    def qsize(self) -> int:
        return self._queue.qsize()
