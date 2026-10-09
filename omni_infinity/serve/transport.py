# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Cross-process role transport (P7 Phase 1).

Each split role runs in its own spawned process owning one device, per
the plan's "separate processes with a thin queue" decision (no
torch.distributed).  ``torch.multiprocessing`` queues carry
:class:`~omni_infinity.serve.roles.StagePayload` items between role
processes; CUDA tensors cross via CUDA-IPC handles, CPU tensors via
shared memory, so the same loop is unit-testable without a GPU.  A
``None`` item is the shutdown sentinel.
"""

from __future__ import annotations

import queue
import time
from typing import Any, Callable

import torch.multiprocessing as mp

from omni_infinity.serve.roles import Role, StagePayload

StageFn = Callable[[StagePayload], StagePayload | None]


def role_worker_loop(
    role: Role,
    device: str,
    stage_fn: StageFn,
    inbox: Any,
    outbox: Any,
) -> None:
    """Pull payloads, run the role's stage, push results downstream.

    Runs inside the spawned role process.  ``stage_fn`` receives each
    payload already annotated with ``meta["role"]``/``meta["device"]``
    and returns the downstream payload (or ``None`` to drop).  The
    ``None`` sentinel is forwarded so shutdown propagates along the
    role chain.
    """

    while True:
        payload = inbox.get()
        if payload is None:
            outbox.put(None)
            inbox.get()
            return
        payload.meta["role"] = role.value
        payload.meta["device"] = device
        result = stage_fn(payload)
        if result is not None:
            outbox.put(result)


def spawn_role_worker(
    role: Role,
    device: str,
    stage_fn: StageFn,
    *,
    queue_size: int = 2,
    inbox: Any = None,
    outbox: Any = None,
) -> tuple[mp.Process, Any, Any]:
    """Spawn one role process; returns ``(process, inbox, outbox)``.

    ``stage_fn`` must be spawn-picklable (a module-level callable);
    lambdas, closures, and objects holding device state will fail at
    ``process.start()``.  Workers are non-daemon: the owner must send
    the ``None`` sentinel and ``join()`` (or ``terminate()``) them, so
    queued payloads are never cut off mid-handoff.
    """

    context = mp.get_context("spawn")
    if inbox is None:
        inbox = context.Queue(maxsize=queue_size)
    if outbox is None:
        outbox = context.Queue(maxsize=queue_size)
    process = context.Process(
        target=role_worker_loop,
        args=(role, device, stage_fn, inbox, outbox),
    )
    process.start()
    return process, inbox, outbox


def shutdown_role_worker(
    process: mp.Process,
    inbox: Any,
    outbox: Any,
    *,
    timeout: float = 60.0,
) -> None:
    """Sentinel -> drain echo -> release -> join shutdown protocol.

    The worker echoes the ``None`` sentinel after its last output and
    then waits for a release ack before exiting, so every queued
    payload (including any undrained leftovers) is deserialized while
    the producer is still alive — required for tensors shared over
    multiprocessing.  The sentinel put drains ``outbox`` when queues
    are full, so bounded queues cannot deadlock shutdown.  CUDA
    tensors received from the worker must still be copied to
    consumer-owned memory (``HandoffQueue.put``'s D2D copy, or
    ``clone()``) before this returns/raises, since the producer exits
    at the end.
    """

    deadline = time.monotonic() + timeout
    leftovers = []
    while True:
        try:
            inbox.put(None, timeout=0.2)
            break
        except queue.Full:
            try:
                leftovers.append(outbox.get(timeout=0.2))
            except queue.Empty:
                pass
        if time.monotonic() > deadline:
            process.terminate()
            process.join(timeout=timeout)
            raise RuntimeError("timed out enqueueing shutdown sentinel")
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.terminate()
            process.join(timeout=timeout)
            raise RuntimeError("timed out draining role worker outputs")
        try:
            item = outbox.get(timeout=remaining)
        except queue.Empty:
            process.terminate()
            process.join(timeout=timeout)
            raise RuntimeError(
                "role worker produced no sentinel echo before the deadline"
            ) from None
        if item is None:
            break
        leftovers.append(item)
    try:
        inbox.put(None, timeout=max(0.2, deadline - time.monotonic()))
    except queue.Full:
        process.terminate()
        process.join(timeout=timeout)
        raise RuntimeError(
            "role worker did not accept the release ack"
        ) from None
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=timeout)
        raise RuntimeError("role worker did not exit after sentinel")
    if leftovers:
        error = RuntimeError(
            f"shutdown_role_worker found {len(leftovers)} undrained "
            "output(s); consume every result before shutdown. Recovered "
            "payloads are on the exception's 'leftovers' attribute — CPU "
            "tensors stay valid (shared memory), CUDA tensors are "
            "invalidated by the producer's exit"
        )
        error.leftovers = leftovers
        raise error
