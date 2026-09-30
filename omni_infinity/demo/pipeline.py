# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

Stage = tuple[str, int]


class PlaybackClock(Protocol):
    def now(self) -> float: ...

    def wait_until(self, second: float) -> None: ...


class SimClock:
    def __init__(self) -> None:
        self._second = 0.0

    def now(self) -> float:
        return self._second

    def wait_until(self, second: float) -> None:
        self._second = max(self._second, second)


def ready_stages(
    done: set[Stage], times: list[int | None], now: float
) -> list[Stage]:
    n = len(times)
    ready: list[Stage] = []

    for i in range(n):
        if ("backbone", i) in done or ("encoder", i) not in done:
            continue
        if i > 0 and ("decoder", i - 1) not in done:
            continue
        ready.append(("backbone", i))

    for i in range(n):
        if ("encoder", i) in done:
            continue
        if i > 0:
            second = times[i]
            if second is None or now < second:
                continue
        ready.append(("encoder", i))

    for i in range(n):
        if ("decoder", i) in done or ("backbone", i) not in done:
            continue
        ready.append(("decoder", i))

    return ready


def _noop(name: str, index: int) -> None:
    return None


def run_pipeline(
    n: int,
    times: list[int | None],
    *,
    clock: PlaybackClock | None = None,
    run_stage: Callable[[str, int], None] | None = None,
) -> list[Stage]:
    if clock is None:
        clock = SimClock()
    if run_stage is None:
        run_stage = _noop

    result: list[Stage] = []
    done: set[Stage] = set()

    for name in ("encoder", "backbone", "decoder"):
        run_stage(name, 0)
        result.append((name, 0))
        done.add((name, 0))

    total = 3 * n
    while len(done) < total:
        ready = ready_stages(done, times, clock.now())
        if ready:
            for name, i in ready:
                run_stage(name, i)
                result.append((name, i))
                done.add((name, i))
            continue

        pending = [
            times[i]
            for i in range(1, n)
            if ("encoder", i) not in done and times[i] is not None
        ]
        clock.wait_until(min(pending))

    return result
