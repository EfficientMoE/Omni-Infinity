# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from omni_infinity.serve.roles import Role, StagePayload
from omni_infinity.serve.transport import (
    shutdown_role_worker,
    spawn_role_worker,
)


def _double(payload: StagePayload) -> StagePayload:
    return StagePayload(
        job_id=payload.job_id,
        stage=payload.stage,
        tensors={k: v * 2 for k, v in payload.tensors.items()},
        meta=payload.meta,
    )


@pytest.mark.timeout(120)
def test_role_worker_round_trip_and_shutdown():
    process, inbox, outbox = spawn_role_worker(
        Role.ENCODER, "cpu", _double, queue_size=2
    )
    try:
        inbox.put(
            StagePayload(
                job_id="j1",
                stage=Role.ENCODER,
                tensors={"x": torch.ones(3)},
            )
        )
        result = outbox.get(timeout=60)
        assert result.job_id == "j1"
        assert torch.equal(result.tensors["x"], torch.full((3,), 2.0))
        assert result.meta["role"] == "encoder"
        assert result.meta["device"] == "cpu"

        inbox.put(
            StagePayload(
                job_id="j2",
                stage=Role.ENCODER,
                tensors={},
                meta={"role": "stale", "device": "stale"},
            )
        )
        overwritten = outbox.get(timeout=60)
        assert overwritten.meta["role"] == "encoder"
        assert overwritten.meta["device"] == "cpu"

        shutdown_role_worker(process, inbox, outbox, timeout=60)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        for q in (inbox, outbox):
            q.close()
            q.join_thread()


@pytest.mark.timeout(120)
def test_shutdown_with_undrained_output_raises_and_recovers():
    process, inbox, outbox = spawn_role_worker(
        Role.DECODER, "cpu", _double, queue_size=2
    )
    try:
        inbox.put(
            StagePayload(
                job_id="pending",
                stage=Role.DECODER,
                tensors={"x": torch.ones(2)},
            )
        )
        with pytest.raises(RuntimeError, match="undrained") as excinfo:
            shutdown_role_worker(process, inbox, outbox, timeout=60)
        leftovers = excinfo.value.leftovers
        assert [p.job_id for p in leftovers] == ["pending"]
        assert torch.equal(leftovers[0].tensors["x"], torch.full((2,), 2.0))
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        for q in (inbox, outbox):
            q.close()
            q.join_thread()


@pytest.mark.timeout(120)
def test_shutdown_drains_when_bounded_queues_are_full():
    process, inbox, outbox = spawn_role_worker(
        Role.DENOISER, "cpu", _double, queue_size=1
    )
    try:
        for index in range(3):
            inbox.put(
                StagePayload(
                    job_id=f"j{index}",
                    stage=Role.DENOISER,
                    tensors={"x": torch.ones(1)},
                ),
                timeout=60,
            )
        with pytest.raises(RuntimeError, match="undrained") as excinfo:
            shutdown_role_worker(process, inbox, outbox, timeout=60)
        assert sorted(p.job_id for p in excinfo.value.leftovers) == [
            "j0",
            "j1",
            "j2",
        ]
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        for q in (inbox, outbox):
            q.close()
            q.join_thread()
