# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Role pipeline: encoder/denoiser/decoder as separate processes.

Each role process loads only its own components fully resident on its
device (``RolePipeline`` chains them with shared queues), receives the
``PipelineState`` via the multiprocessing tensor pickler, moves it to
its device with :func:`~omni_infinity.serve.split.move_state_tensors`,
and runs its block subset.  The encoder additionally builds the state
from the request fields it receives in ``meta``; the decoder returns
the final outputs to the parent.  Split-vs-one-shot parity is gated by
``tests/test_split_parity.py`` and the goldens comparison in
``benchmarks/role_pipeline_bench.py``.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from omni_infinity.serve.roles import Role, StagePayload
from omni_infinity.serve.split import move_state_tensors, run_stage
from omni_infinity.serve.transport import spawn_role_worker

ROLE_COMPONENTS: dict[Role, tuple[str, ...]] = {
    Role.ENCODER: ("text_encoder", "tokenizer", "processor", "vae"),
    Role.DENOISER: ("transformer", "scheduler", "audio_scheduler"),
    Role.DECODER: ("vae", "audio_vae"),
}

_WORKER_RUNNER: dict[str, Any] = {}


def _worker_pipeline(role: Role, config: dict[str, Any]):
    from omni_infinity.runner import ReferenceRunner

    key = role.value
    if key not in _WORKER_RUNNER:
        device = config["device"]
        torch.cuda.set_device(torch.device(device))
        store_components = tuple(
            name
            for name in ("transformer", "vae", "audio_vae")
            if name in ROLE_COMPONENTS[role]
        )
        runner = ReferenceRunner.from_pretrained(
            config["checkpoint"],
            device=device,
            offload=False,
            components=ROLE_COMPONENTS[role],
            store_dir=config.get("store_dir") if store_components else None,
            store_components=store_components,
        )
        _WORKER_RUNNER[key] = runner
    return _WORKER_RUNNER[key].pipeline


def _run_role_stage(role: Role, config: dict[str, Any], payload: StagePayload):
    from omni_infinity.serve.split import prepare_state

    pipeline = _worker_pipeline(role, config)
    device = config["device"]
    if role is Role.ENCODER:
        request = payload.meta.pop("request")
        state = prepare_state(pipeline, **request)
    else:
        state = payload.meta.pop("state")
    move_state_tensors(state, device)
    state = run_stage(pipeline, role, state)
    if role is Role.DECODER:
        outputs = {
            key: state.get(key) for key in ("videos", "audio", "sampling_rate")
        }
        latents = state.get("latents")
        audio_latents = state.get("audio_latents")
        outputs["latents"] = (
            latents.cpu() if torch.is_tensor(latents) else latents
        )
        outputs["audio_latents"] = (
            audio_latents.cpu()
            if torch.is_tensor(audio_latents)
            else audio_latents
        )
        return StagePayload(
            job_id=payload.job_id,
            stage=role,
            tensors={},
            meta={"outputs": outputs},
        )
    torch.cuda.synchronize(torch.device(device))
    return StagePayload(
        job_id=payload.job_id,
        stage=role,
        tensors={},
        meta={"state": state},
    )


def encoder_stage(config, payload):
    return _run_role_stage(Role.ENCODER, config, payload)


def denoiser_stage(config, payload):
    return _run_role_stage(Role.DENOISER, config, payload)


def decoder_stage(config, payload):
    return _run_role_stage(Role.DECODER, config, payload)


_STAGE_FNS = {
    Role.ENCODER: encoder_stage,
    Role.DENOISER: denoiser_stage,
    Role.DECODER: decoder_stage,
}


class RolePipeline:
    """Owns the three chained role processes for one serving profile.

    Bounded queues cap in-flight jobs at roughly ``4 * queue_size``
    (entry + two handoffs + exit) plus the three in-stage jobs;
    ``submit`` blocks past that, so callers must ``collect`` results
    concurrently (a collector thread) or keep submissions under the
    cap — submitting everything before collecting deadlocks.
    """

    def __init__(
        self,
        checkpoint: str,
        device_map: dict[Role, str],
        *,
        store_dir: str | None = None,
        queue_size: int = 2,
    ) -> None:
        for role in (Role.ENCODER, Role.DENOISER, Role.DECODER):
            if role not in device_map:
                raise ValueError(f"device_map is missing {role.value!r}")
        self._workers: list[tuple[Any, Any, Any]] = []
        previous_outbox = None
        try:
            for role in (Role.ENCODER, Role.DENOISER, Role.DECODER):
                config = {
                    "checkpoint": checkpoint,
                    "device": device_map[role],
                    "store_dir": store_dir,
                }
                stage_fn = functools.partial(_STAGE_FNS[role], config)
                process, inbox, outbox = spawn_role_worker(
                    role,
                    device_map[role],
                    stage_fn,
                    queue_size=queue_size,
                    inbox=previous_outbox,
                )
                self._workers.append((process, inbox, outbox))
                previous_outbox = outbox
        except BaseException:
            for process, _, _ in self._workers:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=30)
            raise
        self._entry = self._workers[0][1]
        self._exit = self._workers[-1][2]

    def submit(self, job_id: str, request: dict[str, Any]) -> None:
        self._entry.put(
            StagePayload(
                job_id=job_id,
                stage=Role.ENCODER,
                tensors={},
                meta={"request": request},
            )
        )

    def collect(self, timeout: float | None = None) -> StagePayload:
        return self._exit.get(timeout=timeout)

    def shutdown(self, timeout: float = 120.0) -> None:
        """Drain the chain, then release each worker in order.

        The shutdown sentinel propagates encoder -> denoiser ->
        decoder; each worker echoes it downstream and then waits for a
        release ack on its own inbox (see ``role_worker_loop``).  For
        chained workers the upstream has already exited, so the parent
        — which holds every queue — sends each release itself after
        the exit-side sentinel proves the whole chain drained.
        """

        leftovers = []
        try:
            self._entry.put(None, timeout=timeout)
            while True:
                item = self._exit.get(timeout=timeout)
                if item is None:
                    break
                leftovers.append(item)
            for process, inbox, _ in self._workers:
                inbox.put(None, timeout=timeout)
                process.join(timeout=timeout)
                if process.is_alive():
                    raise RuntimeError("role worker did not exit after release")
        finally:
            for process, _, _ in self._workers:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=timeout)
        if leftovers:
            error = RuntimeError(
                f"RolePipeline.shutdown found {len(leftovers)} uncollected "
                "result(s); collect() every submitted job first"
            )
            error.leftovers = leftovers
            raise error
