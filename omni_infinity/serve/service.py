# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import binascii
import io
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from omni_infinity.registry import ResolvedProfile, resolve_profile
from omni_infinity.serve.artifacts import write_artifacts
from omni_infinity.serve.models import (
    GenerationRequest,
    GraphTelemetry,
    JobRecord,
    JobStatus,
)
from omni_infinity.serve.store import InvalidTransition, JobStore

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _graph_telemetry_delta(
    before: GraphTelemetry, after: GraphTelemetry
) -> GraphTelemetry:
    fallback_reasons = {
        reason: count - before.fallback_reasons.get(reason, 0)
        for reason, count in after.fallback_reasons.items()
        if count - before.fallback_reasons.get(reason, 0) > 0
    }
    return GraphTelemetry(
        captures=after.captures - before.captures,
        replays=after.replays - before.replays,
        capture_failures=after.capture_failures - before.capture_failures,
        graph_pool_bytes=after.graph_pool_bytes,
        capture_time_ms=after.capture_time_ms - before.capture_time_ms,
        warmup_calls=after.warmup_calls - before.warmup_calls,
        capture_warmup_calls=(
            after.capture_warmup_calls - before.capture_warmup_calls
        ),
        graphs=after.graphs,
        generation=after.generation,
        fallback_reasons=fallback_reasons,
    )


class JobServiceError(RuntimeError):
    pass


class ProfileConflict(JobServiceError):
    pass


class Ref2VANotImplemented(JobServiceError):
    pass


class SessionLimit(JobServiceError):
    pass


class InvalidMedia(JobServiceError, ValueError):
    pass


class JobService:
    def __init__(
        self,
        runner,
        profile: ResolvedProfile,
        store: JobStore,
    ):
        self.runner = runner
        self.profile = profile
        self.store = store
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="omni-job"
        )
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, request: GenerationRequest) -> JobRecord:
        self.validate_request(request)
        first, last = self.decode_inputs(request)
        record = self.store.create(request)
        if first is not None:
            self.save_image(record.id, "input-first.png", first)
        if last is not None:
            self.save_image(record.id, "input-last.png", last)
        future = self.executor.submit(self._execute, record.id)
        with self._lock:
            self._futures[record.id] = future
        return record

    def validate_request(self, request: GenerationRequest) -> None:
        if request.type == "ref2va":
            raise Ref2VANotImplemented(
                "Ref2VA requires Task 3 (issue #8), which is not merged"
            )
        try:
            requested = resolve_profile(
                request.model_arch, request.optimizations
            )
        except ValueError as exc:
            raise ProfileConflict(str(exc)) from exc
        if (
            requested.model_arch != self.profile.model_arch
            or requested.optimizations != self.profile.optimizations
        ):
            raise ProfileConflict(
                "request profile does not match the loaded server profile"
            )

    def decode_inputs(
        self, request: GenerationRequest
    ) -> tuple[bytes | None, bytes | None]:
        return (
            self._decode_image(request.first_frame_base64, "first frame"),
            self._decode_image(request.last_frame_base64, "last frame"),
        )

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            futures = tuple(self._futures.items())
        for job_id, future in futures:
            if not future.cancelled():
                continue
            try:
                self.store.transition(job_id, JobStatus.CANCELLED)
            except InvalidTransition:
                pass

    def _execute(self, job_id: str) -> None:
        try:
            record = self.store.transition(job_id, JobStatus.RUNNING)
            first = self.load_image(job_id, "input-first.png")
            last = self.load_image(job_id, "input-last.png")
            manager = getattr(self.runner, "cuda_graph_manager", None)
            telemetry_before = self._graph_telemetry_snapshot(manager, job_id)
            result = self._generate(record.request, first, last, job_id)
            artifacts = write_artifacts(
                result,
                self.store.job_dir(job_id),
                artifact_url=f"/v1/jobs/{job_id}/artifacts",
            )
            telemetry = None
            if telemetry_before is not None:
                telemetry_after = self._graph_telemetry_snapshot(
                    manager, job_id
                )
                if telemetry_after is not None:
                    telemetry = _graph_telemetry_delta(
                        telemetry_before, telemetry_after
                    )
            self.store.transition(
                job_id,
                JobStatus.SUCCEEDED,
                artifacts=artifacts,
                graph_telemetry=telemetry,
            )
        except Exception as exc:
            self.store.video_path(job_id).unlink(missing_ok=True)
            self.store.audio_path(job_id).unlink(missing_ok=True)
            logger.exception("job %s failed", job_id)
            try:
                record = self.store.get(job_id)
                if record.status == JobStatus.RUNNING:
                    self.store.transition(
                        job_id,
                        JobStatus.FAILED,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            except Exception:
                logger.exception("could not persist failure for job %s", job_id)

    @staticmethod
    def _graph_telemetry_snapshot(manager, job_id: str):
        if manager is None:
            return None
        try:
            return GraphTelemetry(**manager.stats_snapshot())
        except Exception:
            logger.exception(
                "could not capture CUDA graph telemetry for job %s", job_id
            )
            return None

    def _generate(self, request, first, last, job_id: str):
        common = {
            "seed": request.seed,
            "num_frames": request.num_frames,
            "image": first,
            "last_image": last,
            "step_callback": lambda completed, total: (
                self.store.update_progress(job_id, completed, total)
            ),
        }
        if self.profile.model_arch == "h3-dense":
            return self.runner.generate(
                request.prompt,
                num_inference_steps=request.num_inference_steps,
                resolution=request.resolution,
                **common,
            )
        if request.resolution != "768p":
            raise ValueError("vdn-hybrid requires the 768p canvas")
        return self.runner.generate(
            request.prompt,
            num_evaluations=request.num_inference_steps,
            **common,
        )

    @staticmethod
    def _decode_image(value: str | None, label: str) -> bytes | None:
        if value is None:
            return None
        try:
            payload = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidMedia(f"invalid base64 for {label}") from exc
        if len(payload) > _MAX_IMAGE_BYTES:
            raise InvalidMedia(f"{label} exceeds the 20 MiB limit")
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise InvalidMedia(f"invalid image for {label}") from exc
        return payload

    def save_image(self, job_id: str, name: str, payload: bytes) -> None:
        path = self.store.input_path(job_id, name)
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.convert("RGB").save(path, format="PNG")
        except (OSError, UnidentifiedImageError) as exc:
            raise InvalidMedia(f"could not save {name}") from exc

    def load_image(self, job_id: str, name: str):
        path: Path = self.store.input_path(job_id, name)
        if not path.exists():
            return None
        with Image.open(path) as image:
            image.load()
            return image.copy()
