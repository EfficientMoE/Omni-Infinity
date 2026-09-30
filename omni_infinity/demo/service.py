# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from concurrent.futures import Executor, Future

from PIL import Image

from omni_infinity.demo.models import DemoRecord, DemoRequest
from omni_infinity.demo.schedule import (
    PRESETS,
    SEGMENT_FRAMES,
    playback_frames,
    segment_count,
)
from omni_infinity.demo.stitch import last_frame_image, stitch_results
from omni_infinity.demo.store import DemoStore
from omni_infinity.registry import resolve_profile
from omni_infinity.runner import GenerationResult
from omni_infinity.serve.artifacts import write_artifacts
from omni_infinity.serve.models import JobStatus
from omni_infinity.serve.service import (
    InvalidMedia,
    JobService,
    ProfileConflict,
)

logger = logging.getLogger(__name__)

StepCallback = Callable[[int, int], None]


class DemoService:
    def __init__(
        self,
        runner,
        model_arch: str,
        *,
        store: DemoStore | None = None,
        executor: Executor | None = None,
        optimizations: tuple[str, ...] = (),
    ):
        self.runner = runner
        self.model_arch = model_arch
        self.store = store
        self.executor = executor
        self.optimizations = optimizations
        self._futures: dict[str, Future] = {}

    def submit(self, request: DemoRequest) -> DemoRecord:
        if self.store is None or self.executor is None:
            raise RuntimeError("demo submission is not configured")
        try:
            requested = resolve_profile(
                request.model_arch, request.optimizations
            )
        except ValueError as exc:
            raise ProfileConflict(str(exc)) from exc
        if (
            requested.model_arch != self.model_arch
            or requested.optimizations != self.optimizations
        ):
            raise ProfileConflict(
                "request profile does not match the loaded server profile"
            )
        if not request.first_frame_base64:
            raise InvalidMedia("first frame is required")
        first = JobService._decode_image(
            request.first_frame_base64, "first frame"
        )
        assert first is not None
        record = self.store.create(request)
        path = self.store.input_path(record.id, "input-first.png")
        with Image.open(io.BytesIO(first)) as image:
            image.convert("RGB").save(path, format="PNG")
        self._futures[record.id] = self.executor.submit(
            self._execute, record.id
        )
        return record

    def _execute(self, demo_id: str) -> None:
        assert self.store is not None
        try:
            record = self.store.transition(demo_id, JobStatus.RUNNING)
            with Image.open(
                self.store.input_path(demo_id, "input-first.png")
            ) as image:
                image.load()
                first = image.copy()
            result = self.run(
                record.request,
                first,
                step_callback=lambda completed, total: (
                    self.store.update_progress(demo_id, completed, total)
                ),
            )
            artifacts = write_artifacts(
                result,
                self.store.job_dir(demo_id),
                artifact_url=f"/v1/demos/{demo_id}/artifacts",
            )
            self.store.transition(
                demo_id, JobStatus.SUCCEEDED, artifacts=artifacts
            )
        except Exception as exc:
            for name in (
                "output.mp4",
                "output.wav",
                "output.partial.mp4",
                "output.partial.wav",
            ):
                self.store.input_path(demo_id, name).unlink(missing_ok=True)
            logger.exception("demo %s failed", demo_id)
            try:
                record = self.store.get(demo_id)
                if record.status == JobStatus.RUNNING:
                    self.store.transition(
                        demo_id,
                        JobStatus.FAILED,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            except Exception:
                logger.exception(
                    "could not persist failure for demo %s", demo_id
                )

    def run(
        self,
        request: DemoRequest,
        first: Image.Image,
        *,
        step_callback: StepCallback | None = None,
    ) -> GenerationResult:
        results: list[GenerationResult] = []
        frame: Image.Image = first
        for i in range(len(request.prompts)):
            result = self._segment(
                i, request, frame, step_callback=step_callback
            )
            results.append(result)
            frame = last_frame_image(result.videos[0])
        return stitch_results(
            results,
            playback_frames=playback_frames(PRESETS[request.duration]),
        )

    def _segment(
        self,
        i: int,
        request: DemoRequest,
        image: Image.Image,
        *,
        step_callback: StepCallback | None = None,
    ) -> GenerationResult:
        total = segment_count(PRESETS[request.duration])
        total_steps = total * request.num_inference_steps

        inner: StepCallback | None = None
        if step_callback is not None:

            def inner(completed: int, _segment_total: int) -> None:
                step_callback(
                    i * request.num_inference_steps + completed, total_steps
                )

        common = {
            "seed": request.seed + i,
            "num_frames": SEGMENT_FRAMES,
            "image": image,
            "last_image": None,
            "step_callback": inner,
        }
        if self.model_arch == "h3-dense":
            return self.runner.generate(
                request.prompts[i],
                num_inference_steps=request.num_inference_steps,
                resolution=request.resolution,
                **common,
            )
        if request.resolution != "768p":
            raise ValueError("vdn-hybrid requires the 768p canvas")
        return self.runner.generate(
            request.prompts[i],
            num_evaluations=request.num_inference_steps,
            **common,
        )
