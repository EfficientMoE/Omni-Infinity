# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import logging
import time
from collections.abc import Callable
from concurrent.futures import Executor

from PIL import Image

from omni_infinity.demo.models import DemoRecord, DemoRequest
from omni_infinity.demo.pipeline import PlaybackClock, SimClock, run_pipeline
from omni_infinity.demo.schedule import (
    PRESETS,
    SEGMENT_FRAMES,
    playback_frames,
    prompt_times,
    segment_count,
)
from omni_infinity.demo.stitch import last_frame_image, stitch_results
from omni_infinity.demo.store import DemoStore
from omni_infinity.registry import resolve_profile
from omni_infinity.runner import GenerationResult
from omni_infinity.serve.artifacts import (
    _stereo_audio,
    _video_frames,
    write_artifacts,
)
from omni_infinity.serve.models import JobStatus
from omni_infinity.serve.service import (
    InvalidMedia,
    JobService,
    ProfileConflict,
)
from omni_infinity.serve.stream import DemoStreamService, chunk_message
from omni_infinity.streaming import CODEC, MediaChunk, fragment_clip

logger = logging.getLogger(__name__)

StepCallback = Callable[[int, int], None]


class _LivePlaybackClock(PlaybackClock):
    def __init__(self) -> None:
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.monotonic()

    def now(self) -> float:
        if self._t0 is None:
            raise RuntimeError(
                "playback clock has not started; decoder slot 0 must run first"
            )
        return time.monotonic() - self._t0

    def wait_until(self, second: float) -> None:
        while self.now() < second:
            time.sleep(min(0.05, max(0.0, second - self.now())))


class DemoService:
    def __init__(
        self,
        runner,
        model_arch: str,
        *,
        store: DemoStore | None = None,
        executor: Executor | None = None,
        optimizations: tuple[str, ...] = (),
        stream_service: DemoStreamService | None = None,
        stream_chunk_frames: int = 24,
    ):
        self.runner = runner
        self.model_arch = model_arch
        self.store = store
        self.executor = executor
        self.optimizations = optimizations
        self.stream_service = stream_service
        self.stream_chunk_frames = stream_chunk_frames

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
        if first is None:
            raise InvalidMedia("first frame is required")
        record = self.store.create(request)
        path = self.store.input_path(record.id, "input-first.png")
        with Image.open(io.BytesIO(first)) as image:
            image.convert("RGB").save(path, format="PNG")
        if self.stream_service is not None:
            self.stream_service.open_session(record.id)
        self.executor.submit(self._execute, record.id)
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
            result = self._run_pipeline(
                demo_id,
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
            self._publish(
                demo_id,
                {
                    "type": "end",
                    "artifact_url": f"/v1/demos/{demo_id}/artifacts",
                },
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
            self._publish(
                demo_id,
                {"type": "error", "detail": f"{type(exc).__name__}: {exc}"},
            )
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

    def _run_pipeline(
        self,
        demo_id: str,
        request: DemoRequest,
        first: Image.Image,
        *,
        step_callback: StepCallback | None = None,
    ) -> GenerationResult:
        count = len(request.prompts)
        results: list[GenerationResult | None] = [None] * count
        handoffs: dict[int, Image.Image] = {0: first}
        admitted: set[int] = set()
        live_clock = (
            _LivePlaybackClock() if self.stream_service is not None else None
        )
        clock: PlaybackClock = live_clock or SimClock()

        def run_stage(name: str, i: int) -> None:
            if name == "encoder":
                admitted.add(i)
                return
            if name == "backbone":
                if i not in admitted:
                    raise RuntimeError(f"prompt {i} was not admitted")
                results[i] = self._segment(
                    i,
                    request,
                    handoffs[i],
                    step_callback=step_callback,
                )
                return
            if name != "decoder":
                raise RuntimeError(f"unknown pipeline stage: {name}")

            result = results[i]
            if result is None:
                raise RuntimeError(f"clip {i} was not generated")
            if self.stream_service is not None:
                self._publish_clip(demo_id, request, i, result)
                if i == 0:
                    assert live_clock is not None
                    live_clock.start()
            if i + 1 < count:
                handoffs[i + 1] = last_frame_image(result.videos[0])

        times = prompt_times(
            count,
            request.schedule_start,
            request.schedule_end,
            request.seed,
        )
        run_pipeline(count, times, clock=clock, run_stage=run_stage)
        completed = [result for result in results if result is not None]
        if len(completed) != count:
            raise RuntimeError("demo pipeline did not generate every clip")
        return stitch_results(
            completed,
            playback_frames=playback_frames(PRESETS[request.duration]),
        )

    def _publish_clip(
        self,
        demo_id: str,
        request: DemoRequest,
        i: int,
        result: GenerationResult,
    ) -> None:
        nominal_frames = playback_frames(PRESETS[request.duration])
        frame_count = min(SEGMENT_FRAMES, nominal_frames - i * SEGMENT_FRAMES)
        frames = _video_frames(result.videos)[:frame_count]
        sample_count = round(frame_count / 24 * result.sampling_rate)
        audio = _stereo_audio(result.audio)[:, :sample_count]
        init, fragments = fragment_clip(
            frames,
            audio,
            result.sampling_rate,
            chunk_frames=self.stream_chunk_frames,
        )
        self._publish(
            demo_id,
            {
                "type": "init",
                "codec": CODEC,
                "init_b64": base64.b64encode(init).decode("ascii"),
                "clip": i,
            },
        )
        timestamp_offset = i * SEGMENT_FRAMES / 24
        for position, fragment in enumerate(fragments):
            chunk = MediaChunk(
                index=fragment.index,
                pts=timestamp_offset + fragment.pts,
                duration=fragment.duration,
                keyframe=fragment.keyframe,
                video_bytes=fragment.video_bytes,
                audio_bytes=None,
                prompt=request.prompts[i],
                done=(
                    i == len(request.prompts) - 1
                    and position == len(fragments) - 1
                ),
            )
            message = chunk_message(chunk)
            message.update({"clip": i, "timestamp_offset": timestamp_offset})
            self._publish(demo_id, message)

    def _publish(self, demo_id: str, message: dict) -> None:
        if self.stream_service is not None:
            self.stream_service.publish(demo_id, message)

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
            if i + 1 < len(request.prompts):
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
