from __future__ import annotations

import threading
from typing import Any

from pydantic import Field

from omni_infinity.serve.models import GenerationRequest
from omni_infinity.serve.service import JobService, SessionLimit


class StreamRequest(GenerationRequest):
    action_script: list[dict[str, Any]] = Field(default_factory=list)


class StreamService:
    def __init__(
        self,
        jobs: JobService,
        *,
        max_sessions: int,
        chunk_frames: int,
        hls: bool,
    ):
        self.jobs = jobs
        self.max_sessions = max_sessions
        self.chunk_frames = chunk_frames
        self.hls = hls
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def open_session(self, request: StreamRequest) -> str:
        self.jobs.validate_request(request)
        first, last = self.jobs.decode_inputs(request)
        with self._lock:
            if len(self._active) >= self.max_sessions:
                raise SessionLimit("stream session limit reached")
            generation_request = GenerationRequest.model_validate(
                request.model_dump(exclude={"action_script"})
            )
            record = self.jobs.store.create(generation_request)
            if first is not None:
                self.jobs._save_image(record.id, "input-first.png", first)
            if last is not None:
                self.jobs._save_image(record.id, "input-last.png", last)
            self._active.add(record.id)
            return record.id

    def finish(self, stream_id: str) -> None:
        with self._lock:
            self._active.discard(stream_id)
