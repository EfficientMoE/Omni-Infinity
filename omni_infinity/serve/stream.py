# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import threading
from collections import deque
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from omni_infinity.serve.artifacts import write_artifacts
from omni_infinity.serve.models import (
    TERMINAL_STATUSES,
    ArtifactMetadata,
    GenerationRequest,
    JobStatus,
)
from omni_infinity.serve.service import JobService, SessionLimit
from omni_infinity.serve.store import JobStore, JobStoreError
from omni_infinity.streaming import (
    CODEC,
    ClipChunker,
    MediaChunk,
    NativeChunker,
    StreamRequest,
)

logger = logging.getLogger(__name__)

MAX_PENDING_CHUNKS = 8


class ChunkPipe:
    """Bounded hand-off between the GPU worker and the socket sender.

    The producer never blocks: once the buffer is full it drops fragments
    until the next keyframe, so a slow client cannot hold the single job
    executor. Chunk zero and the final chunk are never dropped because a
    client cannot start without the first fragment and cannot learn the
    stream ended without the last one.
    """

    def __init__(
        self, loop: asyncio.AbstractEventLoop, maxsize: int = MAX_PENDING_CHUNKS
    ) -> None:
        self._loop = loop
        self._maxsize = max(1, int(maxsize))
        self._lock = threading.Lock()
        self._ready = asyncio.Event()
        self._items: deque[MediaChunk] = deque()
        self._closed = False
        self._error: str | None = None
        self._aborted = False
        self._resyncing = False
        self.dropped = 0

    @property
    def aborted(self) -> bool:
        with self._lock:
            return self._aborted

    def put(self, chunk: MediaChunk) -> None:
        with self._lock:
            if self._aborted or self._closed:
                return
            if not (chunk.index == 0 or chunk.done):
                if len(self._items) >= self._maxsize:
                    self._resyncing = True
                if self._resyncing:
                    if not chunk.keyframe or len(self._items) >= self._maxsize:
                        self.dropped += 1
                        return
                    self._resyncing = False
            self._items.append(chunk)
        self._wake()

    def close(self, error: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._error = error
        self._wake()

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            self._closed = True
            self._items.clear()
        self._wake()

    async def drain(self) -> tuple[list[MediaChunk], bool, str | None]:
        await self._ready.wait()
        with self._lock:
            items = list(self._items)
            self._items.clear()
            closed, error = self._closed, self._error
            if not closed:
                self._ready.clear()
        return items, closed, error

    def _wake(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._ready.set)
        except RuntimeError:
            logger.debug("stream event loop is gone; dropping wakeup")


class StreamAborted(RuntimeError):
    pass


class StreamSession:
    def __init__(self, stream_id: str, request: StreamRequest) -> None:
        self.stream_id = stream_id
        self.request = request
        self.source: Any = None
        self.pipe: ChunkPipe | None = None
        self.connected = False
        self._lock = threading.Lock()
        self._terminal = False

    def mark_terminal(
        self,
        store: JobStore,
        status: JobStatus,
        *,
        error: str | None = None,
        artifacts: ArtifactMetadata | None = None,
    ) -> bool:
        with self._lock:
            if self._terminal:
                return False
            self._terminal = True
        try:
            store.transition(
                self.stream_id, status, error=error, artifacts=artifacts
            )
        except JobStoreError:
            logger.exception(
                "could not finish stream %s as %s", self.stream_id, status
            )
            return False
        return True


def _init_message(session: StreamSession) -> dict:
    init = getattr(session.source, "init", None) or b""
    return {
        "type": "init",
        "codec": CODEC,
        "init_b64": base64.b64encode(init).decode("ascii"),
        "action_script": [
            {"t": cue.t, "action": cue.action, "instruction": cue.instruction}
            for cue in session.request.action_script
        ],
    }


def _chunk_message(chunk: MediaChunk) -> dict:
    return {
        "type": "chunk",
        "index": chunk.index,
        "pts": chunk.pts,
        "duration": chunk.duration,
        "keyframe": chunk.keyframe,
        "video_b64": _encode(chunk.video_bytes),
        "audio_b64": _encode(chunk.audio_bytes),
        "prompt": chunk.prompt,
        "instruction": chunk.instruction,
        "action": chunk.action,
        "done": chunk.done,
    }


def _encode(payload: bytes | None) -> str | None:
    if payload is None:
        return None
    return base64.b64encode(payload).decode("ascii")


class StreamService:
    def __init__(
        self,
        jobs: JobService,
        *,
        max_sessions: int,
        chunk_frames: int,
        hls: bool,
        queue_chunks: int = MAX_PENDING_CHUNKS,
    ):
        self.jobs = jobs
        self.max_sessions = max_sessions
        self.chunk_frames = chunk_frames
        self.hls = hls
        self.queue_chunks = queue_chunks
        self._sessions: dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def open_session(self, request: StreamRequest) -> str:
        self.jobs.validate_request(request)
        first, last = self.jobs.decode_inputs(request)
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise SessionLimit("stream session limit reached")
            generation_request = GenerationRequest.model_validate(
                request.model_dump(exclude={"action_script", "source"})
            )
            record = self.jobs.store.create(generation_request)
            if first is not None:
                self.jobs.save_image(record.id, "input-first.png", first)
            if last is not None:
                self.jobs.save_image(record.id, "input-last.png", last)
            self._sessions[record.id] = StreamSession(record.id, request)
            return record.id

    def finish(self, stream_id: str) -> None:
        with self._lock:
            self._sessions.pop(stream_id, None)

    async def run_socket(self, websocket: WebSocket, stream_id: str) -> None:
        await websocket.accept()
        session, detail = self._claim(stream_id)
        if session is None:
            await _send_error(websocket, detail)
            await _close(websocket)
            return

        session.pipe = ChunkPipe(asyncio.get_running_loop(), self.queue_chunks)
        session.source = self._make_source(session)
        reader = asyncio.create_task(self._read_inputs(websocket, session))
        try:
            self._start_worker(session)
            await self._pump(websocket, session)
        except WebSocketDisconnect:
            logger.info("stream %s: client disconnected", stream_id)
        except Exception:
            logger.exception("stream %s: socket failed", stream_id)
        finally:
            session.pipe.abort()
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
            self._cancel_unfinished(session)
            self.finish(stream_id)
            await _close(websocket)

    def _claim(self, stream_id: str) -> tuple[StreamSession | None, str | None]:
        with self._lock:
            session = self._sessions.get(stream_id)
            if session is None:
                return None, "unknown stream session"
            if session.connected:
                return None, "stream session already has a player"
            session.connected = True
            return session, None

    def _make_source(self, session: StreamSession):
        if session.request.source == "native":
            return NativeChunker(chunk_frames=self.chunk_frames)
        return ClipChunker(
            self.jobs.runner,
            arch=self.jobs.profile.model_arch,
            chunk_frames=self.chunk_frames,
        )

    def _start_worker(self, session: StreamSession) -> None:
        assert session.pipe is not None
        try:
            future = self.jobs.executor.submit(self._produce, session)
        except RuntimeError as exc:
            session.pipe.close(error=f"{type(exc).__name__}: {exc}")
            return
        future.add_done_callback(
            lambda done: (
                session.pipe.close(error="stream worker was cancelled")
                if done.cancelled()
                else None
            )
        )

    def _produce(self, session: StreamSession) -> None:
        """Run the chunk iterator on the single job executor.

        The worker never waits on the socket: it hands each fragment to the
        bounded pipe and returns as soon as generation ends, so queued jobs
        are not held behind network I/O.
        """
        pipe = session.pipe
        stream_id = session.stream_id
        try:
            if pipe.aborted:
                raise StreamAborted(stream_id)
            self.jobs.store.transition(stream_id, JobStatus.RUNNING)
            first = self.jobs.load_image(stream_id, "input-first.png")
            last = self.jobs.load_image(stream_id, "input-last.png")
            for chunk in session.source.iter_chunks(
                session.request,
                step_callback=lambda completed, total: self._report_progress(
                    session, completed, total
                ),
                image=first,
                last_image=last,
            ):
                if pipe.aborted:
                    raise StreamAborted(stream_id)
                pipe.put(chunk)
            pipe.close()
        except StreamAborted:
            logger.info("stream %s: worker stopped early", stream_id)
            pipe.close()
        except Exception as exc:
            logger.exception("stream %s: generation failed", stream_id)
            pipe.close(error=f"{type(exc).__name__}: {exc}")

    def _report_progress(
        self, session: StreamSession, completed: int, total: int
    ) -> None:
        if session.pipe.aborted:
            raise StreamAborted(session.stream_id)
        try:
            self.jobs.store.update_progress(session.stream_id, completed, total)
        except JobStoreError as exc:
            raise StreamAborted(session.stream_id) from exc

    async def _pump(self, websocket: WebSocket, session: StreamSession) -> None:
        pipe = session.pipe
        sent_init = False
        failure: str | None = None
        while True:
            chunks, closed, failure = await pipe.drain()
            for chunk in chunks:
                if not sent_init:
                    await websocket.send_json(_init_message(session))
                    sent_init = True
                await websocket.send_json(_chunk_message(chunk))
            if closed:
                break
        if pipe.aborted:
            return
        if failure is not None:
            self._fail(session, failure)
            await _send_error(websocket, failure)
            return
        await self._publish(websocket, session)

    async def _publish(
        self, websocket: WebSocket, session: StreamSession
    ) -> None:
        """Write the job artifact only after the client has the fragments."""
        try:
            future = self.jobs.executor.submit(self._write_artifact, session)
        except RuntimeError as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self._fail(session, detail)
            await _send_error(websocket, detail)
            return
        try:
            await asyncio.wrap_future(future)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self._fail(session, detail)
            await _send_error(websocket, detail)
            return
        await websocket.send_json(
            {
                "type": "end",
                "artifact_url": f"/v1/jobs/{session.stream_id}/artifacts",
            }
        )

    def _write_artifact(self, session: StreamSession) -> None:
        result = getattr(session.source, "result", None)
        if result is None:
            raise RuntimeError("chunk source produced no generation result")
        stream_id = session.stream_id
        artifacts = write_artifacts(
            result,
            self.jobs.store.job_dir(stream_id),
            artifact_url=f"/v1/jobs/{stream_id}/artifacts",
        )
        if not session.mark_terminal(
            self.jobs.store, JobStatus.SUCCEEDED, artifacts=artifacts
        ):
            raise RuntimeError("stream session already finished")

    def _fail(self, session: StreamSession, detail: str) -> None:
        store = self.jobs.store
        store.video_path(session.stream_id).unlink(missing_ok=True)
        store.audio_path(session.stream_id).unlink(missing_ok=True)
        session.mark_terminal(store, JobStatus.FAILED, error=detail)

    def _cancel_unfinished(self, session: StreamSession) -> None:
        try:
            record = self.jobs.store.get(session.stream_id)
        except JobStoreError:
            logger.exception("stream %s: record is gone", session.stream_id)
            return
        if record.status in TERMINAL_STATUSES:
            return
        session.mark_terminal(self.jobs.store, JobStatus.CANCELLED)

    async def _read_inputs(
        self, websocket: WebSocket, session: StreamSession
    ) -> None:
        """Accept and ignore client messages; Phase 2 consumes them."""
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        session.pipe.abort()


async def _send_error(websocket: WebSocket, detail: str) -> None:
    with contextlib.suppress(Exception):
        await websocket.send_json({"type": "error", "detail": detail})


async def _close(websocket: WebSocket) -> None:
    with contextlib.suppress(Exception):
        await websocket.close()
