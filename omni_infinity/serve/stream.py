# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

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
_CLOSED = object()
StreamSource = ClipChunker | NativeChunker


class ChunkPipe:
    """Bounded async hand-off that preserves every generated fragment."""

    def __init__(self, maxsize: int = MAX_PENDING_CHUNKS) -> None:
        self._queue: asyncio.Queue[MediaChunk | object] = asyncio.Queue(
            maxsize=max(1, int(maxsize))
        )
        self._error: str | None = None
        self._aborted = False

    @property
    def aborted(self) -> bool:
        return self._aborted

    async def put(self, chunk: MediaChunk) -> None:
        if self._aborted:
            raise StreamAborted("stream pipe is closed")
        await self._queue.put(chunk)
        if self._aborted:
            raise StreamAborted("stream pipe is closed")

    async def close(self, error: str | None = None) -> None:
        self._error = error
        await self._queue.put(_CLOSED)

    def abort(self) -> None:
        self._aborted = True
        while not self._queue.empty():
            self._queue.get_nowait()
        self._queue.put_nowait(_CLOSED)

    async def drain(self) -> tuple[list[MediaChunk], bool, str | None]:
        item = await self._queue.get()
        if item is _CLOSED:
            return [], True, self._error
        return [item], False, None


class StreamAborted(RuntimeError):
    pass


class StreamGenerationError(RuntimeError):
    pass


class TerminalPersistenceError(RuntimeError):
    pass


class StreamSession:
    def __init__(self, stream_id: str, request: StreamRequest) -> None:
        self.stream_id = stream_id
        self.request = request
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
            try:
                store.transition(
                    self.stream_id, status, error=error, artifacts=artifacts
                )
            except JobStoreError:
                logger.exception(
                    "could not finish stream %s as %s", self.stream_id, status
                )
                return False
            self._terminal = True
        return True


def _init_message(session: StreamSession, source: StreamSource) -> dict:
    if not source.init:
        raise RuntimeError("chunk source produced no initialization segment")
    return {
        "type": "init",
        "codec": CODEC,
        "init_b64": base64.b64encode(source.init).decode("ascii"),
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
        self.artifact_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="stream-artifact"
        )
        self._sessions: dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def shutdown(self) -> None:
        self.artifact_executor.shutdown(wait=True, cancel_futures=True)

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
            assert detail is not None
            await _send_error(websocket, detail)
            await _close(websocket)
            return

        pipe = ChunkPipe(self.queue_chunks)
        source = self._make_source(session)
        reader = asyncio.create_task(self._read_inputs(websocket, pipe))
        feeder: asyncio.Task[None] | None = None
        try:
            chunks = await self._generate(session, source, pipe)
            feeder = asyncio.create_task(self._feed(pipe, chunks))
            await self._pump(websocket, session, source, pipe)
        except WebSocketDisconnect:
            logger.info("stream %s: client disconnected", stream_id)
        except StreamAborted:
            logger.info("stream %s: client disconnected", stream_id)
        except StreamGenerationError as exc:
            await _send_error(websocket, str(exc))
        except RuntimeError as exc:
            if _is_closed_socket_error(exc):
                logger.info("stream %s: client socket closed", stream_id)
            else:
                logger.exception("stream %s: socket failed", stream_id)
        except Exception:
            logger.exception("stream %s: socket failed", stream_id)
        finally:
            pipe.abort()
            if feeder is not None:
                feeder.cancel()
                with contextlib.suppress(asyncio.CancelledError, StreamAborted):
                    await feeder
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

    def _make_source(self, session: StreamSession) -> StreamSource:
        if session.request.source == "native":
            return NativeChunker(chunk_frames=self.chunk_frames)
        return ClipChunker(
            self.jobs.runner,
            arch=self.jobs.profile.model_arch,
            chunk_frames=self.chunk_frames,
        )

    async def _generate(
        self,
        session: StreamSession,
        source: StreamSource,
        pipe: ChunkPipe,
    ) -> list[MediaChunk]:
        try:
            future = self.jobs.executor.submit(
                self._collect_chunks, session, source, pipe
            )
        except RuntimeError as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self._fail(session, detail)
            raise StreamGenerationError(detail) from exc
        try:
            return await asyncio.wrap_future(future)
        except StreamAborted:
            raise
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            logger.exception("stream %s: generation failed", session.stream_id)
            self._fail(session, detail)
            raise StreamGenerationError(detail) from exc

    def _collect_chunks(
        self,
        session: StreamSession,
        source: StreamSource,
        pipe: ChunkPipe,
    ) -> list[MediaChunk]:
        """Finish GPU-backed generation before bounded async delivery."""
        stream_id = session.stream_id
        if pipe.aborted:
            raise StreamAborted(stream_id)
        self.jobs.store.transition(stream_id, JobStatus.RUNNING)
        first = self.jobs.load_image(stream_id, "input-first.png")
        last = self.jobs.load_image(stream_id, "input-last.png")
        chunks = list(
            source.iter_chunks(
                session.request,
                step_callback=lambda completed, total: self._report_progress(
                    session, pipe, completed, total
                ),
                image=first,
                last_image=last,
            )
        )
        if pipe.aborted:
            raise StreamAborted(stream_id)
        if not chunks:
            raise RuntimeError("chunk source produced no media fragments")
        if not source.init:
            raise RuntimeError(
                "chunk source produced no initialization segment"
            )
        return chunks

    def _report_progress(
        self,
        session: StreamSession,
        pipe: ChunkPipe,
        completed: int,
        total: int,
    ) -> None:
        if pipe.aborted:
            raise StreamAborted(session.stream_id)
        try:
            self.jobs.store.update_progress(session.stream_id, completed, total)
        except JobStoreError as exc:
            raise StreamAborted(session.stream_id) from exc

    async def _feed(self, pipe: ChunkPipe, chunks: list[MediaChunk]) -> None:
        for chunk in chunks:
            await pipe.put(chunk)
        await pipe.close()

    async def _pump(
        self,
        websocket: WebSocket,
        session: StreamSession,
        source: StreamSource,
        pipe: ChunkPipe,
    ) -> None:
        sent_init = False
        failure: str | None = None
        while True:
            chunks, closed, failure = await pipe.drain()
            for chunk in chunks:
                if not sent_init:
                    await websocket.send_json(_init_message(session, source))
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
        await self._publish(websocket, session, source)

    async def _publish(
        self,
        websocket: WebSocket,
        session: StreamSession,
        source: StreamSource,
    ) -> None:
        """Write the job artifact only after the client has the fragments."""
        try:
            future = self.artifact_executor.submit(
                self._write_artifact, session, source
            )
        except RuntimeError as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self._fail(session, detail)
            await _send_error(websocket, detail)
            return
        try:
            await asyncio.wrap_future(future)
        except TerminalPersistenceError as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self._fail(session, detail, remove_artifacts=False)
            await _send_error(websocket, detail)
            return
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

    def _write_artifact(
        self, session: StreamSession, source: StreamSource
    ) -> None:
        result = source.result
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
            raise TerminalPersistenceError(
                "could not persist succeeded stream status"
            )

    def _fail(
        self,
        session: StreamSession,
        detail: str,
        *,
        remove_artifacts: bool = True,
    ) -> None:
        store = self.jobs.store
        if remove_artifacts:
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

    async def _read_inputs(self, websocket: WebSocket, pipe: ChunkPipe) -> None:
        """Accept and ignore client messages; Phase 2 consumes them."""
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        pipe.abort()


async def _send_error(websocket: WebSocket, detail: str) -> None:
    with contextlib.suppress(Exception):
        await websocket.send_json({"type": "error", "detail": detail})


async def _close(websocket: WebSocket) -> None:
    with contextlib.suppress(Exception):
        await websocket.close()


def _is_closed_socket_error(exc: RuntimeError) -> bool:
    detail = str(exc).lower()
    return (
        "once a close message has been sent" in detail
        or "after sending 'websocket.close'" in detail
        or "websocket is not connected" in detail
    )
