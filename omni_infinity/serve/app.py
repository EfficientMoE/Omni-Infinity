# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, WebSocket
from fastapi.responses import FileResponse

from omni_infinity.registry import resolve_profile
from omni_infinity.serve.models import (
    GenerationRequest,
    JobResponse,
    JobStatus,
)
from omni_infinity.serve.service import (
    InvalidMedia,
    JobService,
    ProfileConflict,
    Ref2VANotImplemented,
    SessionLimit,
)
from omni_infinity.serve.store import JobNotFound, JobStore

_VRAM_UNITS = (
    ("gib", 1024**3),
    ("gb", 10**9),
    ("mib", 1024**2),
    ("mb", 10**6),
)
_WEBUI_DIR = Path(__file__).with_name("webui")


def parse_bytes(value: str) -> int:
    lowered = value.strip().lower()
    for suffix, multiplier in _VRAM_UNITS:
        if lowered.endswith(suffix):
            return int(float(lowered[: -len(suffix)]) * multiplier)
    return int(lowered)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServerSettings:
    jobs_dir: Path = Path("./jobs")
    model_arch: str = "h3-dense"
    optimizations: tuple[str, ...] = (
        "adaln-host-cache",
        "block-stream",
        "text-encoder-stream",
    )
    checkpoint: str | None = None
    device: str = "cuda"
    store_dir: str | None = None
    store_components: tuple[str, ...] = ("transformer", "vae", "audio_vae")
    max_vram: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    stream_enabled: bool = False
    stream_max_sessions: int = 1
    stream_session_ttl: float = 30.0
    stream_chunk_frames: int = 24
    stream_queue_chunks: int = 8
    stream_fallback_hls: bool = False

    @classmethod
    def from_env(cls) -> ServerSettings:
        optimizations = os.environ.get(
            "OMNI_OPTIMIZATIONS",
            "adaln-host-cache,block-stream,text-encoder-stream",
        )
        components = os.environ.get(
            "OMNI_STORE_COMPONENTS", "transformer,vae,audio_vae"
        )
        return cls(
            jobs_dir=Path(os.environ.get("OMNI_JOBS_DIR", "./jobs")),
            model_arch=os.environ.get("OMNI_MODEL_ARCH", "h3-dense"),
            optimizations=tuple(
                value.strip()
                for value in optimizations.split(",")
                if value.strip()
            ),
            checkpoint=os.environ.get("OMNI_CHECKPOINT") or None,
            device=os.environ.get("OMNI_DEVICE", "cuda"),
            store_dir=os.environ.get("OMNI_STORE_DIR") or None,
            store_components=tuple(
                value.strip()
                for value in components.split(",")
                if value.strip()
            ),
            max_vram=os.environ.get("OMNI_MAX_VRAM") or None,
            host=os.environ.get("OMNI_HOST", "127.0.0.1"),
            port=int(os.environ.get("OMNI_PORT", "8000")),
            workers=int(os.environ.get("OMNI_WORKERS", "1")),
            stream_enabled=parse_bool(
                os.environ.get("OMNI_STREAM_ENABLED", "false")
            ),
            stream_max_sessions=int(
                os.environ.get("OMNI_STREAM_MAX_SESSIONS", "1")
            ),
            stream_session_ttl=float(
                os.environ.get("OMNI_STREAM_SESSION_TTL", "30")
            ),
            stream_chunk_frames=int(
                os.environ.get("OMNI_STREAM_CHUNK_FRAMES", "24")
            ),
            stream_queue_chunks=int(
                os.environ.get("OMNI_STREAM_QUEUE_CHUNKS", "8")
            ),
            stream_fallback_hls=parse_bool(
                os.environ.get("OMNI_STREAM_FALLBACK_HLS", "false")
            ),
        )


def load_runner(settings: ServerSettings):
    profile = resolve_profile(
        settings.model_arch, settings.optimizations, settings.checkpoint
    )
    kwargs = dict(profile.runner_kwargs)
    if profile.model_arch == "h3-dense":
        if settings.store_dir is not None:
            kwargs["store_dir"] = settings.store_dir
            kwargs["store_components"] = settings.store_components
        if settings.max_vram is not None:
            import torch

            if torch.cuda.is_available():
                device_index = (
                    int(settings.device.partition(":")[2])
                    if ":" in settings.device
                    else torch.cuda.current_device()
                )
                total = torch.cuda.get_device_properties(
                    device_index
                ).total_memory
                if total > 24 * 1024**3:
                    margin = max(total - parse_bytes(settings.max_vram), 0)
                    kwargs["offload_memory_margin"] = f"{margin / 1e9:.1f}GB"
    else:
        kwargs["workflow"] = "fl2va"
    runner = profile.runner.from_pretrained(
        profile.checkpoint, device=settings.device, **kwargs
    )
    return runner


def create_app(
    settings: ServerSettings | None = None,
    runner_factory: Callable[[], object] | None = None,
) -> FastAPI:
    from omni_infinity.serve.stream import StreamRequest, StreamService

    configured = settings or ServerSettings.from_env()
    profile = resolve_profile(
        configured.model_arch,
        configured.optimizations,
        configured.checkpoint,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = JobStore(configured.jobs_dir)
        store.recover_interrupted()
        runner = runner_factory() if runner_factory else load_runner(configured)
        service = JobService(runner, profile, store)
        stream_service = StreamService(
            service,
            max_sessions=configured.stream_max_sessions,
            session_ttl=configured.stream_session_ttl,
            chunk_frames=configured.stream_chunk_frames,
            hls=configured.stream_fallback_hls,
            queue_chunks=configured.stream_queue_chunks,
        )
        app.state.store = store
        app.state.service = service
        app.state.stream_service = stream_service
        app.state.profile = profile
        try:
            yield
        finally:
            stream_service.shutdown()
            service.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/jobs", response_model=JobResponse, status_code=202)
    def create_job(request: GenerationRequest, response: Response):
        try:
            record = app.state.service.submit(request)
        except Ref2VANotImplemented as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except ProfileConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InvalidMedia as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        response.headers["Location"] = f"/v1/jobs/{record.id}"
        return JobResponse.from_record(record)

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str):
        try:
            return JobResponse.from_record(app.state.store.get(job_id))
        except JobNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}/artifacts")
    def get_artifacts(job_id: str):
        try:
            record = app.state.store.get(job_id)
        except JobNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = app.state.store.video_path(job_id)
        if record.status != JobStatus.SUCCEEDED or not path.is_file():
            raise HTTPException(status_code=409, detail="artifact is not ready")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"{job_id}.mp4",
        )

    if configured.stream_enabled:

        @app.get("/", include_in_schema=False)
        def stream_player():
            return FileResponse(_WEBUI_DIR / "index.html")

        @app.get("/player.js", include_in_schema=False)
        def stream_player_script():
            return FileResponse(
                _WEBUI_DIR / "player.js", media_type="text/javascript"
            )

        def create_stream(request: StreamRequest):
            try:
                stream_id = app.state.stream_service.open_session(request)
            except SessionLimit as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Ref2VANotImplemented as exc:
                raise HTTPException(status_code=501, detail=str(exc)) from exc
            except ProfileConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except InvalidMedia as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return {"stream_id": stream_id}

        create_stream.__annotations__["request"] = StreamRequest
        app.add_api_route(
            "/v1/streams",
            create_stream,
            methods=["POST"],
            status_code=202,
        )

        async def stream_socket(websocket: WebSocket, stream_id: str) -> None:
            await app.state.stream_service.run_socket(websocket, stream_id)

        app.add_api_websocket_route("/v1/streams/{stream_id}/ws", stream_socket)

        if configured.stream_fallback_hls:

            def stream_media(stream_id: str):
                try:
                    app.state.stream_service.start_hls(stream_id)
                    init, chunks, ended = (
                        app.state.stream_service.media_snapshot(stream_id)
                    )
                except KeyError as exc:
                    raise HTTPException(
                        status_code=404, detail="unknown stream session"
                    ) from exc
                if init is None:
                    raise HTTPException(
                        status_code=409, detail="stream media is not ready"
                    )
                return init, chunks, ended

            @app.get("/v1/streams/{stream_id}/playlist.m3u8")
            def stream_playlist(stream_id: str):
                _, chunks, ended = stream_media(stream_id)
                durations = [chunk.duration for chunk in chunks]
                target = max(1, math.ceil(max(durations, default=1.0)))
                lines = [
                    "#EXTM3U",
                    "#EXT-X-VERSION:7",
                    f"#EXT-X-TARGETDURATION:{target}",
                    "#EXT-X-MEDIA-SEQUENCE:0",
                    "#EXT-X-PLAYLIST-TYPE:EVENT",
                    '#EXT-X-MAP:URI="init.mp4"',
                ]
                for chunk in chunks:
                    lines.extend(
                        [
                            f"#EXTINF:{chunk.duration:.6f},",
                            f"seg/{chunk.index}.m4s",
                        ]
                    )
                if ended:
                    lines.append("#EXT-X-ENDLIST")
                return Response(
                    "\n".join(lines) + "\n",
                    media_type="application/vnd.apple.mpegurl",
                )

            @app.get("/v1/streams/{stream_id}/init.mp4")
            def stream_init(stream_id: str):
                init, _, _ = stream_media(stream_id)
                return Response(init, media_type="video/mp4")

            @app.get("/v1/streams/{stream_id}/seg/{index}.m4s")
            def stream_segment(stream_id: str, index: int):
                _, chunks, _ = stream_media(stream_id)
                chunk = next(
                    (item for item in chunks if item.index == index), None
                )
                if chunk is None or chunk.video_bytes is None:
                    raise HTTPException(
                        status_code=404, detail="stream segment not found"
                    )
                return Response(
                    chunk.video_bytes, media_type="video/iso.segment"
                )

            @app.get("/v1/streams/{stream_id}/prompts.json")
            def stream_prompts(stream_id: str):
                _, chunks, _ = stream_media(stream_id)
                return {
                    "cues": [
                        {
                            "index": chunk.index,
                            "pts": chunk.pts,
                            "duration": chunk.duration,
                            "prompt": chunk.prompt,
                            "instruction": chunk.instruction,
                            "action": chunk.action,
                        }
                        for chunk in chunks
                    ]
                }

    return app
