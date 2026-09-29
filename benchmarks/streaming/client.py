# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Typed, CPU-safe client utilities for streaming benchmarks.

This module deliberately does not import ``omni_infinity.streaming``: that
package imports the GPU runner and torch. HTTP, WebSocket, and media decoder
dependencies are imported only when their corresponding operation is used.
"""

from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import io
import json
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias
from urllib.parse import urlparse

_CODEC = 'video/mp4; codecs="avc1.42E01E,mp4a.40.2"'
_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class ProtocolError(ValueError):
    """The server sent data outside the documented stream protocol."""


class StreamRequestError(RuntimeError):
    """The stream creation request was rejected."""


@dataclass(frozen=True)
class ActionCueEvent:
    t: float
    action: str
    instruction: str


@dataclass(frozen=True)
class InitEvent:
    codec: str
    init_bytes: bytes
    action_script: tuple[ActionCueEvent, ...]
    type: Literal["init"] = "init"


@dataclass(frozen=True)
class ChunkEvent:
    index: int
    pts: float
    duration: float
    keyframe: bool
    video_bytes: bytes | None
    audio_bytes: bytes | None
    prompt: str
    instruction: str | None
    action: str | None
    done: bool
    type: Literal["chunk"] = "chunk"


@dataclass(frozen=True)
class EndEvent:
    artifact_url: str
    type: Literal["end"] = "end"


@dataclass(frozen=True)
class ErrorEvent:
    detail: str
    type: Literal["error"] = "error"


StreamEvent: TypeAlias = InitEvent | ChunkEvent | EndEvent | ErrorEvent


@dataclass(frozen=True)
class TimedEvent:
    event: StreamEvent
    received_ns: int


@dataclass(frozen=True)
class AcceptedStream:
    stream_id: str
    accepted_ns: int


class StreamTransport(Protocol):
    """Transport boundary shared by in-process and localhost clients."""

    def create_stream(self, request: Mapping[str, Any]) -> AcceptedStream: ...

    def events(self, stream_id: str) -> Iterator[TimedEvent]: ...


@dataclass(frozen=True)
class MetricSupport:
    supported: bool
    reason: str

    @property
    def value(self) -> Literal["supported", "unsupported"]:
        return "supported" if self.supported else "unsupported"


@dataclass(frozen=True)
class ObservedMetrics:
    """Metrics derivable from client-visible bytes and receive timestamps."""

    ttff_ms: float | None
    arrival_gaps_ms: tuple[float, ...]
    stall_count: int
    av_pts_boundary_offset_ms: float | None
    cue_alignment: bool | None
    production_latency: MetricSupport
    detailed_spans: MetricSupport
    native_performance: MetricSupport
    hls_ttff: MetricSupport

    def contract_fields(self) -> dict[str, Any]:
        """Serialize values under the shared CSV contract's field names."""

        gaps = list(self.arrival_gaps_ms)
        return {
            "ttff_ms": self.ttff_ms,
            "arrival_gap_ms": json.dumps(gaps, separators=(",", ":")),
            "arrival_gap_p50": _percentile(gaps, 50),
            "arrival_gap_p95": _percentile(gaps, 95),
            "stall_count": self.stall_count,
            "av_offset_ms": self.av_pts_boundary_offset_ms,
            "cue_alignment": (
                None if self.cue_alignment is None else int(self.cue_alignment)
            ),
            "production_latency_support": self.production_latency.value,
            "production_latency_reason": self.production_latency.reason,
            "detailed_spans_support": self.detailed_spans.value,
            "detailed_spans_reason": self.detailed_spans.reason,
            "native_performance_support": self.native_performance.value,
            "native_performance_reason": self.native_performance.reason,
            "hls_ttff_support": self.hls_ttff.value,
            "hls_ttff_reason": self.hls_ttff.reason,
        }


@dataclass(frozen=True)
class StreamRun:
    stream_id: str
    accepted_ns: int
    events: tuple[TimedEvent, ...]
    metrics: ObservedMetrics


FragmentProbe = Callable[[bytes, bytes], tuple[float | None, float | None]]


class TestClientTransport:
    """FastAPI TestClient adapter used by CPU protocol tests."""

    __test__ = False

    def __init__(self, client: Any):
        self._client = client

    def create_stream(self, request: Mapping[str, Any]) -> AcceptedStream:
        response = self._client.post("/v1/streams", json=dict(request))
        accepted_ns = time.monotonic_ns()
        if response.status_code != 202:
            raise StreamRequestError(
                f"POST /v1/streams returned {response.status_code}: "
                f"{response.text}"
            )
        stream_id = _parse_created(response.json())
        return AcceptedStream(stream_id, accepted_ns)

    def events(self, stream_id: str) -> Iterator[TimedEvent]:
        with self._client.websocket_connect(
            f"/v1/streams/{stream_id}/ws"
        ) as socket:
            while True:
                event = parse_event(socket.receive_json())
                yield TimedEvent(event, time.monotonic_ns())
                if isinstance(event, (EndEvent, ErrorEvent)):
                    return


@dataclass(frozen=True)
class LocalhostConfig:
    """Connection settings for a real server bound to the local machine."""

    base_url: str = "http://127.0.0.1:8000"
    request_timeout_s: float = 30.0
    websocket_timeout_s: float = 600.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in _LOCAL_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(
                "base_url must be an HTTP(S) localhost origin without a path"
            )
        if self.request_timeout_s <= 0 or self.websocket_timeout_s <= 0:
            raise ValueError("timeouts must be positive")

    @property
    def origin(self) -> str:
        return self.base_url.rstrip("/")

    def websocket_url(self, stream_id: str) -> str:
        parsed = urlparse(self.origin)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{scheme}://{parsed.netloc}/v1/streams/{stream_id}/ws"


class LocalhostTransport:
    """Synchronous HTTP + WebSocket transport for local measurements."""

    def __init__(self, config: LocalhostConfig | None = None):
        self.config = config or LocalhostConfig()

    def create_stream(self, request: Mapping[str, Any]) -> AcceptedStream:
        import httpx

        response = httpx.post(
            f"{self.config.origin}/v1/streams",
            json=dict(request),
            timeout=self.config.request_timeout_s,
        )
        accepted_ns = time.monotonic_ns()
        if response.status_code != 202:
            raise StreamRequestError(
                f"POST /v1/streams returned {response.status_code}: "
                f"{response.text}"
            )
        return AcceptedStream(_parse_created(response.json()), accepted_ns)

    def events(self, stream_id: str) -> Iterator[TimedEvent]:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError(
                "the 'websockets' package is required for live measurements"
            ) from exc

        with connect(
            self.config.websocket_url(stream_id),
            open_timeout=self.config.request_timeout_s,
            close_timeout=self.config.request_timeout_s,
        ) as socket:
            while True:
                payload = socket.recv(timeout=self.config.websocket_timeout_s)
                if not isinstance(payload, str):
                    raise ProtocolError("WebSocket events must be JSON text")
                try:
                    decoded = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ProtocolError("WebSocket event is not JSON") from exc
                event = parse_event(decoded)
                yield TimedEvent(event, time.monotonic_ns())
                if isinstance(event, (EndEvent, ErrorEvent)):
                    return


class BenchmarkClient:
    def __init__(
        self,
        transport: StreamTransport,
        *,
        fragment_probe: FragmentProbe | None = None,
    ):
        self.transport = transport
        self.fragment_probe = fragment_probe or probe_fragment

    def run(self, request: Mapping[str, Any]) -> StreamRun:
        accepted = self.transport.create_stream(request)
        events = tuple(self.transport.events(accepted.stream_id))
        metrics = observe_events(
            accepted_ns=accepted.accepted_ns,
            events=events,
            request_prompt=str(request.get("prompt", "")),
            fragment_probe=self.fragment_probe,
            native_requested=request.get("source") == "native",
        )
        return StreamRun(
            stream_id=accepted.stream_id,
            accepted_ns=accepted.accepted_ns,
            events=events,
            metrics=metrics,
        )


def parse_event(payload: Any) -> StreamEvent:
    if not isinstance(payload, Mapping):
        raise ProtocolError("event must be a JSON object")
    event_type = payload.get("type")
    if event_type == "init":
        _exact_fields(
            payload, {"type", "codec", "init_b64", "action_script"}, "init"
        )
        codec = _string(payload["codec"], "codec")
        if codec != _CODEC:
            raise ProtocolError(f"unsupported codec: {codec!r}")
        cues_value = payload["action_script"]
        if not isinstance(cues_value, list):
            raise ProtocolError("action_script must be a list")
        cues = tuple(_parse_cue(cue) for cue in cues_value)
        init_bytes = _decode_b64(payload["init_b64"], "init_b64")
        return InitEvent(codec, init_bytes, cues)
    if event_type == "chunk":
        _exact_fields(
            payload,
            {
                "type",
                "index",
                "pts",
                "duration",
                "keyframe",
                "video_b64",
                "audio_b64",
                "prompt",
                "instruction",
                "action",
                "done",
            },
            "chunk",
        )
        index = _integer(payload["index"], "index", minimum=0)
        pts = _number(payload["pts"], "pts", minimum=0.0)
        duration = _number(payload["duration"], "duration", minimum=0.0)
        if duration == 0:
            raise ProtocolError("duration must be positive")
        return ChunkEvent(
            index=index,
            pts=pts,
            duration=duration,
            keyframe=_boolean(payload["keyframe"], "keyframe"),
            video_bytes=_optional_b64(payload["video_b64"], "video_b64"),
            audio_bytes=_optional_b64(payload["audio_b64"], "audio_b64"),
            prompt=_string(payload["prompt"], "prompt"),
            instruction=_optional_string(payload["instruction"], "instruction"),
            action=_optional_string(payload["action"], "action"),
            done=_boolean(payload["done"], "done"),
        )
    if event_type == "end":
        _exact_fields(payload, {"type", "artifact_url"}, "end")
        return EndEvent(_string(payload["artifact_url"], "artifact_url"))
    if event_type == "error":
        _exact_fields(payload, {"type", "detail"}, "error")
        return ErrorEvent(_string(payload["detail"], "detail"))
    raise ProtocolError(f"unknown event type: {event_type!r}")


def build_request(
    *,
    model_arch: Literal["h3-dense", "vdn-hybrid"] = "h3-dense",
    resolution: Literal["256p", "512p", "768p"] | None = None,
    prompt: str = "a red ball bouncing",
    source: Literal["clip", "native"] = "clip",
    **overrides: Any,
) -> dict[str, Any]:
    """Build requests with architecture-appropriate resolution metadata."""

    selected_resolution = resolution or (
        "768p" if model_arch == "vdn-hybrid" else "256p"
    )
    if model_arch == "vdn-hybrid" and selected_resolution != "768p":
        raise ValueError("vdn-hybrid benchmark requests require 768p")
    request: dict[str, Any] = {
        "type": "fl2va",
        "prompt": prompt,
        "model_arch": model_arch,
        "optimizations": [],
        "seed": 0,
        "num_inference_steps": 8,
        "resolution": selected_resolution,
        "num_frames": 120,
        "source": source,
        "action_script": [],
    }
    request.update(overrides)
    return request


def observe_events(
    *,
    accepted_ns: int,
    events: Sequence[TimedEvent],
    request_prompt: str,
    fragment_probe: FragmentProbe | None = None,
    native_requested: bool = False,
) -> ObservedMetrics:
    """Validate a session and derive only client-observable measurements."""

    init, chunks = _validate_sequence(events)
    probe = fragment_probe or probe_fragment
    first_decodable_ns: int | None = None
    offsets_ms: list[float] = []
    chunk_times: list[int] = []
    for timed in events:
        if not isinstance(timed.event, ChunkEvent):
            continue
        chunk = timed.event
        chunk_times.append(timed.received_ns)
        if chunk.video_bytes is None:
            continue
        video_pts, audio_pts = probe(init.init_bytes, chunk.video_bytes)
        if first_decodable_ns is None:
            first_decodable_ns = timed.received_ns
        if video_pts is not None and audio_pts is not None:
            offsets_ms.append(abs(video_pts - audio_pts) * 1000.0)
    if chunks and first_decodable_ns is None:
        raise ProtocolError("stream contains no decodable media fragment")

    arrival_gaps = tuple(
        (current - previous) / 1_000_000.0
        for previous, current in zip(chunk_times, chunk_times[1:])
    )
    stalls = _count_stalls(chunks, chunk_times)
    native_support = (
        detect_native_chunker()
        if native_requested
        else MetricSupport(False, "session does not use the native source")
    )
    return ObservedMetrics(
        ttff_ms=(
            None
            if first_decodable_ns is None
            else (first_decodable_ns - accepted_ns) / 1_000_000.0
        ),
        arrival_gaps_ms=arrival_gaps,
        stall_count=stalls,
        av_pts_boundary_offset_ms=max(offsets_ms) if offsets_ms else None,
        cue_alignment=_cue_alignment(init, chunks, request_prompt),
        production_latency=MetricSupport(
            False, "wire protocol has no server production timestamps"
        ),
        detailed_spans=MetricSupport(
            False,
            "wire protocol has no denoise, VAE, audio, or fMP4 telemetry",
        ),
        native_performance=native_support,
        hls_ttff=MetricSupport(
            False, "HLS playlist is available only after generation completes"
        ),
    )


def probe_fragment(
    init_bytes: bytes, fragment_bytes: bytes
) -> tuple[float | None, float | None]:
    """Decode one fMP4 fragment and return first video/audio PTS in seconds."""

    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to probe media fragments") from exc

    first_pts: dict[str, float] = {}
    decoded_video = False
    try:
        with av.open(io.BytesIO(init_bytes + fragment_bytes)) as container:
            for packet in container.demux():
                stream_type = packet.stream.type
                if (
                    stream_type in {"video", "audio"}
                    and stream_type not in first_pts
                    and packet.pts is not None
                ):
                    first_pts[stream_type] = float(
                        packet.pts * packet.time_base
                    )
                if stream_type == "video":
                    decoded_video = bool(packet.decode()) or decoded_video
    except Exception as exc:
        raise ProtocolError("media fragment is not decodable") from exc
    if not decoded_video:
        raise ProtocolError("media fragment contains no decodable video frame")
    return first_pts.get("video"), first_pts.get("audio")


def decoded_video_digest(payload: bytes) -> tuple[str, int]:
    """Return a decoded RGB-frame digest for stream/artifact parity checks."""

    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for decoded parity") from exc

    digest = hashlib.sha256()
    frames = 0
    with av.open(io.BytesIO(payload)) as container:
        for frame in container.decode(video=0):
            array = frame.to_ndarray(format="rgb24")
            digest.update(array.tobytes())
            frames += 1
    if frames == 0:
        raise ProtocolError("media contains no decoded video frames")
    return digest.hexdigest(), frames


def decoded_parity(
    init_bytes: bytes,
    chunks: Sequence[ChunkEvent],
    artifact_bytes: bytes,
) -> bool:
    """Compare decoded video frames, not container byte layout."""

    stream_bytes = init_bytes + b"".join(
        chunk.video_bytes or b"" for chunk in chunks
    )
    return decoded_video_digest(stream_bytes) == decoded_video_digest(
        artifact_bytes
    )


def detect_native_chunker(path: Path | None = None) -> MetricSupport:
    """Detect the checked-in synthetic NativeChunker without importing torch."""

    source_path = path or (
        Path(__file__).parents[2] / "omni_infinity/streaming/native.py"
    )
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return MetricSupport(False, f"cannot inspect NativeChunker: {exc}")
    native = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NativeChunker"
        ),
        None,
    )
    if native is None:
        return MetricSupport(False, "NativeChunker implementation is absent")
    for node in ast.walk(native):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "full"
            and isinstance(function.value, ast.Name)
            and function.value.id == "np"
            and node.args
            and isinstance(node.args[0], ast.Tuple)
        ):
            continue
        dimensions = [
            item.value
            for item in node.args[0].elts
            if isinstance(item, ast.Constant) and isinstance(item.value, int)
        ]
        if dimensions[-3:] == [16, 16, 3]:
            return MetricSupport(
                False, "NativeChunker is the synthetic 16x16 stub"
            )
    return MetricSupport(True, "NativeChunker is not the synthetic 16x16 stub")


def _parse_created(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        raise ProtocolError("create response must be a JSON object")
    _exact_fields(payload, {"stream_id"}, "create response")
    stream_id = _string(payload["stream_id"], "stream_id")
    if len(stream_id) != 32 or any(
        character not in "0123456789abcdef" for character in stream_id
    ):
        raise ProtocolError("stream_id must be 32 lowercase hexadecimal chars")
    return stream_id


def _parse_cue(payload: Any) -> ActionCueEvent:
    if not isinstance(payload, Mapping):
        raise ProtocolError("action cue must be a JSON object")
    _exact_fields(payload, {"t", "action", "instruction"}, "action cue")
    return ActionCueEvent(
        t=_number(payload["t"], "t", minimum=0.0),
        action=_string(payload["action"], "action"),
        instruction=_string(payload["instruction"], "instruction"),
    )


def _validate_sequence(
    events: Sequence[TimedEvent],
) -> tuple[InitEvent, tuple[ChunkEvent, ...]]:
    if not events:
        raise ProtocolError("stream produced no events")
    if not isinstance(events[0].event, InitEvent):
        if isinstance(events[0].event, ErrorEvent):
            raise ProtocolError(f"stream failed: {events[0].event.detail}")
        raise ProtocolError("first stream event must be init")
    init = events[0].event
    chunks: list[ChunkEvent] = []
    ended = False
    saw_done = False
    previous_ns = events[0].received_ns
    for timed in events[1:]:
        if timed.received_ns < previous_ns:
            raise ProtocolError("event timestamps must be monotonic")
        previous_ns = timed.received_ns
        event = timed.event
        if isinstance(event, InitEvent):
            raise ProtocolError("stream sent more than one init event")
        if isinstance(event, ErrorEvent):
            raise ProtocolError(f"stream failed: {event.detail}")
        if isinstance(event, ChunkEvent):
            if ended or saw_done:
                raise ProtocolError("chunk arrived after stream completion")
            if event.index != len(chunks):
                raise ProtocolError(
                    "chunk indexes must be contiguous from zero"
                )
            if chunks and event.pts < chunks[-1].pts + chunks[-1].duration:
                raise ProtocolError("chunk PTS ranges must not overlap")
            chunks.append(event)
            saw_done = event.done
            continue
        if isinstance(event, EndEvent):
            if ended:
                raise ProtocolError("stream sent more than one end event")
            if chunks and not saw_done:
                raise ProtocolError("end arrived before a done chunk")
            ended = True
    if not ended:
        raise ProtocolError("stream did not send an end event")
    return init, tuple(chunks)


def _count_stalls(chunks: Sequence[ChunkEvent], times: Sequence[int]) -> int:
    if not chunks:
        return 0
    playable_until = times[0] + int(chunks[0].duration * 1_000_000_000)
    stalls = 0
    for chunk, arrived in zip(chunks[1:], times[1:]):
        if arrived > playable_until:
            stalls += 1
        playable_until = max(arrived, playable_until) + int(
            chunk.duration * 1_000_000_000
        )
    return stalls


def _cue_alignment(
    init: InitEvent, chunks: Sequence[ChunkEvent], prompt: str
) -> bool | None:
    if not chunks:
        return None
    if any(chunk.prompt != prompt for chunk in chunks):
        return False
    for cue in init.action_script:
        matching = next(
            (
                chunk
                for chunk in chunks
                if chunk.pts <= cue.t < chunk.pts + chunk.duration
            ),
            None,
        )
        if (
            matching is None
            or matching.action != cue.action
            or matching.instruction != cue.instruction
        ):
            return False
    return True


def _exact_fields(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ProtocolError(f"{label} missing fields: {missing}")
    if extra:
        raise ProtocolError(f"{label} has unexpected fields: {extra}")


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"{field} must be a boolean")
    return value


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: Any, field: str, *, minimum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ProtocolError(f"{field} must be a number >= {minimum}")
    return float(value)


def _decode_b64(value: Any, field: str) -> bytes:
    text = _string(value, field)
    try:
        decoded = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError(f"{field} must contain valid base64") from exc
    if not decoded:
        raise ProtocolError(f"{field} must not decode to empty bytes")
    return decoded


def _optional_b64(value: Any, field: str) -> bytes | None:
    if value is None:
        return None
    return _decode_b64(value, field)


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (pct / 100.0) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight
