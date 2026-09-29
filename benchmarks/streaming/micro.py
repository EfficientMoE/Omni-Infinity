# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Client-observed microbenchmark for one streaming session.

The stream protocol does not expose denoise, VAE, audio, fragment-production,
or WebSocket-send spans. Those fields stay explicitly unsupported. The
benchmark measures only top-level receive boundaries and post-generation
artifact retrieval; WebSocket arrival gaps are never treated as production
latency.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.streaming.client import (
    ChunkEvent,
    EndEvent,
    InitEvent,
    LocalhostConfig,
    LocalhostTransport,
    ProtocolError,
    StreamTransport,
    build_request,
    detect_native_chunker,
    observe_events,
)
from benchmarks.streaming.contract import (
    BASELINE_CHUNK_FRAMES,
    BASELINE_OPTS,
    BASELINE_REPS,
    BASELINE_TRANSPORT,
    ENVELOPE_TOLERANCE,
    FIELDS,
    FPS,
    PROMPT,
    REQUESTED_FRAMES,
    SEED,
    SHORT_EDGE,
    STACKS,
    WARMUP_DISCARD,
)

UNSUPPORTED = "UNSUPPORTED"
SPAN_KEYS: tuple[str, ...] = (
    "denoise_ms",
    "vae_decode_ms",
    "audio_decode_ms",
    "fmp4_init_ms",
    "fmp4_frag_ms",
    "ws_send_ms",
)
PHASE_KEYS: tuple[str, ...] = (
    "accept_to_init_ms",
    "init_to_first_chunk_ms",
    "first_to_last_chunk_arrival_ms",
    "last_chunk_to_end_ms",
)
EXTRA_FIELDS: tuple[str, ...] = SPAN_KEYS + (
    *PHASE_KEYS,
    "other_ms",
    "session_wall_ms",
    "envelope_ok",
    "artifact_retrieval_ms",
    "artifact_retrieval_support",
    "artifact_retrieval_reason",
)
MICRO_FIELDS: tuple[str, ...] = FIELDS + EXTRA_FIELDS


def envelope(
    spans: dict[str, float], session_wall_ms: float
) -> tuple[float, bool]:
    """Check only explicitly instrumented, non-overlapping phase spans."""
    if not math.isfinite(session_wall_ms) or session_wall_ms <= 0:
        raise ValueError("session_wall_ms must be positive")
    values = [float(value) for value in spans.values()]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("instrumented spans must be finite and non-negative")
    named = sum(values)
    other = session_wall_ms - named
    ok = abs(other) / session_wall_ms <= ENVELOPE_TOLERANCE
    return other, ok


def dry_run_text() -> str:
    lines = [
        "streaming micro dry-run",
        f"shape={SHORT_EDGE}p/{REQUESTED_FRAMES}f",
        f"seed={SEED}",
        f"prompt={PROMPT}",
        f"fps={FPS}",
        f"warmup-discard={WARMUP_DISCARD}",
        f"reps={BASELINE_REPS}",
        "clock=client-monotonic",
        "arrival-gaps=client-receive-only",
        "hls-ttff=UNSUPPORTED:post-generation-only",
    ]
    lines.extend(f"phase={name}" for name in PHASE_KEYS)
    lines.extend(f"span={name}:UNSUPPORTED" for name in SPAN_KEYS)
    lines.extend(f"stack={stack}" for stack in STACKS)
    return "\n".join(lines) + "\n"


def _blank_extra(row: dict, *, phase_value: str) -> dict:
    scored = dict(row)
    for name in SPAN_KEYS:
        scored[name] = UNSUPPORTED
    for name in PHASE_KEYS:
        scored[name] = phase_value
    scored["other_ms"] = phase_value
    scored["session_wall_ms"] = phase_value
    scored["envelope_ok"] = phase_value
    scored["artifact_retrieval_ms"] = phase_value
    scored["artifact_retrieval_support"] = "unsupported"
    scored["artifact_retrieval_reason"] = "no completed stream session"
    scored["production_latency_support"] = "unsupported"
    scored["production_latency_reason"] = (
        "wire protocol has no server production timestamps"
    )
    scored["detailed_spans_support"] = "unsupported"
    scored["detailed_spans_reason"] = (
        "wire protocol has no denoise, VAE, audio, fMP4, or send telemetry"
    )
    scored["hls_ttff_support"] = "unsupported"
    scored["hls_ttff_reason"] = (
        "HLS playlist is available only after generation completes"
    )
    return scored


def _skip_row(stack: str, arch: str, rep: int, notes: str) -> dict:
    row = {field: "" for field in FIELDS}
    row.update(
        {
            "track": "micro",
            "stack": stack,
            "arch": arch,
            "chunk_frames": BASELINE_CHUNK_FRAMES,
            "transport": BASELINE_TRANSPORT,
            "rep": rep,
            "quality_vs_job": "skipped",
            "notes": notes,
            "video_duration_s": REQUESTED_FRAMES / FPS,
        }
    )
    if stack == "omni-native":
        row["native_runner"] = False
    row["verdict"] = "SKIP"
    return row


ArtifactFetch = Callable[[str], Any]
FragmentProbe = Callable[[bytes, bytes], tuple[float | None, float | None]]


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    if end_ns < start_ns:
        raise ProtocolError("event timestamps must be monotonic")
    return (end_ns - start_ns) / 1_000_000.0


def measure_session(
    transport: StreamTransport,
    *,
    stack: str,
    arch: str,
    rep: int,
    artifact_fetch: ArtifactFetch,
    fragment_probe: FragmentProbe | None = None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> dict:
    """Measure client-visible boundaries for one completed stream session."""
    source = "native" if stack == "omni-native" else "clip"
    request = build_request(
        model_arch=arch,
        prompt=PROMPT,
        source=source,
        optimizations=list(BASELINE_OPTS),
        seed=SEED,
        num_frames=REQUESTED_FRAMES,
    )
    accepted = transport.create_stream(request)
    events = tuple(transport.events(accepted.stream_id))
    metrics = observe_events(
        accepted_ns=accepted.accepted_ns,
        events=events,
        request_prompt=PROMPT,
        fragment_probe=fragment_probe,
        native_requested=source == "native",
    )

    init_events = [item for item in events if isinstance(item.event, InitEvent)]
    chunks = [item for item in events if isinstance(item.event, ChunkEvent)]
    end_events = [item for item in events if isinstance(item.event, EndEvent)]
    if len(init_events) != 1 or not chunks or len(end_events) != 1:
        raise ProtocolError(
            "microbenchmark requires one init, media chunks, and one end"
        )
    init = init_events[0]
    first_chunk = chunks[0]
    last_chunk = chunks[-1]
    end = end_events[0]
    phases = {
        "accept_to_init_ms": _elapsed_ms(
            accepted.accepted_ns, init.received_ns
        ),
        "init_to_first_chunk_ms": _elapsed_ms(
            init.received_ns, first_chunk.received_ns
        ),
        "first_to_last_chunk_arrival_ms": _elapsed_ms(
            first_chunk.received_ns, last_chunk.received_ns
        ),
        "last_chunk_to_end_ms": _elapsed_ms(
            last_chunk.received_ns, end.received_ns
        ),
    }
    session_wall_ms = _elapsed_ms(accepted.accepted_ns, end.received_ns)
    other_ms, envelope_ok = envelope(phases, session_wall_ms)

    retrieval_start_ns = clock_ns()
    artifact_fetch(end.event.artifact_url)
    artifact_retrieval_ms = _elapsed_ms(retrieval_start_ns, clock_ns())

    row = {field: "" for field in FIELDS}
    row.update(
        {
            "track": "micro",
            "stack": stack,
            "arch": arch,
            "chunk_frames": BASELINE_CHUNK_FRAMES,
            "transport": BASELINE_TRANSPORT,
            "rep": rep,
            "quality_vs_job": "skipped",
            "e2e_ms": session_wall_ms,
            "video_duration_s": sum(item.event.duration for item in chunks),
            "e2e_rtf": session_wall_ms
            / (sum(item.event.duration for item in chunks) * 1000.0),
            "resolution": request["resolution"],
            "notes": "client-observed-top-level",
            "verdict": "REPORT",
            **metrics.contract_fields(),
            **phases,
            "session_wall_ms": session_wall_ms,
            "other_ms": other_ms,
            "envelope_ok": envelope_ok,
            "artifact_retrieval_ms": artifact_retrieval_ms,
            "artifact_retrieval_support": "supported",
            "artifact_retrieval_reason": (
                "timed GET after the stream end event"
            ),
            "detailed_spans_reason": (
                "wire protocol has no denoise, VAE, audio, fMP4, "
                "or send telemetry"
            ),
        }
    )
    for name in SPAN_KEYS:
        row[name] = UNSUPPORTED
    return row


_WEIGHT_ENV = {
    "vllm-helios": "OMNI_HELIOS_WEIGHTS",
    "sglang-lingbot": "OMNI_LINGBOT_WEIGHTS",
    "sglang-sana-wm": "OMNI_SANA_WM_WEIGHTS",
}


def _weights_ready(stack: str) -> bool:
    key = _WEIGHT_ENV.get(stack)
    if key is None:
        return False
    raw = os.environ.get(key, "")
    return bool(raw) and Path(raw).is_dir()


_ARCHES = {
    "omni-clip": ("h3-dense", "vdn-hybrid"),
    "omni-native": ("h3-dense",),
    "vllm-helios": ("helios-distilled",),
    "sglang-lingbot": ("lingbot-world",),
    "sglang-sana-wm": ("sana-wm",),
}


def _live_artifact_fetcher(config: LocalhostConfig) -> ArtifactFetch:
    def fetch(path: str) -> bytes:
        if not path.startswith("/") or "://" in path:
            raise ProtocolError("artifact_url must be a relative server path")
        import httpx

        response = httpx.get(
            f"{config.origin}{path}", timeout=config.request_timeout_s
        )
        response.raise_for_status()
        return response.content

    return fetch


def measure_rows(
    repo_root: Path,
    stacks: tuple[str, ...],
    *,
    base_url: str = "http://127.0.0.1:8000",
) -> list[dict]:
    del repo_root
    rows: list[dict] = []
    config = LocalhostConfig(base_url)
    transport = LocalhostTransport(config)
    fetch_artifact = _live_artifact_fetcher(config)
    native_support = detect_native_chunker()
    for stack in stacks:
        for arch in _ARCHES[stack]:
            if stack == "omni-native" and not native_support.supported:
                rows.extend(
                    _blank_extra(
                        _skip_row(
                            stack,
                            arch,
                            rep,
                            f"native-unsupported:{native_support.reason}",
                        ),
                        phase_value="SKIP",
                    )
                    for rep in range(BASELINE_REPS)
                )
                continue
            if stack not in ("omni-clip", "omni-native"):
                notes = (
                    "external-adapter-absent"
                    if _weights_ready(stack)
                    else "weights-absent"
                )
                rows.extend(
                    _blank_extra(
                        _skip_row(stack, arch, rep, notes),
                        phase_value="SKIP",
                    )
                    for rep in range(BASELINE_REPS)
                )
                continue

            measured: list[dict] = []
            failure: Exception | None = None
            for run in range(BASELINE_REPS + WARMUP_DISCARD):
                try:
                    row = measure_session(
                        transport,
                        stack=stack,
                        arch=arch,
                        rep=max(0, run - WARMUP_DISCARD),
                        artifact_fetch=fetch_artifact,
                    )
                except Exception as exc:
                    failure = exc
                    break
                if run >= WARMUP_DISCARD:
                    measured.append(row)
            if failure is None:
                rows.extend(measured)
                continue
            notes = f"server-down:{type(failure).__name__}"
            for rep in range(BASELINE_REPS):
                rows.append(
                    _blank_extra(
                        _skip_row(stack, arch, rep, notes),
                        phase_value="SKIP",
                    )
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MICRO_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in MICRO_FIELDS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stack", action="append", choices=STACKS)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="localhost origin serving the issue #14 stream API",
    )
    parser.add_argument("--out", default="results/streaming/micro.csv")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        sys.stdout.write(dry_run_text())
        return 0
    selected = tuple(args.stack) if args.stack else STACKS
    rows = measure_rows(Path(args.repo_root), selected, base_url=args.base_url)
    if any(row.get("envelope_ok") is False for row in rows):
        write_csv(Path(args.out), rows)
        return 1
    write_csv(Path(args.out), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
