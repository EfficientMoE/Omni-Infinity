# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Client-observed baseline for the video-streaming workload.

``--dry-run`` remains dependency-light and does not import torch or write a
CSV. Measurements use the shared typed stream client against a real localhost
server and record only values observable from the documented wire protocol.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.streaming.client import (  # noqa: E402
    BenchmarkClient,
    ChunkEvent,
    EndEvent,
    FragmentProbe,
    InitEvent,
    LocalhostConfig,
    LocalhostTransport,
    MetricSupport,
    StreamRun,
    StreamTransport,
    build_request,
    decoded_parity,
    detect_native_chunker,
)
from benchmarks.streaming.contract import (  # noqa: E402
    ARCH_COMPARISON_RESOLUTION,
    BASELINE_CHUNK_FRAMES,
    BASELINE_REPS,
    BASELINE_TRANSPORT,
    DENSE_DEFAULT_RESOLUTION,
    FIELDS,
    FPS,
    PROMPT,
    REQUESTED_FRAMES,
    SEED,
    STACKS,
    WARMUP_DISCARD,
    legal_frames,
    verdict_for,
)

_WEIGHT_ENV = {
    "vllm-helios": "OMNI_HELIOS_WEIGHTS",
    "sglang-lingbot": "OMNI_LINGBOT_WEIGHTS",
    "sglang-sana-wm": "OMNI_SANA_WM_WEIGHTS",
}
_ARCHES = {
    "omni-clip": ("h3-dense", "vdn-hybrid"),
    "omni-native": ("h3-dense",),
    "vllm-helios": ("helios-distilled",),
    "sglang-lingbot": ("lingbot-world",),
    "sglang-sana-wm": ("sana-wm",),
}

ArtifactFetcher = Callable[[str], bytes]
ParityChecker = Callable[[bytes, Sequence[ChunkEvent], bytes], bool]


def dry_run_text() -> str:
    frames = legal_frames(REQUESTED_FRAMES)
    lines = [
        "streaming baseline dry-run",
        f"shape={DENSE_DEFAULT_RESOLUTION}/{frames}f",
        f"vdn-shape={ARCH_COMPARISON_RESOLUTION}/{frames}f",
        f"requested-frames={REQUESTED_FRAMES}",
        f"seed={SEED}",
        f"prompt={PROMPT}",
        f"warmup-discard={WARMUP_DISCARD}",
        f"reps={BASELINE_REPS}",
        f"fps={FPS}",
        f"chunk_frames={BASELINE_CHUNK_FRAMES}",
        f"transport={BASELINE_TRANSPORT}",
    ]
    lines.extend(f"stack={stack}" for stack in STACKS)
    return "\n".join(lines) + "\n"


def weights_ready(stack: str, env: dict[str, str] | None = None) -> bool:
    key = _WEIGHT_ENV.get(stack)
    if key is None:
        return False
    source = os.environ if env is None else env
    raw = source.get(key, "")
    return bool(raw) and Path(raw).is_dir()


def measure_run(
    *,
    transport: StreamTransport,
    stack: str,
    arch: str,
    rep: int,
    fragment_probe: FragmentProbe | None = None,
    artifact_fetcher: ArtifactFetcher | None = None,
    parity_checker: ParityChecker = decoded_parity,
) -> dict:
    """Run one stream and serialize client-observable metrics."""

    resolution = _resolution_for(arch)
    request = build_request(
        model_arch=arch,
        resolution=resolution,
        prompt=PROMPT,
        seed=SEED,
        num_frames=legal_frames(REQUESTED_FRAMES),
        source="clip",
    )
    run = BenchmarkClient(transport, fragment_probe=fragment_probe).run(request)
    init, chunks, end = _media_events(run)
    end_ns = next(
        timed.received_ns for timed in run.events if timed.event is end
    )
    e2e_ms = (end_ns - run.accepted_ns) / 1_000_000.0
    duration_s = max(
        (chunk.pts + chunk.duration for chunk in chunks), default=0.0
    )
    parity = "unsupported"
    notes = ""
    if artifact_fetcher is not None:
        try:
            artifact = artifact_fetcher(end.artifact_url)
            parity = (
                "bitwise"
                if parity_checker(init.init_bytes, chunks, artifact)
                else "mismatch"
            )
        except (OSError, RuntimeError, ValueError) as exc:
            notes = f"decoded-parity-unsupported: {exc}"

    row = {field: "" for field in FIELDS}
    row.update(
        {
            "track": "baseline",
            "stack": stack,
            "arch": arch,
            "chunk_frames": BASELINE_CHUNK_FRAMES,
            "transport": BASELINE_TRANSPORT,
            "rep": rep,
            "t_job_ready_ms": e2e_ms,
            "prompt_index_match": run.metrics.contract_fields()[
                "cue_alignment"
            ],
            "quality_vs_job": parity,
            "e2e_ms": e2e_ms,
            "video_duration_s": duration_s,
            "e2e_rtf": (e2e_ms / (duration_s * 1000.0) if duration_s else ""),
            "resolution": resolution,
            "decoded_parity": parity,
            "notes": notes,
        }
    )
    row.update(_csv_values(run.metrics.contract_fields()))
    row["verdict"] = verdict_for(row)
    return row


def skip_rows(
    stack: str,
    notes: str,
    *,
    arches: Sequence[str] | None = None,
    native_support: MetricSupport | None = None,
) -> list[dict]:
    rows = []
    selected_arches = tuple(arches) if arches is not None else _ARCHES[stack]
    for arch in selected_arches:
        for rep in range(BASELINE_REPS):
            row = {field: "" for field in FIELDS}
            row.update(
                {
                    "track": "baseline",
                    "stack": stack,
                    "arch": arch,
                    "chunk_frames": BASELINE_CHUNK_FRAMES,
                    "transport": BASELINE_TRANSPORT,
                    "rep": rep,
                    "quality_vs_job": "skipped",
                    "verdict": "SKIP",
                    "notes": notes,
                    "resolution": _resolution_for(arch),
                    "production_latency_support": "unsupported",
                    "production_latency_reason": (
                        "wire protocol has no server production timestamps"
                    ),
                    "detailed_spans_support": "unsupported",
                    "detailed_spans_reason": (
                        "wire protocol has no denoise, VAE, audio, or fMP4 "
                        "telemetry"
                    ),
                    "hls_ttff_support": "unsupported",
                    "hls_ttff_reason": (
                        "HLS playlist is available only after generation "
                        "completes"
                    ),
                }
            )
            if native_support is not None:
                row["native_performance_support"] = native_support.value
                row["native_performance_reason"] = native_support.reason
            rows.append(row)
    return rows


def measure_stack(
    stack: str,
    *,
    transport: StreamTransport | None,
    env: dict[str, str] | None = None,
    arches: Sequence[str] | None = None,
    fragment_probe: FragmentProbe | None = None,
    artifact_fetcher: ArtifactFetcher | None = None,
    parity_checker: ParityChecker = decoded_parity,
) -> list[dict]:
    """Discard one warmup per arch, then record the configured repetitions."""

    selected_arches = tuple(arches) if arches is not None else _ARCHES[stack]
    if stack == "omni-native":
        support = detect_native_chunker()
        return skip_rows(
            stack,
            support.reason,
            arches=selected_arches,
            native_support=support,
        )
    if stack not in ("omni-clip", "omni-native") and not weights_ready(
        stack, env
    ):
        return skip_rows(stack, "weights-absent")
    if stack != "omni-clip":
        return skip_rows(
            stack,
            f"stream-api-unsupported: no {stack} adapter",
            arches=selected_arches,
        )
    if transport is None:
        raise ValueError("omni-clip measurements require a stream transport")

    rows: list[dict] = []
    for arch in selected_arches:
        try:
            for _ in range(WARMUP_DISCARD):
                measure_run(
                    transport=transport,
                    stack=stack,
                    arch=arch,
                    rep=-1,
                    fragment_probe=fragment_probe,
                    artifact_fetcher=artifact_fetcher,
                    parity_checker=parity_checker,
                )
            for rep in range(BASELINE_REPS):
                rows.append(
                    measure_run(
                        transport=transport,
                        stack=stack,
                        arch=arch,
                        rep=rep,
                        fragment_probe=fragment_probe,
                        artifact_fetcher=artifact_fetcher,
                        parity_checker=parity_checker,
                    )
                )
        except Exception as exc:
            reason = f"server-down: {type(exc).__name__}: {exc}"
            rows.extend(skip_rows(stack, reason, arches=(arch,)))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def _resolution_for(arch: str) -> str:
    if arch == "vdn-hybrid":
        return ARCH_COMPARISON_RESOLUTION
    return DENSE_DEFAULT_RESOLUTION


def _media_events(
    run: StreamRun,
) -> tuple[InitEvent, tuple[ChunkEvent, ...], EndEvent]:
    init = next(
        event.event
        for event in run.events
        if isinstance(event.event, InitEvent)
    )
    chunks = tuple(
        event.event
        for event in run.events
        if isinstance(event.event, ChunkEvent)
    )
    end = next(
        event.event for event in run.events if isinstance(event.event, EndEvent)
    )
    return init, chunks, end


def _csv_values(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: "" if value is None else value for key, value in values.items()
    }


def _artifact_fetcher(config: LocalhostConfig) -> ArtifactFetcher:
    def fetch(path: str) -> bytes:
        import httpx

        response = httpx.get(
            f"{config.origin}{path}",
            timeout=config.request_timeout_s,
        )
        response.raise_for_status()
        return response.content

    return fetch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stack", action="append", choices=STACKS)
    parser.add_argument(
        "--out",
        default="results/streaming/baseline.csv",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--websocket-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    if args.dry_run:
        sys.stdout.write(dry_run_text())
        return 0
    selected = tuple(args.stack) if args.stack else STACKS
    rows: list[dict] = []
    config = LocalhostConfig(
        base_url=args.base_url,
        request_timeout_s=args.request_timeout,
        websocket_timeout_s=args.websocket_timeout,
    )
    transport = LocalhostTransport(config)
    fetch_artifact = _artifact_fetcher(config)
    for stack in selected:
        rows.extend(
            measure_stack(
                stack,
                transport=transport,
                artifact_fetcher=fetch_artifact,
            )
        )
    write_csv(Path(args.out), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
