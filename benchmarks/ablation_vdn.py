# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Run the VDN-H3 grid or the resident P2 optimization ablation.

The grid compares the upstream native inference stack with Omni-Infinity's
portable diffusers stack.  Importing this module and using ``--dry-run`` are
CPU-only operations; PyTorch is imported lazily only to compare saved latents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import statistics
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

FRAMES = 222
SEED = 0
VDN_HUB_REPO = "OpenVDN/vdn-minimax-h3"
PROMPT = "a red ball bouncing"
P2_FRAMES = 120
P2_RESOLUTION = "256p"
P2_CHECKPOINT = pathlib.Path(
    "/mnt/raid0nvme0/leyang/.cache/huggingface/hub/"
    "models--MiniMaxAI--MiniMax-H3/snapshots/"
    "42ed227ee7df40d41602854ae760620d6eb651fe"
)
P2_STORE = pathlib.Path("/mnt/raid0nvme0/leyang/h3-store-v2")
CSV_FIELDS = (
    "name",
    "stack",
    "s_per_nfe",
    "peak_gib",
    "rms_rel",
    "cosine",
    "notes",
    "config_json",
)


@dataclass(frozen=True)
class Run:
    """One declarative point in the ablation grid."""

    name: str
    stack: str
    tags: tuple[str, ...]
    params: dict[str, Any]


GRID = [
    Run(
        "dense-8nfe",
        "upstream",
        ("arch",),
        {"config": "8nfe.yaml", "checkpoint": None, "backend": "flex"},
    ),
    Run(
        "hybrid-8nfe",
        "upstream",
        ("arch",),
        {
            "config": "8nfe.yaml",
            "checkpoint": "stage-dmd-step-250",
            "backend": "flex",
        },
    ),
    Run(
        "dense-turbo-lora-8nfe",
        "upstream",
        ("arch", "lora"),
        {
            "config": "8nfe.yaml",
            "checkpoint": None,
            "backend": "flex",
            "external_lora": True,
        },
    ),
    Run(
        "hybrid-50ckpt-50nfe",
        "upstream",
        ("nfe",),
        {"config": "50nfe.yaml", "checkpoint": "stage-b-step-2000"},
    ),
    Run(
        "hybrid-50ckpt-8nfe",
        "upstream",
        ("nfe", "lora"),
        {
            "config": "8nfe.yaml",
            "checkpoint": "stage-b-step-2000",
            "num_steps": 8,
        },
    ),
    Run(
        "hybrid-8nfe-tuned",
        "upstream",
        ("kernels",),
        {"config": "8nfe_tuned.yaml", "checkpoint": "stage-dmd-step-250"},
    ),
    Run(
        "hybrid-8nfe-tuned-fp8",
        "upstream",
        ("precision",),
        {
            "config": "8nfe_tuned_fp8.yaml",
            "checkpoint": "stage-dmd-step-250",
            "fp8": True,
        },
    ),
    Run(
        "hybrid-8nfe-tuned-fp8-skip4",
        "upstream",
        ("precision",),
        {
            "config": "8nfe_tuned_fp8.yaml",
            "checkpoint": "stage-dmd-step-250",
            "fp8": True,
            "fp8_skip": 4,
        },
    ),
    Run(
        "hybrid-8nfe-flex",
        "upstream",
        ("backend",),
        {
            "config": "8nfe_tuned.yaml",
            "checkpoint": "stage-dmd-step-250",
            "backend": "flex",
        },
    ),
    Run(
        "hybrid-8nfe-decomposed",
        "upstream",
        ("backend",),
        {
            "config": "8nfe_tuned.yaml",
            "checkpoint": "stage-dmd-step-250",
            "backend": "decomposed",
        },
    ),
    Run(
        "omni-resident-bf16",
        "omni",
        ("memory",),
        {"offload": True},
    ),
    Run(
        "omni-stream-bf16",
        "omni",
        ("memory",),
        {"offload": True, "block_stream": 1, "stream_text_encoder": True},
    ),
    Run(
        "omni-stream-fp8",
        "omni",
        ("memory", "precision"),
        {
            "offload": True,
            "block_stream": 1,
            "stream_text_encoder": True,
            "fp8": True,
        },
    ),
    Run(
        "omni-stream-groups4",
        "omni",
        ("memory",),
        {"offload": True, "block_stream": 4, "stream_text_encoder": True},
    ),
    Run(
        "omni-50step-bf16",
        "omni",
        ("nfe",),
        {
            "offload": True,
            "block_stream": 1,
            "stream_text_encoder": True,
            "variant": "50-step",
            "evals": 50,
        },
    ),
    Run(
        "omni-notes-textenc",
        "omni",
        ("memory",),
        {"offload": True, "block_stream": 1},
    ),
]

P2_CONFIG_OPTIMIZATIONS = {
    "baseline": ["adaln-host-cache"],
    "compile": ["adaln-host-cache", "compile-blocks"],
    "graph": ["adaln-host-cache", "cuda-graph"],
    "compile+graph": [
        "adaln-host-cache",
        "compile-blocks",
        "cuda-graph",
    ],
}
P2_GRID = [
    Run(
        f"p2-{config.replace('+', '-')}-{nfe}nfe",
        "p2",
        ("p2",),
        {
            "config": config,
            "nfe": nfe,
            "optimizations": optimizations,
        },
    )
    for config, optimizations in P2_CONFIG_OPTIMIZATIONS.items()
    for nfe in (8, 16)
]


def _path(value: os.PathLike[str] | str) -> pathlib.Path:
    return pathlib.Path(value).expanduser().resolve()


def build_command(
    run: Run,
    results_dir: os.PathLike[str] | str,
    vdn_dir: os.PathLike[str] | str,
    ckpts: os.PathLike[str] | str,
) -> list[str]:
    """Build one subprocess argv without importing GPU libraries."""
    results = _path(results_dir)
    checkpoints = _path(ckpts)
    if run.stack == "p2":
        return build_p2_cell_command(run, mode="timing", results_dir=results)
    if run.stack == "upstream":
        out = results / f"{run.name}.mp4"
        checkpoint = run.params.get("checkpoint")
        argv = [
            "python",
            "src/inference/infer.py",
            "--config",
            f"configs/inference/{run.params['config']}",
            f"checkpoint={checkpoints / checkpoint}"
            if checkpoint
            else "checkpoint=null",
        ]
        if run.params.get("external_lora"):
            lora = checkpoints / (
                "external/minimax_h3_turbo_v4_step600_ema.safetensors"
            )
            argv.append(f"external_loras=[{{path: {lora}}}]")
        if backend := run.params.get("backend"):
            argv.append(f"kernels.softmax_backend={backend}")
        if "num_steps" in run.params:
            argv.append(f"render.num_steps={run.params['num_steps']}")
        if "fp8_skip" in run.params:
            argv.append(
                "precision.fp8.skip_end_blocks=" f"{run.params['fp8_skip']}"
            )
        argv.extend(
            [
                "render.prompt_file=prompts/example_0.pt",
                f"render.out={out}",
                f"render.num_frames={FRAMES}",
                f"render.seed={SEED}",
                "render.warmup_steps=2",
                "render.record=true",
                "render.save_latents=true",
                f"render.latents_root={results / 'latents'}",
            ]
        )
        return argv

    if run.stack != "omni":
        raise ValueError(f"unknown stack: {run.stack}")
    argv = [
        "python",
        "examples/vdn_smoke.py",
        "--prompt",
        PROMPT,
        # Hub repo-id, NOT the local dir: the vendored diffusers
        # strips trust_remote_code when the pipeline repo differs
        # from the remote-code transformer repo (see
        # docs/repro_vdn.md, cross-repo guard). Resolved offline
        # from the pre-populated HF_HOME cache.
        "--checkpoint",
        VDN_HUB_REPO,
        "--variant",
        str(run.params.get("variant", "8-step")),
        "--seed",
        str(SEED),
        "--evals",
        str(run.params.get("evals", 8)),
        "--frames",
        str(FRAMES),
    ]
    if run.params.get("fp8"):
        argv.append("--fp8")
    if backend := run.params.get("backend"):
        argv.extend(["--softmax-backend", str(backend)])
    if run.params.get("offload"):
        argv.append("--offload")
    if block_stream := run.params.get("block_stream"):
        argv.extend(["--block-stream", str(block_stream)])
    if run.params.get("stream_text_encoder"):
        argv.append("--stream-text-encoder")
    # No --goldens in grid runs: the fixture is 120-frame; grid runs
    # are 222-frame (temporal dims differ -> hard shape error). Bitwise
    # parity is covered by tests/test_vdn_parity.py and the streaming
    # parity check in docs/repro_vdn.md instead.
    argv.extend(["--metrics-out", str(results / f"{run.name}.metrics.json")])
    return argv


def build_p2_cell_command(
    run: Run,
    *,
    mode: str,
    results_dir: os.PathLike[str] | str,
    checkpoint: os.PathLike[str] | str = P2_CHECKPOINT,
    store_dir: os.PathLike[str] | str = P2_STORE,
    first_frame: os.PathLike[str] | str | None = None,
    goldens: os.PathLike[str] | str | None = None,
) -> list[str]:
    """Build one isolated P2 timing or parity subprocess command."""
    if run.stack != "p2":
        raise ValueError(f"not a P2 run: {run.name}")
    if mode not in {"timing", "parity"}:
        raise ValueError(f"unknown P2 cell mode: {mode}")
    repo = pathlib.Path(__file__).resolve().parents[1]
    argv = [
        "python",
        "benchmarks/ablation_vdn.py",
        "--p2-cell",
        mode,
        "--p2-config",
        str(run.params["config"]),
        "--steps",
        str(run.params["nfe"]),
        "--results-dir",
        str(_path(results_dir)),
        "--checkpoint",
        str(_path(checkpoint)),
        "--store-dir",
        str(_path(store_dir)),
    ]
    argv.extend(
        [
            "--first-frame",
            str(
                _path(first_frame)
                if first_frame is not None
                else repo / "tests" / "fixtures" / "ref.png"
            ),
            "--goldens",
            str(
                _path(goldens)
                if goldens is not None
                else repo
                / "tests"
                / "fixtures"
                / "goldens"
                / "fl2va_goldens.pt"
            ),
        ]
    )
    return argv


def _sample_peak_vram(
    gpu: int, stop: threading.Event, samples: list[float]
) -> None:
    while not stop.is_set():
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            try:
                samples.append(float(result.stdout.strip().splitlines()[0]))
            except (IndexError, ValueError):
                pass
        stop.wait(0.2)


def _run_subprocess(
    argv: list[str],
    cwd: pathlib.Path,
    env: dict[str, str],
    gpu: int,
    sample_vram: bool,
) -> float | None:
    stop = threading.Event()
    samples: list[float] = []
    sampler = None
    if sample_vram:
        sampler = threading.Thread(
            target=_sample_peak_vram,
            args=(gpu, stop, samples),
            daemon=True,
        )
        sampler.start()
    try:
        subprocess.run(argv, cwd=cwd, env=env, check=True)
    finally:
        stop.set()
        if sampler is not None:
            sampler.join()
    return max(samples) / 1024 if samples else None


def _upstream_latents(results_dir: pathlib.Path, name: str) -> pathlib.Path:
    return results_dir / "latents" / f"{name}.mp4.latents.pt"


def compare_latents(
    candidate: pathlib.Path, reference: pathlib.Path
) -> tuple[float, float]:
    """Return relative RMS error and cosine similarity for video latents."""
    import torch

    got = torch.load(candidate, map_location="cpu", weights_only=False)["video"]
    ref = torch.load(reference, map_location="cpu", weights_only=False)["video"]
    got = got.float().flatten()
    ref = ref.float().flatten()
    rms_rel = (((got - ref) ** 2).mean().sqrt() / (ref**2).mean().sqrt()).item()
    cosine = torch.nn.functional.cosine_similarity(
        got.unsqueeze(0), ref.unsqueeze(0)
    ).item()
    return rms_rel, cosine


def _read_upstream_metrics(record_path: pathlib.Path) -> float:
    payload = json.loads(record_path.read_text())
    step_seconds = payload["timings"]["step_seconds"]
    if not step_seconds:
        raise ValueError(f"no evaluation timings in {record_path}")
    return statistics.mean(float(value) for value in step_seconds)


def _read_omni_metrics(path: pathlib.Path) -> tuple[float, float, Any]:
    payload = json.loads(path.read_text())
    seconds = payload.get("s_per_eval")
    if seconds is None:
        seconds = payload["s_per_eval_incl_overhead"]
    return float(seconds), float(payload["peak_gib"]), payload.get("rms_rel")


def _p2_artifact_path(
    results: pathlib.Path, config: str, nfe: int, mode: str
) -> pathlib.Path:
    slug = config.replace("+", "-")
    return results / f"{slug}-nfe{nfe}-{mode}.json"


def _p2_graph_execution(
    optimizations: list[str], graph_stats: dict[str, Any] | None
) -> tuple[bool | None, str]:
    if "cuda-graph" not in optimizations:
        return None, "not requested"
    if graph_stats is None:
        return False, "graph manager unavailable"
    fallback_reasons = {
        reason: int(count)
        for reason, count in graph_stats["fallback_reasons"].items()
        if count and reason not in {"warmup_not_done", "shape_bucket_miss"}
    }
    replayed = bool(
        graph_stats["captures"] >= 1
        and graph_stats["replays"] >= 1
        and graph_stats["capture_failures"] == 0
        and graph_stats["graphs"] >= 1
        and not fallback_reasons
    )
    if replayed:
        return True, "captured and replayed"
    if fallback_reasons:
        reasons = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(fallback_reasons.items())
        )
        return False, f"graph fallback: {reasons}"
    return False, (
        "graph fallback: "
        f"captures={graph_stats['captures']}, "
        f"replays={graph_stats['replays']}, "
        f"capture_failures={graph_stats['capture_failures']}, "
        f"graphs={graph_stats['graphs']}"
    )


def _p2_cell_gate_passes(
    *,
    config: str,
    mode: str,
    optimizations: list[str],
    provenance: dict[str, Any] | None,
    parity: dict[str, Any] | None,
    graph_replayed: bool | None,
) -> bool:
    if "cuda-graph" in optimizations and graph_replayed is not True:
        return False
    if mode != "parity":
        return True
    if provenance is None or not provenance["comparable"]:
        return False
    if config in {"baseline", "graph"}:
        return bool(parity is not None and parity["bitwise"])
    return bool(
        parity is not None
        and parity.get("allclose", {}).get("rtol=atol=2e-2", False)
    )


def _is_expected_negative_parity(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    parity = payload.get("parity") or {}
    provenance = payload.get("golden_provenance") or {}
    config = payload.get("config")
    return bool(
        config in {"compile", "compile+graph"}
        and (config != "compile+graph" or payload.get("graph_replayed") is True)
        and payload.get("golden_gate_passed") is False
        and provenance.get("comparable") is True
        and not parity.get("allclose", {}).get("rtol=atol=2e-2", False)
    )


def _parse_nvidia_driver_version(banner: str) -> str:
    match = re.search(r"Kernel Module for \S+\s+([0-9.]+)", banner)
    return match.group(1) if match else "unknown"


def _p2_environment(args, torch) -> dict[str, Any]:
    import diffusers

    driver_path = pathlib.Path("/proc/driver/nvidia/version")
    driver = "unknown"
    if driver_path.is_file():
        driver = _parse_nvidia_driver_version(
            driver_path.read_text(encoding="utf-8")
        )
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "cuda": torch.version.cuda,
        "driver": driver,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "checkpoint": str(args.checkpoint),
        "store_dir": str(args.store_dir),
    }


def _run_p2_cell(args) -> int:
    """Run one GPU-isolated P2 cell; imported dependencies stay lazy."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    if args.steps < 2:
        raise SystemExit("--steps must be at least two")
    optimizations = P2_CONFIG_OPTIMIZATIONS[args.p2_config]
    compile_cache = None
    if "compile-blocks" in optimizations:
        compile_cache = tempfile.mkdtemp(prefix="p2-inductor-")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = compile_cache

    import torch

    from benchmarks.bench_graph_denoise import (
        GraphDenoiseStepProbe,
        replay_summary,
    )
    from benchmarks.profile_denoise_step import (
        DenoiseStepProbe,
        analyze_steps,
        effective_video_frames,
        golden_provenance,
        latent_parity,
        patched_transformer_config_lookup,
        summarize,
    )
    from omni_infinity.registry import resolve_profile
    from omni_infinity.runner import _transformer_component

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("exactly one visible CUDA device is required")
    resolved = resolve_profile(
        "h3-dense", optimizations, checkpoint=str(args.checkpoint)
    )
    runner_kwargs = dict(resolved.runner_kwargs)
    runner_kwargs.update(
        {
            "offload": True,
            "store_dir": str(args.store_dir),
            "store_components": ("transformer", "vae", "audio_vae"),
        }
    )
    print(
        f"loading P2 config={args.p2_config} mode={args.p2_cell} "
        f"nfe={args.steps} optimizations={optimizations} "
        f"on {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    with patched_transformer_config_lookup(args.checkpoint):
        runner = resolved.runner.from_pretrained(
            resolved.checkpoint,
            device="cuda",
            torch_dtype=torch.bfloat16,
            **runner_kwargs,
        )
    transformer = _transformer_component(runner.pipeline)
    manager = runner.cuda_graph_manager
    from PIL import Image

    first_frame = Image.open(args.first_frame).convert("RGB")
    started = datetime.now(timezone.utc)
    wall_started = __import__("time").perf_counter()
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    if args.p2_cell == "timing":
        from torch.profiler import ProfilerActivity, profile

        if manager is None:
            probe = DenoiseStepProbe(transformer, args.steps)
        else:
            probe = GraphDenoiseStepProbe(transformer, args.steps, manager)
        with probe:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                acc_events=True,
            ) as profiler:
                result = runner.generate(
                    PROMPT,
                    image=first_frame,
                    seed=SEED,
                    num_inference_steps=args.steps,
                    resolution=P2_RESOLUTION,
                    num_frames=P2_FRAMES,
                )
        torch.cuda.synchronize()
        rows = analyze_steps(profiler.events(), probe.steps)
        if manager is not None:
            for row, phase in zip(rows, probe.phases, strict=True):
                row["phase"] = phase
            replay = replay_summary(rows)
            summary = replay if replay["steps"] else summarize(rows)
            summary["measurement_scope"] = (
                "steady graph replays"
                if replay["steps"]
                else "post-warmup eager fallback"
            )
        else:
            summary = summarize(rows)
            summary["measurement_scope"] = "post-first-forward median"
        median_wall_ms = float(summary["median_step_wall_ms"])
        peak_gib = max(float(row["peak_allocated_gib"]) for row in rows)
        rms_rel = None
        cosine = None
        parity = None
        provenance = None
    else:
        result = runner.generate(
            PROMPT,
            image=first_frame,
            seed=SEED,
            num_inference_steps=args.steps,
            resolution=P2_RESOLUTION,
            num_frames=P2_FRAMES,
            step_callback=lambda completed, total: print(
                f"denoise progress {completed}/{total}", flush=True
            ),
        )
        torch.cuda.synchronize()
        golden = torch.load(
            args.goldens, map_location="cpu", weights_only=False
        )
        parity = latent_parity(result.latents, golden["latents"])
        with tempfile.TemporaryDirectory(prefix="p2-latents-") as temporary:
            temporary_path = pathlib.Path(temporary)
            candidate = temporary_path / "candidate.pt"
            reference = temporary_path / "reference.pt"
            torch.save({"video": result.latents.detach().cpu()}, candidate)
            torch.save({"video": golden["latents"]}, reference)
            rms_rel, cosine = compare_latents(candidate, reference)
        import diffusers

        first_frame_sha256 = hashlib.sha256(
            args.first_frame.read_bytes()
        ).hexdigest()
        provenance = golden_provenance(
            golden,
            torch_version=torch.__version__,
            diffusers_version=diffusers.__version__,
            gpu_name=torch.cuda.get_device_name(0),
            first_frame_sha256=first_frame_sha256,
            prompt=PROMPT,
            seed=SEED,
            steps=args.steps,
            resolution=P2_RESOLUTION,
            frames=P2_FRAMES,
        )
        median_wall_ms = 0.0
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    elapsed = __import__("time").perf_counter() - wall_started
    graph_stats = manager.stats_snapshot() if manager is not None else None
    graph_replayed, graph_note = _p2_graph_execution(optimizations, graph_stats)
    execution_gate_passed = _p2_cell_gate_passes(
        config=args.p2_config,
        mode=args.p2_cell,
        optimizations=optimizations,
        provenance=provenance,
        parity=parity,
        graph_replayed=graph_replayed,
    )
    payload = {
        "schema_version": 1,
        "timestamp_utc": started.isoformat(),
        "mode": args.p2_cell,
        "config": args.p2_config,
        "model_arch": resolved.model_arch,
        "optimizations": optimizations,
        "parameters": {
            "prompt": PROMPT,
            "first_frame": str(args.first_frame),
            "steps": args.steps,
            "actual_transformer_forwards": (
                len(rows) if args.p2_cell == "timing" else None
            ),
            "resolution": P2_RESOLUTION,
            "frames": P2_FRAMES,
            "effective_frames": effective_video_frames(P2_FRAMES),
            "seed": SEED,
            "goldens": str(args.goldens) if args.p2_cell == "parity" else None,
        },
        "environment": _p2_environment(args, torch),
        "full_generation_wall_seconds": elapsed,
        "s_per_eval": median_wall_ms / 1e3,
        "peak_gib": peak_gib,
        "rms_rel": rms_rel,
        "cosine": cosine,
        "parity": parity,
        "golden_provenance": provenance,
        "summary": summary,
        "steps": rows,
        "pinned_adaln_gib": runner.pinned_adaln_bytes / 1024**3,
        "graph_stats": graph_stats,
        "graph_replayed": graph_replayed,
        "graph_note": graph_note,
        "execution_gate_passed": execution_gate_passed,
        "golden_gate_passed": (
            execution_gate_passed if args.p2_cell == "parity" else None
        ),
    }
    output = _p2_artifact_path(
        _path(args.results_dir), args.p2_config, args.steps, args.p2_cell
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}", flush=True)
    if args.p2_cell == "timing":
        print(
            f"timing median={median_wall_ms:.2f} ms peak={peak_gib:.2f} GiB; "
            f"{graph_note}",
            flush=True,
        )
    else:
        print(
            f"parity tier={parity['tier']} rms_rel={rms_rel:.7g} "
            f"cosine={cosine:.9f}; {graph_note}",
            flush=True,
        )
    if not execution_gate_passed:
        print("P2 execution/provenance gate failed", flush=True)
        return 1
    if compile_cache is not None:
        shutil.rmtree(compile_cache, ignore_errors=True)
    return 0


def execute_p2(
    run: Run,
    results_dir: os.PathLike[str] | str,
    *,
    timing_gpu: int,
    parity_gpu: int,
    checkpoint: os.PathLike[str] | str = P2_CHECKPOINT,
    store_dir: os.PathLike[str] | str = P2_STORE,
    first_frame: os.PathLike[str] | str | None = None,
    goldens: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Execute one timing cell and its NFE-8 parity companion."""
    results = _path(results_dir)
    repo = pathlib.Path(__file__).resolve().parents[1]
    nfe = int(run.params["nfe"])
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(timing_gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": str(repo),
        }
    )
    timing_command = build_p2_cell_command(
        run,
        mode="timing",
        results_dir=results,
        checkpoint=checkpoint,
        store_dir=store_dir,
        first_frame=first_frame,
        goldens=goldens,
    )
    _run_subprocess(timing_command, repo, env, timing_gpu, sample_vram=False)
    if nfe == 8:
        parity_env = env.copy()
        parity_env["CUDA_VISIBLE_DEVICES"] = str(parity_gpu)
        parity_command = build_p2_cell_command(
            run,
            mode="parity",
            results_dir=results,
            checkpoint=checkpoint,
            store_dir=store_dir,
            first_frame=first_frame,
            goldens=goldens,
        )
        try:
            _run_subprocess(
                parity_command,
                repo,
                parity_env,
                parity_gpu,
                sample_vram=False,
            )
        except subprocess.CalledProcessError:
            parity_path = _p2_artifact_path(
                results,
                str(run.params["config"]),
                nfe,
                "parity",
            )
            if not _is_expected_negative_parity(parity_path):
                raise
            print(
                f"collected expected failed parity gate from {parity_path}",
                flush=True,
            )
    return collect_p2_row(run, results)


def collect_p2_row(
    run: Run, results_dir: os.PathLike[str] | str
) -> dict[str, Any]:
    """Collect one P2 CSV row from already measured JSON artifacts."""
    if run.stack != "p2":
        raise ValueError(f"not a P2 run: {run.name}")
    results = _path(results_dir)
    config = str(run.params["config"])
    nfe = int(run.params["nfe"])
    timing_path = _p2_artifact_path(results, config, nfe, "timing")
    seconds, peak_gib, _ = _read_omni_metrics(timing_path)
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    notes = [str(timing["summary"]["measurement_scope"])]
    if source := timing.get("canonical_source_artifact"):
        source_path = pathlib.Path(source)
        if not source_path.is_absolute():
            repo = pathlib.Path(__file__).resolve().parents[1]
            source_path = repo / source_path
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        source_summary = source_payload.get("replay_summary")
        if source_summary is None:
            source_summary = source_payload["summary"]
        seconds = float(source_summary["median_step_wall_ms"]) / 1e3
        notes.append(f"canonical timing source={source}")
    if "cuda-graph" in run.params["optimizations"]:
        notes.append(str(timing["graph_note"]))
    rms_rel = None
    cosine = None
    if nfe == 8:
        parity_path = _p2_artifact_path(results, config, nfe, "parity")
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        rms_rel = parity["rms_rel"]
        cosine = parity["cosine"]
        notes.append(f"parity={parity['parity']['tier']}")
        if "cuda-graph" in run.params["optimizations"]:
            notes.append(str(parity["graph_note"]))
    else:
        notes.append("no golden at this NFE")
    return {
        "name": run.name,
        "stack": run.stack,
        "s_per_nfe": seconds,
        "peak_gib": peak_gib,
        "rms_rel": rms_rel,
        "cosine": cosine,
        "notes": "; ".join(notes),
        "config_json": json.dumps(run.params, sort_keys=True),
    }


def execute(
    run: Run,
    results_dir: os.PathLike[str] | str,
    vdn_dir: os.PathLike[str] | str,
    ckpts: os.PathLike[str] | str,
    gpu: int = 0,
) -> dict[str, Any]:
    """Execute one run and return a row suitable for ``results.csv``."""
    results = _path(results_dir)
    upstream = _path(vdn_dir)
    repo = pathlib.Path(__file__).resolve().parents[1]
    argv = build_command(run, results, upstream, ckpts)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if run.stack == "omni":
        env.setdefault(
            "HF_HOME",
            "/mnt/raid0nvme0/leyang/.cache/huggingface",
        )
    cwd = upstream if run.stack == "upstream" else repo
    peak_gib = _run_subprocess(
        argv, cwd, env, gpu, sample_vram=run.stack == "upstream"
    )
    notes: list[str] = []
    rms_rel = None
    cosine = None
    if run.stack == "upstream":
        record = results / f"{run.name}.mp4.inference.json"
        seconds = _read_upstream_metrics(record)
        candidate = _upstream_latents(results, run.name)
        reference = _upstream_latents(results, "hybrid-8nfe")
        if candidate.exists() and reference.exists():
            rms_rel, cosine = compare_latents(candidate, reference)
        if run.params.get("fp8"):
            notes.append("indicative: fp8 changes the sample")
    else:
        metrics = results / f"{run.name}.metrics.json"
        seconds, peak_gib, rms_rel = _read_omni_metrics(metrics)
        notes.append("no warmup flag in vdn_smoke.py")
    return {
        "name": run.name,
        "stack": run.stack,
        "s_per_nfe": seconds,
        "peak_gib": peak_gib,
        "rms_rel": rms_rel,
        "cosine": cosine,
        "notes": "; ".join(notes),
        "config_json": json.dumps(run.params, sort_keys=True),
    }


def _read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _refresh_upstream_quality(
    rows: list[dict[str, Any]], results: pathlib.Path
) -> None:
    reference = _upstream_latents(results, "hybrid-8nfe")
    if not reference.exists():
        return
    run_by_name = {run.name: run for run in GRID}
    for row in rows:
        if row["stack"] != "upstream":
            continue
        candidate = _upstream_latents(results, str(row["name"]))
        if not candidate.exists():
            continue
        row["rms_rel"], row["cosine"] = compare_latents(candidate, reference)
        run = run_by_name[str(row["name"])]
        if run.params.get("fp8") and "indicative" not in str(row["notes"]):
            row["notes"] = "indicative: fp8 changes the sample"


def _dry_run_line(
    run: Run, argv: list[str], vdn_dir: pathlib.Path, gpu: int
) -> str:
    repo = pathlib.Path(__file__).resolve().parents[1]
    cwd = vdn_dir if run.stack == "upstream" else repo
    note = ""
    if run.stack == "omni":
        note = "; note=no warmup flag in vdn_smoke.py"
    return (
        f"{shlex.join(argv)} # cwd={cwd}; "
        f"env=CUDA_VISIBLE_DEVICES={gpu}{note}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/vdn/ablation")
    parser.add_argument("--vdn-dir", default="third_party/vdn-minimax-h3")
    parser.add_argument("--ckpts", default="ckpts/vdn")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timing-gpu", type=int, default=4)
    parser.add_argument("--parity-gpu", type=int, default=0)
    parser.add_argument("--suite", choices=("vdn", "p2"), default="vdn")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only", choices=[run.name for run in (*GRID, *P2_GRID)]
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--p2-cell", choices=("timing", "parity"))
    parser.add_argument("--p2-config", choices=tuple(P2_CONFIG_OPTIMIZATIONS))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--checkpoint", type=pathlib.Path, default=P2_CHECKPOINT
    )
    parser.add_argument("--store-dir", type=pathlib.Path, default=P2_STORE)
    repo = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--first-frame",
        type=pathlib.Path,
        default=repo / "tests" / "fixtures" / "ref.png",
    )
    parser.add_argument(
        "--goldens",
        type=pathlib.Path,
        default=repo / "tests" / "fixtures" / "goldens" / "fl2va_goldens.pt",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.p2_cell is not None:
        if args.p2_config is None:
            raise SystemExit("--p2-cell requires --p2-config")
        return _run_p2_cell(args)
    results = _path(args.results_dir)
    vdn_dir = _path(args.vdn_dir)
    grid = GRID if args.suite == "vdn" else P2_GRID
    runs = [run for run in grid if args.only in (None, run.name)]
    if args.dry_run:
        for run in runs:
            if run.stack == "p2":
                timing = build_p2_cell_command(
                    run,
                    mode="timing",
                    results_dir=results,
                    checkpoint=args.checkpoint,
                    store_dir=args.store_dir,
                    first_frame=args.first_frame,
                    goldens=args.goldens,
                )
                print(_dry_run_line(run, timing, vdn_dir, args.timing_gpu))
                if run.params["nfe"] == 8:
                    parity = build_p2_cell_command(
                        run,
                        mode="parity",
                        results_dir=results,
                        checkpoint=args.checkpoint,
                        store_dir=args.store_dir,
                        first_frame=args.first_frame,
                        goldens=args.goldens,
                    )
                    print(_dry_run_line(run, parity, vdn_dir, args.parity_gpu))
            else:
                argv = build_command(run, results, vdn_dir, args.ckpts)
                print(_dry_run_line(run, argv, vdn_dir, args.gpu))
        return 0

    results.mkdir(parents=True, exist_ok=True)
    csv_path = results / "results.csv"
    rows: list[dict[str, Any]] = _read_rows(csv_path)
    completed = {row["name"] for row in rows}
    for run in runs:
        if run.name in completed and not args.force:
            continue
        if args.collect_only:
            if run.stack != "p2":
                raise SystemExit(
                    "--collect-only is supported only for --suite p2"
                )
            row = collect_p2_row(run, results)
        elif run.stack == "p2":
            row = execute_p2(
                run,
                results,
                timing_gpu=args.timing_gpu,
                parity_gpu=args.parity_gpu,
                checkpoint=args.checkpoint,
                store_dir=args.store_dir,
                first_frame=args.first_frame,
                goldens=args.goldens,
            )
        else:
            row = execute(run, results, vdn_dir, args.ckpts, args.gpu)
        rows = [old for old in rows if old["name"] != run.name]
        rows.append(row)
        if args.suite == "vdn":
            _refresh_upstream_quality(rows, results)
        _write_rows(csv_path, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
