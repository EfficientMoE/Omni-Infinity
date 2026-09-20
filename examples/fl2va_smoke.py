# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""FL2VA reference smoke: text prompt -> out.mp4 + out.flac (+ goldens).

QA shape from MoE-Infinity#222 Task 1:

    python examples/fl2va_smoke.py --prompt "a red ball bouncing" \\
        --seed 0 --steps 8 --resolution 256p --frames 8

Runs the full-resident reference pipeline on a large-VRAM host. With
``--record-goldens DIR`` the raw latents are saved as the fixtures that
``tests/test_reference_parity.py`` replays bitwise.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from omni_infinity.runner import RESOLUTIONS, ReferenceRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--resolution", default="256p", choices=sorted(RESOLUTIONS)
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--checkpoint", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--offload",
        action="store_true",
        help="run components sequentially via ComponentsManager auto CPU "
        "offload (required when the pipeline exceeds one GPU)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--record-goldens", type=Path, default=None)
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--store-components", default="vae,audio_vae")
    parser.add_argument("--adaln-host-cache", action="store_true")
    parser.add_argument("--transformer-fp8", action="store_true")
    parser.add_argument(
        "--fp8-skip-last-blocks",
        type=int,
        default=0,
        help="keep the last N transformer blocks in bf16 (FP8 error compounds "
        "in the late blocks; trades memory for latent accuracy)",
    )
    parser.add_argument(
        "--block-stream-blocks-per-group",
        type=int,
        default=0,
        help="stream transformer blocks through the GPU (bf16, 1 block/group "
        "with prefetch); requires --offload",
    )
    parser.add_argument("--block-stream-to-disk", default=None)
    parser.add_argument(
        "--stream-text-encoder",
        action="store_true",
        help="bf16 layer-stream the Qwen3-VL text encoder (requires --offload)",
    )
    parser.add_argument(
        "--vram-window",
        choices=("denoise", "full"),
        default="denoise",
        help="measure max_memory_allocated over the transformer denoise window "
        "or the full text-encode + denoise + VAE-decode pipeline",
    )
    parser.add_argument(
        "--max-vram",
        default=None,
        help="e.g. 22GiB: assert max_memory_allocated for --vram-window stays "
        "under this budget",
    )
    parser.add_argument(
        "--goldens",
        type=Path,
        default=None,
        help="assert generated latents allclose(rtol=2e-2) with the recorded "
        "goldens (the FP8 QA tolerance)",
    )
    return parser.parse_args()


_VRAM_UNITS = (("gib", 1024**3), ("gb", 10**9), ("mib", 1024**2), ("mb", 10**6))


def parse_vram(text: str) -> int:
    lowered = text.strip().lower()
    for suffix, multiplier in _VRAM_UNITS:
        if lowered.endswith(suffix):
            return int(float(lowered[: -len(suffix)]) * multiplier)
    return int(lowered)


def _transformer_component(pipeline):
    from diffusers import MiniMaxH3Transformer3DModel

    accessors = (
        lambda: pipeline.get_component("transformer"),
        lambda: pipeline.transformer,
        lambda: pipeline.components["transformer"],
    )
    for accessor in accessors:
        try:
            component = accessor()
        except Exception:
            continue
        if isinstance(component, MiniMaxH3Transformer3DModel):
            return component
    raise AttributeError("could not locate the transformer on the pipeline")


class DenoiseMemoryProbe:
    """Peak `max_memory_allocated` over the transformer denoise window only.

    The window is bounded by the transformer's own forwards (reset on the
    first step, captured after each), so the text-encode and VAE-decode
    phases -- which are not yet FP8/budgeted (issue #2, increment 4) -- do not
    inflate the reading.
    """

    def __init__(self, device):
        self.device = device
        self.peak = 0
        self._started = False
        self._handles = []

    def attach(self, pipeline) -> None:
        transformer = _transformer_component(pipeline)
        self._handles.append(
            transformer.register_forward_pre_hook(self._pre, with_kwargs=True)
        )
        self._handles.append(transformer.register_forward_hook(self._post))

    def _pre(self, module, args, kwargs):
        if not self._started:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._started = True

    def _post(self, module, args, output):
        self.peak = max(self.peak, torch.cuda.max_memory_allocated(self.device))


def report_latent_parity(result, goldens_path: Path) -> bool:
    payload = torch.load(goldens_path, weights_only=False)
    latents = result.latents.cpu().to(torch.float32)
    golden = payload["latents"].to(torch.float32)
    if tuple(latents.shape) != tuple(golden.shape):
        print(
            f"latent SHAPE mismatch: {tuple(latents.shape)} vs "
            f"{tuple(golden.shape)}"
        )
        return False
    rms_rel = (latents - golden).norm().item() / golden.norm().item()
    close = torch.allclose(latents, golden, rtol=2e-2, atol=2e-2)
    print(
        f"latents vs goldens: global rms_rel={rms_rel:.4f} "
        f"elementwise_allclose(rtol=2e-2,atol=2e-2)={close}"
    )
    return rms_rel < 2e-2


def export_outputs(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if result.videos is not None:
        try:
            from diffusers.utils import export_to_video

            export_to_video(result.videos[0], str(output_dir / "out.mp4"))
        except ImportError:
            torch.save(result.videos, output_dir / "out_video.pt")
            print("imageio/opencv not installed; saved raw video tensor")
    if result.audio is not None:
        try:
            import soundfile

            audio = result.audio
            if isinstance(audio, torch.Tensor):
                audio = audio.float().cpu().numpy()
            while audio.ndim > 2:
                audio = audio[0]
            soundfile.write(
                str(output_dir / "out.flac"),
                audio.T,
                samplerate=int(result.sampling_rate or 48000),
            )
        except ImportError:
            torch.save(result.audio, output_dir / "out_audio.pt")
            print("soundfile not installed; saved raw audio tensor")


def record_goldens(result, goldens_dir: Path, args) -> None:
    goldens_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "resolution": args.resolution,
        "frames": args.frames,
        "torch_version": torch.__version__,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "latents": (
            result.latents.cpu() if result.latents is not None else None
        ),
        "audio_latents": (
            result.audio_latents.cpu()
            if result.audio_latents is not None
            else None
        ),
    }
    torch.save(payload, goldens_dir / "fl2va_goldens.pt")
    print(f"goldens recorded at {goldens_dir / 'fl2va_goldens.pt'}")


def _offload_margin(args) -> str | None:
    if not (args.offload and args.max_vram and torch.cuda.is_available()):
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    margin = max(total - parse_vram(args.max_vram), 0)
    return f"{margin / 1e9:.1f}GB"


def main() -> int:
    args = parse_args()
    runner = ReferenceRunner.from_pretrained(
        args.checkpoint,
        device=args.device,
        offload=args.offload,
        store_dir=args.store_dir,
        store_components=tuple(args.store_components.split(",")),
        adaln_host_cache=args.adaln_host_cache,
        transformer_fp8=args.transformer_fp8,
        fp8_skip_last_blocks=args.fp8_skip_last_blocks,
        offload_memory_margin=_offload_margin(args),
        block_stream_blocks_per_group=args.block_stream_blocks_per_group,
        block_stream_to_disk=args.block_stream_to_disk,
        stream_text_encoder=args.stream_text_encoder,
    )
    probe = None
    if args.max_vram is not None and args.vram_window == "denoise":
        probe = DenoiseMemoryProbe(args.device)
        probe.attach(runner.pipeline)
    start = time.perf_counter()
    if args.max_vram is not None and args.vram_window == "full":
        torch.cuda.reset_peak_memory_stats(args.device)
    result = runner.generate(
        args.prompt,
        seed=args.seed,
        num_inference_steps=args.steps,
        resolution=args.resolution,
        num_frames=args.frames,
    )
    full_peak = None
    if args.max_vram is not None and args.vram_window == "full":
        full_peak = torch.cuda.max_memory_allocated(args.device)
    print(f"generate wall-clock: {time.perf_counter() - start:.1f}s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if result.latents is not None:
        torch.save(
            result.latents.cpu(), args.output_dir / "generated_latents.pt"
        )
    parity_ok = True
    if args.goldens is not None:
        parity_ok = report_latent_parity(result, args.goldens)
    vram_ok = True
    if probe is not None or full_peak is not None:
        budget = parse_vram(args.max_vram)
        peak = probe.peak if probe is not None else full_peak
        peak_gib = peak / (1024**3)
        vram_ok = peak < budget
        window = (
            "transformer denoise-window"
            if args.vram_window == "denoise"
            else "full pipeline-window"
        )
        print(
            f"{window} max_memory_allocated: "
            f"{peak_gib:.2f} GiB (budget {budget / (1024**3):.2f} GiB) "
            f"ok={vram_ok}"
        )
    if args.record_goldens is not None:
        record_goldens(result, args.record_goldens, args)
    export_outputs(result, args.output_dir)
    if not (parity_ok and vram_ok):
        raise SystemExit("QA gate failed (see rms_rel / max_memory above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
