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
    return parser.parse_args()


def export_outputs(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if result.videos is not None:
        try:
            from diffusers.utils import export_to_video

            export_to_video(result.videos[0], str(output_dir / "out.mp4"))
        except ImportError:
            torch.save(result.videos, output_dir / "out_video.pt")
            print("imageio/opencv not installed; saved raw video tensor")
    if result.audios is not None:
        try:
            import soundfile

            audio = result.audios[0]
            if isinstance(audio, torch.Tensor):
                audio = audio.float().cpu().numpy()
            soundfile.write(
                str(output_dir / "out.flac"), audio.T, samplerate=40 * 1000
            )
        except ImportError:
            torch.save(result.audios, output_dir / "out_audio.pt")
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


def main() -> int:
    args = parse_args()
    runner = ReferenceRunner.from_pretrained(
        args.checkpoint, device=args.device, offload=args.offload
    )
    result = runner.generate(
        args.prompt,
        seed=args.seed,
        num_inference_steps=args.steps,
        resolution=args.resolution,
        num_frames=args.frames,
    )
    if args.record_goldens is not None:
        record_goldens(result, args.record_goldens, args)
    export_outputs(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
