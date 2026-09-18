# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Reference runner wrapping the unmodified H3-Base modular pipeline.

Task 1 of the bootstrap plan (MoE-Infinity#222): the reference path runs the
diffusers ``MiniMaxH3ModularPipeline`` full-resident and exposes the runner
API that later tasks re-implement with component offload + AdaLN caching.
Deterministic seeding plus latent capture (``latents`` / ``audio_latents``
requested from the modular state) make the outputs recordable as golden
fixtures for the parity gate in ``tests/test_reference_parity.py``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

RESOLUTIONS = {
    "256p": (256, 256),
    "512p": (512, 512),
    "768p": (768, 768),
}

FL2VA_COMPONENTS = (
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
    "transformer",
)

_OUTPUT_KEYS = (
    "videos",
    "audio",
    "sampling_rate",
    "latents",
    "audio_latents",
)


@dataclasses.dataclass
class GenerationResult:
    videos: Any
    audio: Any
    sampling_rate: int | None
    latents: torch.Tensor | None
    audio_latents: torch.Tensor | None


def resolve_resolution(resolution: str) -> tuple[int, int]:
    try:
        return RESOLUTIONS[resolution]
    except KeyError:
        raise ValueError(
            f"unknown resolution {resolution!r}; "
            f"choose from {sorted(RESOLUTIONS)}"
        ) from None


class ReferenceRunner:
    """Full-resident H3-Base FL2VA text-to-audio/video reference path."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "MiniMaxAI/MiniMax-H3",
        *,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        offload: bool = False,
        components: tuple[str, ...] = FL2VA_COMPONENTS,
    ) -> "ReferenceRunner":
        try:
            from diffusers import MiniMaxH3ModularPipeline
        except ImportError as exc:
            raise ImportError(
                "diffusers >= 0.40 with MiniMaxH3ModularPipeline is required "
                "for the reference runner"
            ) from exc

        components_manager = None
        if offload:
            # The full FL2VA component set (~144 GB bf16) exceeds a single
            # GPU, so the reference path can run components sequentially with
            # ComponentsManager auto CPU offload instead of .to(device).
            from diffusers.modular_pipelines import ComponentsManager

            components_manager = ComponentsManager()
            components_manager.enable_auto_cpu_offload(device=device)

        pipeline = MiniMaxH3ModularPipeline.from_pretrained(
            checkpoint, components_manager=components_manager
        )
        pipeline.load_components(
            names=list(components), torch_dtype=torch_dtype
        )
        if not offload:
            pipeline.to(device)
        return cls(pipeline)

    def generate(
        self,
        prompt: str,
        *,
        seed: int = 0,
        num_inference_steps: int = 8,
        resolution: str = "256p",
        num_frames: int = 8,
        output_type: str = "np",
    ) -> GenerationResult:
        height, width = resolve_resolution(resolution)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        state = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type=output_type,
        )
        values = {key: _state_value(state, key) for key in _OUTPUT_KEYS}
        return GenerationResult(**values)


def _state_value(state, key: str):
    getter = getattr(state, "get_intermediate", None)
    if callable(getter):
        try:
            value = getter(key)
        except Exception:
            value = None
        if value is not None:
            return value
    values = getattr(state, "values", None)
    if isinstance(values, dict):
        return values.get(key)
    return getattr(state, key, None)
