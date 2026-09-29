# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""VDN-Minimax-H3 hybrid-attention runner (model-arch: vdn-hybrid).

Loads the published diffusers remote-code component
(``OpenVDN/vdn-minimax-h3``, remote file ``modeling_vdn_h3.py``): the
66 GB H3 base transformer plus the checkpoint's linear-attention branch
and merged LoRA adapters. Two released variants:

- ``8-step``  -- ``stage-dmd-step-250`` (default of the Hub pipeline
  index; ``default`` + ``turbo`` LoRA, 8 model evaluations)
- ``50-step`` -- ``stage-b-step-2000`` (selected per component via
  ``subfolder``; 50 model evaluations)

``num_inference_steps`` upstream counts sigma grid points (= model
evaluations + 1); this runner's API speaks *evaluations* and adds the
one, mirroring upstream's own ``--steps`` flag.

The VDN canvas is fixed at 1344x768; unlike ``ReferenceRunner`` there
is no resolution knob, only ``num_frames`` (120-360, snapped by the
pipeline to 17n+5).
"""

from __future__ import annotations

from typing import Any

import torch

from omni_infinity.runner import (
    _OUTPUT_KEYS,
    GenerationResult,
    StepCallback,
    _capture_condition,
    _consult_condition_cache,
    _denoise_cache_context,
    _denoising_progress,
    _state_value,
)

_VARIANTS = {
    "8-step": {"subfolder": None, "evaluations": 8},
    "50-step": {
        "subfolder": "stage-b-step-2000/diffusers",
        "evaluations": 50,
    },
}


def _sigma_grid_points(evaluations: int) -> int:
    # Upstream counts sigma grid points, the terminal 0 included.
    return evaluations + 1


def _load_modular_pipeline(checkpoint: str, workflow: str):
    from diffusers import ModularPipeline

    return ModularPipeline.from_pretrained(checkpoint, workflow=workflow)


class VdnRunner:
    """VDN-H3 T2VA runner over the published diffusers component."""

    transformer_component = "transformer"

    def __init__(
        self,
        pipeline,
        default_evaluations: int,
        condition_cache=None,
        condition_namespace: str = "",
        vision_cache_controller=None,
        device=None,
    ):
        self.pipeline = pipeline
        self.default_evaluations = default_evaluations
        self.condition_cache = condition_cache
        self.condition_namespace = condition_namespace
        self.vision_cache_controller = vision_cache_controller
        self.device = device
        self._conditioned_pipeline = None
        self._conditioned_pipeline_built = False

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "OpenVDN/vdn-minimax-h3",
        *,
        variant: str = "8-step",
        workflow: str = "t2va",
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        fp8: bool = False,
        softmax_backend: str | None = None,
        offload: bool = False,
        block_stream_blocks_per_group: int = 0,
        stream_text_encoder: bool = False,
        condition_cache: bool = False,
        condition_cache_dir: str | None = None,
        condition_cache_entries: int = 8,
        vision_cache: bool = False,
        vision_cache_entries: int = 4,
    ) -> "VdnRunner":
        try:
            spec = _VARIANTS[variant]
        except KeyError:
            raise ValueError(
                f"unknown variant {variant!r}; "
                f"choose from {sorted(_VARIANTS)}"
            ) from None
        if block_stream_blocks_per_group and not offload:
            raise ValueError("block streaming requires offload=True")
        if stream_text_encoder and not offload:
            raise ValueError("stream_text_encoder requires offload=True")

        pipeline = _load_modular_pipeline(checkpoint, workflow)
        load_kwargs: dict = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        if spec["subfolder"]:
            load_kwargs["subfolder"] = {"transformer": spec["subfolder"]}
        if fp8:
            # Upstream preset: torchao fp8 e4m3 on every Linear >=4096
            # wide on both sides (363 Linears), applied after the LoRA
            # merge. Needs CC >= 9.0.
            load_kwargs["fp8"] = {"transformer": True}
        if softmax_backend:
            load_kwargs["softmax_backend"] = {"transformer": softmax_backend}
        pipeline.load_components(**load_kwargs)
        if pipeline.transformer is None:
            # load_components downgrades component errors to warnings.
            raise RuntimeError(
                "the VDN transformer failed to load; see the diffusers "
                "warning above for the cause"
            )

        if offload:
            _offload_vdn(
                pipeline,
                device,
                block_stream_blocks_per_group,
                stream_text_encoder,
            )
        else:
            pipeline.to(device)
        # Issue #24 opt-in caches (off by default, off the parity gates).
        cache = None
        if condition_cache:
            from omni_infinity.caches.condition import ConditionCache

            cache = ConditionCache(
                max_entries=condition_cache_entries,
                cache_dir=condition_cache_dir,
            )
        vision_controller = None
        if vision_cache:
            from omni_infinity.caches.vision import enable_vision_cache

            vision_controller = enable_vision_cache(
                pipeline.text_encoder, max_entries=vision_cache_entries
            )
        return cls(
            pipeline,
            spec["evaluations"],
            condition_cache=cache,
            condition_namespace=(
                f"vdn-hybrid|{checkpoint}|{variant}|{workflow}|{torch_dtype}"
            ),
            vision_cache_controller=vision_controller,
            device=device,
        )

    def generate(
        self,
        prompt: str,
        *,
        seed: int = 0,
        num_evaluations: int | None = None,
        num_frames: int = 120,
        output_type: str = "np",
        image: Any = None,
        last_image: Any = None,
        step_callback: StepCallback | None = None,
        denoise_cache=None,
    ) -> GenerationResult:
        evaluations = num_evaluations or self.default_evaluations
        generator = torch.Generator(device="cpu").manual_seed(seed)
        call_kwargs = {
            "prompt": prompt,
            "num_frames": num_frames,
            "num_inference_steps": _sigma_grid_points(evaluations),
            "generator": generator,
            "output_type": output_type,
        }
        if image is not None:
            call_kwargs["image"] = image
        if last_image is not None:
            call_kwargs["last_image"] = last_image
        # The whole condition (issue #24 C1): prompt, image slots, frame
        # count (the VDN canvas is fixed, so no height/width knob).
        media = (image, last_image, num_frames)
        pipeline, capture_key = _consult_condition_cache(
            self, prompt, media, call_kwargs
        )
        with (
            _denoise_cache_context(self, denoise_cache, evaluations),
            _denoising_progress(self.pipeline, evaluations, step_callback),
        ):
            state = pipeline(**call_kwargs)
        _capture_condition(self, capture_key, state)
        values = {key: _state_value(state, key) for key in _OUTPUT_KEYS}
        return GenerationResult(**values)


def _offload_vdn(
    pipeline, device, block_stream_blocks_per_group, stream_text_encoder
):
    # The README's own offload recipe: leaf-level streaming for the
    # 62 GB Qwen3-VL encoder, whole-module accelerate hooks for the two
    # decoders, and (optionally) block_level streaming for the
    # transformer. The VDN transformer cannot be leaf-offloaded: its
    # fused kernels read child weights without calling the child.
    from accelerate import cpu_offload_with_hook
    from diffusers.hooks import apply_group_offloading

    if stream_text_encoder:
        apply_group_offloading(
            pipeline.text_encoder,
            onload_device=torch.device(device),
            offload_type="leaf_level",
            use_stream=True,
        )
    else:
        pipeline.text_encoder.to(device)
    _, vae_hook = cpu_offload_with_hook(pipeline.vae, execution_device=device)
    cpu_offload_with_hook(
        pipeline.audio_vae,
        execution_device=device,
        prev_module_hook=vae_hook,
    )
    if block_stream_blocks_per_group:
        apply_group_offloading(
            pipeline.transformer,
            onload_device=torch.device(device),
            offload_type="block_level",
            num_blocks_per_group=block_stream_blocks_per_group,
            use_stream=True,
        )
    else:
        pipeline.transformer.to(device)
