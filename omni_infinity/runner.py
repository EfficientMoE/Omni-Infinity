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
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch

StepCallback = Callable[[int, int], None]


def _transformer_component(pipeline, component_name="transformer"):
    for accessor in (
        lambda: pipeline.get_component(component_name),
        lambda: getattr(pipeline, component_name),
    ):
        try:
            component = accessor()
        except Exception:
            continue
        if component is not None:
            return component
    raise AttributeError("could not locate the transformer on the pipeline")


@contextmanager
def _denoising_progress(
    pipeline,
    total_steps: int,
    callback: StepCallback | None,
    component_name: str = "transformer",
):
    if callback is None:
        yield
        return
    completed = 0

    def after_transformer_forward(module, args, output):
        nonlocal completed
        completed += 1
        callback(min(completed, total_steps), total_steps)

    transformer = _transformer_component(pipeline, component_name)
    handle = transformer.register_forward_hook(after_transformer_forward)
    try:
        yield
    finally:
        handle.remove()


def _enable_block_streaming(
    pipeline,
    device,
    blocks_per_group,
    to_disk,
    component_name="transformer",
):
    # Native diffusers block-level group offload: streams one block's attn/ff
    # weights at a time with CUDA-stream prefetch (use_stream forces
    # num_blocks_per_group=1). The param-less HostResidentAdaLN is skipped, so
    # the 26 GB of AdaLN branches are never streamed. bf16 device-only moves
    # keep the latents bitwise-identical to the full-resident reference.
    transformer = _transformer_component(pipeline, component_name)
    transformer.enable_group_offload(
        onload_device=torch.device(device),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=max(blocks_per_group, 1),
        use_stream=True,
        record_stream=False,
        low_cpu_mem_usage=False,
        offload_to_disk_path=to_disk,
    )


def _APPLY_GROUP_OFFLOADING(module, **kwargs):
    from diffusers.hooks import apply_group_offloading

    apply_group_offloading(module, **kwargs)


def _encoder_layer_host(pipeline):
    # Submodule whose direct child is the longest ModuleList (the decoder
    # layers). block_level group offload must target it: the layers are nested
    # (Qwen3-VL: model.language_model.layers), not a direct child of the top
    # encoder. Dynamic lookup avoids hardcoding a transformers-version path.
    encoder = pipeline.text_encoder
    best_module, best_len = None, -1
    for _name, module in encoder.named_modules():
        for child in module.children():
            if isinstance(child, torch.nn.ModuleList) and len(child) > best_len:
                best_module, best_len = module, len(child)
    if best_module is None:
        raise AttributeError("no decoder-layer ModuleList on the text encoder")
    return best_module


def _stream_text_encoder(pipeline, device):
    # bf16 leaf-level streaming of the Qwen3-VL decoder subtree: device-only
    # moves, so prompt embeds (and thus latents) stay parity-identical. Ref2VA
    # also executes the visual patch embedder, which may be nested under the
    # encoder model rather than exposed directly on the text encoder.
    # leaf_level (not block_level) hooks EVERY leaf -- including embed_tokens --
    # so each self-onloads when it runs; block_level leaves the embedding on the
    # offload device and the first token lookup hits a device mismatch.
    kwargs = {
        "onload_device": torch.device(device),
        "offload_device": torch.device("cpu"),
        "offload_type": "leaf_level",
        "use_stream": True,
        "record_stream": False,
        "low_cpu_mem_usage": False,
    }
    _APPLY_GROUP_OFFLOADING(_encoder_layer_host(pipeline), **kwargs)
    visual = getattr(pipeline.text_encoder, "visual", None)
    if visual is None:
        model = getattr(pipeline.text_encoder, "model", None)
        visual = getattr(model, "visual", None)
    if visual is not None:
        _APPLY_GROUP_OFFLOADING(visual, **kwargs)


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

REF2VA_COMPONENTS = (
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
    "transformer_ref",
)

TRANSFORMER_COMPONENTS = {"fl2va": "transformer", "ref2va": "transformer_ref"}

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


def _h3_component_class(name: str):
    import diffusers

    classes = {
        "vae": "AutoencoderKLMiniMaxH3",
        "audio_vae": "AutoencoderKLMiniMaxH3Audio",
        "transformer": "MiniMaxH3Transformer3DModel",
        "transformer_ref": "MiniMaxH3Transformer3DModel",
    }
    try:
        return getattr(diffusers, classes[name])
    except KeyError:
        raise ValueError(
            f"no store-loadable class known for component {name!r}"
        ) from None


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

    def __init__(
        self,
        pipeline,
        overlap_controller=None,
        transformer_component: str = "transformer",
    ):
        self.pipeline = pipeline
        self.overlap_controller = overlap_controller
        self.transformer_component = transformer_component

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "MiniMaxAI/MiniMax-H3",
        *,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        offload: bool = False,
        workflow: str = "fl2va",
        components: tuple[str, ...] | None = None,
        store_dir: str | None = None,
        store_components: tuple[str, ...] = ("vae", "audio_vae"),
        adaln_host_cache: bool = False,
        transformer_fp8: bool = False,
        fp8_skip_last_blocks: int = 0,
        fp8_scale: str = "block",
        offload_memory_margin: str | None = None,
        block_stream_blocks_per_group: int = 0,
        block_stream_to_disk: str | None = None,
        stream_text_encoder: bool = False,
        step_overlap: bool = False,
    ) -> "ReferenceRunner":
        if step_overlap and not block_stream_blocks_per_group:
            raise ValueError("step_overlap requires bf16 block streaming")
        try:
            from diffusers import MiniMaxH3ModularPipeline
        except ImportError as exc:
            raise ImportError(
                "diffusers >= 0.40 with MiniMaxH3ModularPipeline is required "
                "for the reference runner"
            ) from exc

        transformer_component = TRANSFORMER_COMPONENTS[workflow]
        if components is None:
            components = (
                REF2VA_COMPONENTS if workflow == "ref2va" else FL2VA_COMPONENTS
            )

        components_manager = None
        if offload:
            # The full FL2VA component set (~144 GB bf16) exceeds a single
            # GPU, so the reference path can run components sequentially with
            # ComponentsManager auto CPU offload instead of .to(device). Auto
            # offload is enabled below, after update_components, so its hooks
            # bind to the store-built (and FP8) transformer rather than the
            # from_pretrained placeholder.
            from diffusers.modular_pipelines import ComponentsManager

            components_manager = ComponentsManager()

        substituted = tuple(store_components) if store_dir else ()
        pipeline = MiniMaxH3ModularPipeline.from_pretrained(
            checkpoint,
            workflow=workflow,
            components_manager=components_manager,
        )
        # The Qwen3-VL processor's component spec resolves by repo id, which
        # fails under HF offline mode (a sub-tokenizer config lookup raises
        # rather than skipping a locally-absent file). Load it from the
        # checkpoint path directly and inject it; harmless online.
        preloaded = {}
        loadable = [c for c in components if c not in substituted]
        if "processor" in loadable:
            from transformers import AutoProcessor

            loadable.remove("processor")
            preloaded["processor"] = AutoProcessor.from_pretrained(
                checkpoint, subfolder="processor"
            )
        pipeline.load_components(names=loadable, torch_dtype=torch_dtype)
        built = dict(preloaded)
        if store_dir:
            from omni_infinity.store import (
                StoreComponentSource,
                load_diffusers_component,
                load_transformer_with_adaln_cache,
            )

            source = StoreComponentSource(store_dir)
            for name in substituted:
                if name == transformer_component and (
                    adaln_host_cache or transformer_fp8
                ):
                    built[name] = load_transformer_with_adaln_cache(
                        _h3_component_class(name),
                        checkpoint,
                        source,
                        torch_dtype,
                        component=name,
                        fp8=transformer_fp8,
                        fp8_skip_last_blocks=fp8_skip_last_blocks,
                        fp8_mode=fp8_scale,
                    )
                else:
                    built[name] = load_diffusers_component(
                        _h3_component_class(name),
                        checkpoint,
                        name,
                        source,
                        torch_dtype,
                    )
        if built:
            pipeline.update_components(**built)
        overlap_controller = None
        if block_stream_blocks_per_group:
            if not offload:
                raise ValueError(
                    "block streaming requires offload=True (the transformer "
                    "manages its own device placement; other components need "
                    "the ComponentsManager)"
                )
            _enable_block_streaming(
                pipeline,
                device,
                block_stream_blocks_per_group,
                block_stream_to_disk,
                transformer_component,
            )
        if step_overlap:
            from omni_infinity.step_overlap import enable_step_overlap

            overlap_controller = enable_step_overlap(
                _transformer_component(pipeline, transformer_component)
            )
        if stream_text_encoder:
            if not offload:
                raise ValueError("stream_text_encoder requires offload=True")
            _stream_text_encoder(pipeline, device)
        if offload:
            # memory_reserve_margin = (device total - target budget) keeps the
            # manager evicting until only ~budget stays resident, emulating a
            # small-VRAM envelope on a large dev-box GPU.
            margin = offload_memory_margin or "3GB"
            components_manager.enable_auto_cpu_offload(
                device=device, memory_reserve_margin=margin
            )
        else:
            pipeline.to(device)
        return cls(
            pipeline,
            overlap_controller=overlap_controller,
            transformer_component=transformer_component,
        )

    def generate(
        self,
        prompt: str,
        *,
        seed: int = 0,
        num_inference_steps: int = 8,
        resolution: str = "256p",
        num_frames: int = 8,
        output_type: str = "np",
        references: list[Any] | None = None,
        image: Any = None,
        last_image: Any = None,
        step_callback: StepCallback | None = None,
    ) -> GenerationResult:
        height, width = resolve_resolution(resolution)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        call_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "output_type": output_type,
        }
        if image is not None:
            call_kwargs["image"] = image
        if last_image is not None:
            call_kwargs["last_image"] = last_image
        if references is not None:
            call_kwargs["references"] = references
        with _denoising_progress(
            self.pipeline,
            num_inference_steps,
            step_callback,
            self.transformer_component,
        ):
            state = self.pipeline(**call_kwargs)
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
