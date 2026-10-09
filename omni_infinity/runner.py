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
import logging
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from functools import wraps
from types import MethodType
from typing import Any

import torch

from omni_infinity.caches.attach import attach_caches, bind_generation

StepCallback = Callable[[int, int], None]
logger = logging.getLogger(__name__)


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


def _validate_compile_blocks(
    *, compile_blocks: bool, block_stream_blocks_per_group: int
) -> None:
    if compile_blocks and block_stream_blocks_per_group > 0:
        raise ValueError(
            "compile-blocks is incompatible with block streaming: "
            "group-offload hooks can leave compiled regions with stale "
            "weight pointers"
        )


def _validate_cuda_graph(
    *, cuda_graph: bool, block_stream_blocks_per_group: int
) -> None:
    if cuda_graph and block_stream_blocks_per_group > 0:
        raise ValueError(
            "cuda-graph is incompatible with block streaming: captured "
            "graphs require resident, pointer-stable transformer weights"
        )


def _compile_repeated_transformer_blocks(
    pipeline, component_name="transformer"
) -> None:
    transformer = _transformer_component(pipeline, component_name)
    transformer.compile_repeated_blocks(fullgraph=True, dynamic=True)


def _available_host_memory_bytes() -> int:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("could not read MemAvailable from /proc/meminfo")


def _pin_adaln_for_cuda_graph(
    transformer,
    *,
    available_bytes: int | None = None,
    pin: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> int:
    from omni_infinity.adaln import HostResidentAdaLN

    entries = []
    seen = set()
    for module in transformer.modules():
        if (
            isinstance(module, HostResidentAdaLN)
            and id(module._entry) not in seen
        ):
            seen.add(id(module._entry))
            entries.append(module._entry)
    required = sum(entry.host_nbytes for entry in entries)
    if required == 0:
        logger.info("CUDA graph enabled without an AdaLN host cache")
        return 0
    available = (
        _available_host_memory_bytes()
        if available_bytes is None
        else available_bytes
    )
    if available * 5 < required * 6:
        raise MemoryError(
            "pinning the AdaLN host cache for cuda-graph requires its full "
            "footprint plus 20% headroom: "
            f"required={required / 1024**3:.2f} GiB, "
            f"available={available / 1024**3:.2f} GiB"
        )
    pinned = sum(entry.pin_memory(pin) for entry in entries)
    logger.info("pinned %.2f GiB of AdaLN graph-copy sources", pinned / 1024**3)
    return pinned


def _wrap_transformer_with_cuda_graph(
    pipeline, manager, component_name="transformer"
) -> None:
    transformer = _transformer_component(pipeline, component_name)
    original_forward = transformer.forward

    @wraps(type(transformer).forward)
    def graph_forward(_module, *args, **kwargs):
        output = manager.try_execute(
            manager.current_bucket, original_forward, *args, **kwargs
        )
        if output is None:
            return original_forward(*args, **kwargs)
        return output

    # Installed during model loading. C5 installs its generation-local wrapper
    # later and captures this method as original_forward, so cache hits return
    # outside the graph manager and are accounted by the runner's skip hook.
    transformer.forward = MethodType(graph_forward, transformer)


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


def _effective_video_frames(requested_frames: int) -> int:
    effective = requested_frames
    while effective % 17 != 5:
        effective += 1
    return effective


class ReferenceRunner:
    """Full-resident H3-Base FL2VA text-to-audio/video reference path."""

    def __init__(
        self,
        pipeline,
        overlap_controller=None,
        transformer_component: str = "transformer",
        cuda_graph_manager=None,
        pinned_adaln_bytes: int = 0,
        cuda_graph_invalidate_between_generations: bool = False,
    ):
        self.pipeline = pipeline
        self.overlap_controller = overlap_controller
        self.transformer_component = transformer_component
        self.cuda_graph_manager = cuda_graph_manager
        self.pinned_adaln_bytes = pinned_adaln_bytes
        self.cuda_graph_invalidate_between_generations = (
            cuda_graph_invalidate_between_generations
        )
        self._cuda_graph_generation_started = False

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
        fp8_protect_blocks: str | None = None,
        fp8_scale: str = "block",
        transformer_fp4: bool = False,
        fp4_scale: str | None = None,
        offload_memory_margin: str | None = None,
        block_stream_blocks_per_group: int = 0,
        block_stream_to_disk: str | None = None,
        compile_blocks: bool = False,
        cuda_graph: bool = False,
        stream_text_encoder: bool = False,
        step_overlap: bool = False,
        condition_cache: bool = False,
        condition_cache_dir: str | None = None,
        vision_cache: bool = False,
    ) -> "ReferenceRunner":
        _validate_compile_blocks(
            compile_blocks=compile_blocks,
            block_stream_blocks_per_group=block_stream_blocks_per_group,
        )
        _validate_cuda_graph(
            cuda_graph=cuda_graph,
            block_stream_blocks_per_group=block_stream_blocks_per_group,
        )
        if step_overlap and not block_stream_blocks_per_group:
            raise ValueError("step_overlap requires bf16 block streaming")
        if transformer_fp4:
            if fp4_scale is None:
                raise ValueError(
                    "transformer_fp4 requires an explicit fp4 scale mode "
                    "(--fp4-scale mxfp4); fp4 is opt-in with no default"
                )
            if fp4_scale != "mxfp4":
                raise ValueError(
                    f"unsupported fp4_scale {fp4_scale!r}; only 'mxfp4' "
                    "is supported"
                )
            if transformer_fp8:
                raise ValueError(
                    "transformer_fp8 and transformer_fp4 are mutually "
                    "exclusive"
                )
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
            from omni_infinity.fp8 import parse_protect_blocks
            from omni_infinity.store import (
                StoreComponentSource,
                load_diffusers_component,
                load_transformer_with_adaln_cache,
            )

            source = StoreComponentSource(store_dir)
            for name in substituted:
                if name == transformer_component and (
                    adaln_host_cache or transformer_fp8 or transformer_fp4
                ):
                    built[name] = load_transformer_with_adaln_cache(
                        _h3_component_class(name),
                        checkpoint,
                        source,
                        torch_dtype,
                        component=name,
                        fp8=transformer_fp8,
                        fp8_skip_last_blocks=fp8_skip_last_blocks,
                        fp8_protect_blocks=(
                            parse_protect_blocks(fp8_protect_blocks)
                            if fp8_protect_blocks is not None
                            else None
                        ),
                        fp8_mode=fp8_scale,
                        fp4=transformer_fp4,
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
        pinned_adaln_bytes = 0
        if cuda_graph:
            pinned_adaln_bytes = _pin_adaln_for_cuda_graph(
                _transformer_component(pipeline, transformer_component)
            )
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
        if compile_blocks:
            # The AdaLN host-cache modules are installed before regional
            # compilation. C5 wraps the whole transformer forward later, so
            # compiled repeated blocks sit inside its scope and a cache hit
            # bypasses them entirely.
            _compile_repeated_transformer_blocks(
                pipeline, transformer_component
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
        cuda_graph_manager = None
        if cuda_graph:
            from omni_infinity.cuda_graph import CudaGraphManager

            cuda_graph_manager = CudaGraphManager(device=device, max_buckets=2)
            _wrap_transformer_with_cuda_graph(
                pipeline, cuda_graph_manager, transformer_component
            )
        runner = cls(
            pipeline,
            overlap_controller=overlap_controller,
            transformer_component=transformer_component,
            cuda_graph_manager=cuda_graph_manager,
            pinned_adaln_bytes=pinned_adaln_bytes,
            cuda_graph_invalidate_between_generations=offload and cuda_graph,
        )
        attach_caches(
            runner,
            condition_cache=condition_cache,
            condition_cache_dir=condition_cache_dir,
            vision_cache=vision_cache,
            cache_namespace=f"ReferenceRunner:{checkpoint}",
        )
        return runner

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
        denoise_cache=None,
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
        binding = bind_generation(
            self,
            prompt=prompt,
            media=(image, last_image, *(references or ())),
            height=height,
            width=width,
            num_frames=num_frames,
            call_kwargs=call_kwargs,
            denoise_cache=denoise_cache,
            total_steps=num_inference_steps,
            transformer=None,
        )
        if denoise_cache is not None:
            binding.transformer = _transformer_component(
                binding.pipeline, self.transformer_component
            )
        graph_context = nullcontext()
        if self.cuda_graph_manager is not None:
            from omni_infinity.cuda_graph import GraphKey

            if (
                self.cuda_graph_invalidate_between_generations
                and self._cuda_graph_generation_started
            ):
                self.cuda_graph_manager.invalidate(
                    "component offload generation boundary"
                )
            self._cuda_graph_generation_started = True
            graph_context = self.cuda_graph_manager.bucket(
                GraphKey(
                    height=height,
                    width=width,
                    frames=_effective_video_frames(num_frames),
                )
            )
        with graph_context:
            with binding.denoise() as denoise_stats:
                with _denoising_progress(
                    binding.pipeline,
                    num_inference_steps,
                    step_callback,
                    self.transformer_component,
                ):
                    state = binding.pipeline(**binding.call_kwargs)
        if self.cuda_graph_manager is not None and denoise_stats is not None:
            self.cuda_graph_manager.record_cache_skip(denoise_stats.skipped)
        binding.observe(state)
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
