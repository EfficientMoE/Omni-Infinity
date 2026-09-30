# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed attachment points for optional inference caches."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from contextlib import contextmanager


def _noop(_state) -> None:
    return None


def _module_is_missing(exc: ModuleNotFoundError, module_name: str) -> bool:
    return exc.name == module_name or module_name.startswith(f"{exc.name}.")


@dataclasses.dataclass
class GenerationBinding:
    pipeline: object
    call_kwargs: dict
    observe: Callable[[object], None]
    transformer: object
    total_steps: int
    denoise_cache: object | None = None
    denoise_step_cache: Callable | None = None

    @contextmanager
    def denoise(self):
        if self.denoise_cache is None:
            yield None
            return
        with self.denoise_step_cache(
            self.transformer, self.denoise_cache, self.total_steps
        ) as stats:
            yield stats


def attach_caches(
    runner,
    *,
    condition_cache: bool,
    condition_cache_dir: str | None,
    vision_cache: bool,
    cache_namespace: str,
) -> None:
    """Attach requested optional caches to ``runner`` or fail loudly."""
    runner.condition_cache = None
    if condition_cache:
        try:
            from omni_infinity.caches.condition import ConditionCache
        except ModuleNotFoundError as exc:
            if not _module_is_missing(exc, "omni_infinity.caches.condition"):
                raise
            raise RuntimeError("condition-cache is not installed") from None
        runner.condition_cache = ConditionCache(cache_dir=condition_cache_dir)

    runner.vision_cache_controller = None
    if vision_cache:
        try:
            from omni_infinity.caches.vision import enable_vision_cache
        except ModuleNotFoundError as exc:
            if not _module_is_missing(exc, "omni_infinity.caches.vision"):
                raise
            raise RuntimeError("vision-cache is not installed") from None
        components = getattr(runner.pipeline, "components", {})
        text_encoder = components.get("text_encoder")
        if text_encoder is None:
            text_encoder = runner.pipeline.text_encoder
        runner.vision_cache_controller = enable_vision_cache(text_encoder)

    runner.cache_namespace = cache_namespace


def bind_generation(
    runner,
    *,
    prompt: str,
    media: tuple,
    height: int,
    width: int,
    num_frames: int,
    call_kwargs: dict,
    denoise_cache,
    total_steps: int,
    transformer,
) -> GenerationBinding:
    """Bind enabled caches to one generation call."""
    pipeline = runner.pipeline
    observe = _noop
    bound_kwargs = call_kwargs
    if getattr(runner, "condition_cache", None) is not None:
        from omni_infinity.caches.condition import prepare

        replay = prepare(
            runner.pipeline,
            runner.condition_cache,
            namespace=runner.cache_namespace,
            prompt=prompt,
            media=media,
            height=height,
            width=width,
            num_frames=num_frames,
            call_kwargs=call_kwargs,
        )
        pipeline = replay.pipeline or runner.pipeline
        bound_kwargs = replay.call_kwargs()
        observe = replay.observe

    denoise_step_cache = None
    if denoise_cache is not None:
        try:
            from omni_infinity.caches.denoise import denoise_step_cache
        except ModuleNotFoundError as exc:
            if not _module_is_missing(exc, "omni_infinity.caches.denoise"):
                raise
            raise RuntimeError("denoise-step cache is not installed") from None

    return GenerationBinding(
        pipeline=pipeline,
        call_kwargs=bound_kwargs,
        observe=observe,
        transformer=transformer,
        total_steps=total_steps,
        denoise_cache=denoise_cache,
        denoise_step_cache=denoise_step_cache,
    )


def denoise_config_from_args(args):
    """Build the opt-in denoise cache configuration from CLI arguments."""
    coefficients = getattr(args, "denoise_cache_coefficients", None)
    threshold = getattr(args, "denoise_cache_threshold", None)
    if coefficients is None and threshold is None:
        return None
    if coefficients is None or threshold is None:
        raise ValueError(
            "both denoise cache coefficients and threshold are required"
        )
    try:
        from omni_infinity.caches.denoise import DenoiseCacheConfig
    except ModuleNotFoundError as exc:
        if not _module_is_missing(exc, "omni_infinity.caches.denoise"):
            raise
        raise RuntimeError("denoise-step cache is not installed") from None
    return DenoiseCacheConfig(
        coefficients=tuple(float(value) for value in coefficients.split(",")),
        threshold=threshold,
    )
