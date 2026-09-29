# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in caches from the issue #24 survey (C1/C3/C5).

Nothing in this package is on by default and nothing here may enter the
bitwise golden path: the parity gates (``test_reference_parity``,
``test_ref2va_parity``, ``test_vdn_parity``) run without any of these
caches. C1 and C3 are exact (whole-condition / content-hash reuse); C5 is
approximate and carries its own quality gate — it never claims
``rms_rel=0``.
"""

from omni_infinity.caches.condition import (
    ConditionCache,
    ConditionEntry,
    build_conditioned_pipeline,
    condition_key,
)
from omni_infinity.caches.denoise import (
    DenoiseCacheConfig,
    DenoiseCacheStats,
    denoise_step_cache,
    enable_cache_dit,
)
from omni_infinity.caches.vision import (
    VisionEmbedCache,
    enable_vision_cache,
)

__all__ = [
    "ConditionCache",
    "ConditionEntry",
    "DenoiseCacheConfig",
    "DenoiseCacheStats",
    "VisionEmbedCache",
    "build_conditioned_pipeline",
    "condition_key",
    "denoise_step_cache",
    "enable_cache_dit",
    "enable_vision_cache",
]
