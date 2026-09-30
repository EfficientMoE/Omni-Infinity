# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Exact whole-condition cache for H3 generation."""

from __future__ import annotations

import hashlib

from omni_infinity.caches._tensor_tree import update_hash_for_value

CACHE_SCHEMA_VERSION = 1


def condition_key(
    namespace: str,
    prompt: str,
    media: tuple = (),
    *,
    height: int,
    width: int,
    num_frames: int,
) -> str:
    """Return the digest for one complete encoded condition."""
    hasher = hashlib.sha256()
    for value in (
        f"omni-condition-v{CACHE_SCHEMA_VERSION}",
        namespace,
        prompt,
        tuple(media),
        height,
        width,
        num_frames,
    ):
        update_hash_for_value(hasher, value)
    return hasher.hexdigest()
