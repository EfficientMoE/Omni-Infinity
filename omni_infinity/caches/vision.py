# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""C3 — content-hash vision-embedding cache (issue #24).

Copies vLLM-Omni's multimodal *encoder cache* reuse rule: the key is a
content hash of the vision tower's inputs and the value is the tower's
output — lookup is independent of any block/prefix alignment. When a C1
condition-cache miss still shares its images with an earlier request
(vLLM's "same image with a different prompt" case), the Qwen3-VL vision
tower is skipped while the text encode reruns.

Granularity note: entries are keyed per tower *call* (the H3 encode
passes each request's image patches in one call), not per image inside a
call. Splitting a multi-image call into per-image entries is the C2
encoder-prefix-cache follow-up; this module stays exact rather than
guessing patch boundaries.

The wrapper replaces the tower's instance ``forward`` (the same
attachment point diffusers' offloading hooks use), so with a streamed
encoder a hit also skips the tower's weight onloads.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import torch

from omni_infinity.caches._tensor_tree import (
    tree_map,
    tree_nbytes,
    update_hash_for_value,
)


class VisionEmbedCache:
    """LRU of content-hash -> vision-tower output (host-resident)."""

    def __init__(self, *, max_entries: int = 4):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key]
            self.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "entries": len(self._entries),
                "bytes": sum(tree_nbytes(v) for v in self._entries.values()),
            }


def _call_key(args, kwargs) -> str:
    hasher = hashlib.sha256()
    update_hash_for_value(hasher, "omni-vision-v1")
    update_hash_for_value(hasher, tuple(args))
    update_hash_for_value(
        hasher, {name: kwargs[name] for name in sorted(kwargs)}
    )
    return hasher.hexdigest()


def _first_tensor_device(args, kwargs):
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value.device
    return None


def _visual_module(text_encoder):
    """The Qwen3-VL vision tower: ``visual`` or ``model.visual``."""
    visual = getattr(text_encoder, "visual", None)
    if visual is None:
        model = getattr(text_encoder, "model", None)
        visual = getattr(model, "visual", None)
    if visual is None:
        raise AttributeError(
            "no vision tower ('visual') found on the text encoder"
        )
    return visual


class VisionCacheController:
    """Owns the wrapped ``forward``; ``close()`` restores the original."""

    def __init__(self, module, cache: VisionEmbedCache):
        self.module = module
        self.cache = cache
        self._had_instance_forward = "forward" in module.__dict__
        self._previous_forward = module.__dict__.get("forward")
        original = module.forward

        def cached_forward(*args, **kwargs):
            key = _call_key(args, kwargs)
            hit = cache.get(key)
            if hit is not None:
                device = _first_tensor_device(args, kwargs)
                if device is None:
                    return hit
                return tree_map(lambda t: t.to(device), hit)
            output = original(*args, **kwargs)
            cache.put(key, tree_map(lambda t: t.detach().to("cpu"), output))
            return output

        module.forward = cached_forward
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        if self._had_instance_forward:
            self.module.forward = self._previous_forward
        else:
            del self.module.__dict__["forward"]
        self._closed = True


def enable_vision_cache(
    text_encoder,
    cache: VisionEmbedCache | None = None,
    *,
    max_entries: int = 4,
) -> VisionCacheController:
    """Wrap the encoder's vision tower with a content-hash cache."""
    module = _visual_module(text_encoder)
    if cache is None:
        cache = VisionEmbedCache(max_entries=max_entries)
    return VisionCacheController(module, cache)
