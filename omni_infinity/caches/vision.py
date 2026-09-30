# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Whole-call caching for vision-tower embeddings."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from types import MappingProxyType

from omni_infinity.caches._tensor_tree import (
    tree_map,
    tree_nbytes,
    update_hash_for_value,
)
from omni_infinity.registry import OptimizationSpec


class VisionEmbedCache:
    """Thread-safe, entry-bounded LRU for vision-tower outputs."""

    def __init__(self, max_entries: int = 4):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._entries:
                self._misses += 1
                return None
            self._hits += 1
            self._entries.move_to_end(key)
            return self._entries[key]

    def put(self, key, value) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "entries": len(self._entries),
                "bytes": sum(
                    tree_nbytes(value) for value in self._entries.values()
                ),
            }


def _visual_module(text_encoder):
    visual = getattr(text_encoder, "visual", None)
    if visual is not None:
        return visual
    model = getattr(text_encoder, "model", None)
    visual = getattr(model, "visual", None)
    if visual is not None:
        return visual
    raise AttributeError("text encoder has no visual module")


def _call_key(args, kwargs, namespace: str | None = None) -> bytes:
    hasher = hashlib.sha256()
    update_hash_for_value(hasher, "omni-vision-v1")
    update_hash_for_value(hasher, namespace)
    update_hash_for_value(hasher, tuple(args))
    update_hash_for_value(
        hasher, {name: kwargs[name] for name in sorted(kwargs)}
    )
    return hasher.digest()


def _first_tensor_device(args, kwargs):
    device = None

    def remember(tensor):
        nonlocal device
        if device is None:
            device = tensor.device
        return tensor

    tree_map(remember, args)
    for key in sorted(kwargs):
        tree_map(remember, kwargs[key])
    return device


def _map_output_tensors(fn, output):
    if isinstance(output, dict) and type(output) is not dict:
        mapped = {key: tree_map(fn, value) for key, value in output.items()}
        try:
            return type(output)(**mapped)
        except TypeError:
            return type(output)(mapped)
    return tree_map(fn, output)


class VisionCacheController:
    """Own the reversible instance-level vision ``forward`` wrapper."""

    def __init__(
        self,
        module,
        cache: VisionEmbedCache,
        original_forward,
        previous_instance_forward,
        had_instance_forward: bool,
        forward_delegate=None,
        guardian_handle=None,
    ):
        self.module = module
        self.cache = cache
        self.original_forward = original_forward
        self.previous_instance_forward = previous_instance_forward
        self.had_instance_forward = had_instance_forward
        self.forward_delegate = forward_delegate
        self.guardian_handle = guardian_handle
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        if self.guardian_handle is not None:
            self.guardian_handle.remove()
            self.module.forward = self.forward_delegate[0]
        elif self.had_instance_forward:
            self.module.forward = self.previous_instance_forward
        else:
            del self.module.forward
        self._closed = True


def enable_vision_cache(
    text_encoder,
    cache: VisionEmbedCache | None = None,
    *,
    max_entries: int = 4,
) -> VisionCacheController:
    """Cache complete calls to the encoder's vision tower,
    namespaced by tower identity."""
    module = _visual_module(text_encoder)
    cache = cache or VisionEmbedCache(max_entries=max_entries)
    original_forward = module.forward
    had_instance_forward = "forward" in module.__dict__
    previous_instance_forward = module.__dict__.get("forward")

    module_type = type(module)
    namespace = (
        f"{module_type.__module__}.{module_type.__qualname__}:{id(module)}"
    )

    forward_delegate = [original_forward]

    def cached_forward(*args, **kwargs):
        key = _call_key(args, kwargs, namespace)
        cached = cache.get(key)
        if cached is not None:
            device = _first_tensor_device(args, kwargs)
            if device is None:
                return cached
            return _map_output_tensors(lambda tensor: tensor.to(device), cached)

        output = forward_delegate[0](*args, **kwargs)
        cache.put(
            key,
            _map_output_tensors(
                lambda tensor: tensor.detach().to("cpu"), output
            ),
        )
        return output

    module.forward = cached_forward
    guardian_handle = None
    if hasattr(module, "_diffusers_hook"):

        def preserve_wrapper(current_module, _args, output):
            if current_module.forward is not cached_forward:
                forward_delegate[0] = current_module.forward
                current_module.forward = cached_forward
            return output

        guardian_handle = module.register_forward_hook(preserve_wrapper)
    return VisionCacheController(
        module,
        cache,
        original_forward,
        previous_instance_forward,
        had_instance_forward,
        forward_delegate=forward_delegate,
        guardian_handle=guardian_handle,
    )


OPTIMIZATION = OptimizationSpec(
    name="vision-cache",
    description="Cache complete vision-tower calls by content hash.",
    supported_archs=("h3-dense", "vdn-hybrid"),
    runner_kwargs_by_arch=MappingProxyType(
        {
            "h3-dense": MappingProxyType({"vision_cache": True}),
            "vdn-hybrid": MappingProxyType({"vision_cache": True}),
        }
    ),
)
