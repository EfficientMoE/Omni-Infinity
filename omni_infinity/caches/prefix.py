# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Exact whole-prefix caching for Qwen3-VL encoder hidden states."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from functools import wraps
from types import MappingProxyType

from omni_infinity.caches._tensor_tree import (
    stable_image_bytes,
    tree_map,
    tree_nbytes,
    update_hash_for_value,
)
from omni_infinity.registry import OptimizationSpec

DEFAULT_MAX_BYTES = 8 * 1024**3
_VISION_INPUT_NAMES = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
)


def _reference_bytes(value) -> bytes:
    image = getattr(value, "image", value)
    return stable_image_bytes(image)


def encoder_prefix_key(
    tokenized_prefix,
    image_refs: tuple = (),
    *,
    namespace: str | None = None,
    extra_inputs=None,
) -> str:
    """Return the versioned digest for one exact encoder presentation."""
    hasher = hashlib.sha256()
    for value in (
        "omni-encoder-v1",
        namespace,
        tokenized_prefix,
        tuple(_reference_bytes(reference) for reference in image_refs),
        extra_inputs,
    ):
        update_hash_for_value(hasher, value)
    return hasher.hexdigest()


class EncoderPrefixCache:
    """Thread-safe, entry- and byte-bounded LRU for encoder outputs.

    The admission policy follows MoE-Infinity's residency-policy shape:
    ordinary entries evict the oldest available resident under pressure,
    while an entry that cannot fit even after eviction is transient and is
    returned to its caller without becoming resident. Encoder outputs need no
    lease or refcount machinery because serving uses one serialized worker.
    """

    def __init__(
        self,
        max_entries: int = 4,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
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
        value_bytes = tree_nbytes(value)
        with self._lock:
            if value_bytes > self.max_bytes:
                return
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= tree_nbytes(previous)
            self._entries[key] = value
            self._bytes += value_bytes
            self._entries.move_to_end(key)
            while (
                len(self._entries) > self.max_entries
                or self._bytes > self.max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= tree_nbytes(evicted)

    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "entries": len(self._entries),
                "bytes": self._bytes,
            }


def _encoder_module(text_encoder):
    module = getattr(text_encoder, "model", None)
    if module is not None:
        return module
    raise AttributeError("text encoder has no model module")


def _call_key(args, kwargs, namespace: str) -> str:
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        input_ids = args[0]
    if input_ids is None:
        raise TypeError("encoder call has no input_ids")
    vision_names = tuple(
        name for name in _VISION_INPUT_NAMES if kwargs.get(name) is not None
    )
    image_refs = tuple(kwargs[name] for name in vision_names)
    extra_inputs = (
        tuple(args[1:]),
        vision_names,
        {
            name: kwargs[name]
            for name in sorted(kwargs)
            if name != "input_ids" and name not in _VISION_INPUT_NAMES
        },
    )
    return encoder_prefix_key(
        input_ids, image_refs, namespace=namespace, extra_inputs=extra_inputs
    )


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


class EncoderCacheController:
    """Own the reversible instance-level encoder ``forward`` wrapper."""

    def __init__(
        self,
        module,
        cache: EncoderPrefixCache,
        original_forward,
        previous_instance_forward,
        had_instance_forward: bool,
        forward_delegate=None,
        guardian_handle=None,
    ) -> None:
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


def enable_encoder_cache(
    text_encoder,
    cache: EncoderPrefixCache | None = None,
    *,
    max_entries: int = 4,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> EncoderCacheController:
    """Cache exact calls to the Qwen3-VL model, namespaced by model identity."""
    module = _encoder_module(text_encoder)
    cache = cache or EncoderPrefixCache(
        max_entries=max_entries, max_bytes=max_bytes
    )
    original_forward = module.forward
    had_instance_forward = "forward" in module.__dict__
    previous_instance_forward = module.__dict__.get("forward")
    module_type = type(module)
    namespace = (
        f"{module_type.__module__}.{module_type.__qualname__}:{id(module)}"
    )
    forward_delegate = [original_forward]

    @wraps(original_forward)
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
    return EncoderCacheController(
        module,
        cache,
        original_forward,
        previous_instance_forward,
        had_instance_forward,
        forward_delegate=forward_delegate,
        guardian_handle=guardian_handle,
    )


OPTIMIZATION = OptimizationSpec(
    name="encoder-cache",
    description="Cache exact Qwen3-VL encoder presentations by content hash.",
    supported_archs=("h3-dense",),
    runner_kwargs_by_arch=MappingProxyType(
        {"h3-dense": MappingProxyType({"encoder_cache": True})}
    ),
)
