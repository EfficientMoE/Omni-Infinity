# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Exact whole-condition cache for H3 generation."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch

from omni_infinity.caches._tensor_tree import (
    tree_map,
    tree_nbytes,
    update_hash_for_value,
)

try:
    from diffusers.modular_pipelines import SequentialPipelineBlocks
except ImportError:
    SequentialPipelineBlocks = None

CACHE_SCHEMA_VERSION = 1
DEFAULT_CAPTURE = (
    "prompt_embeds",
    "text_token_tags",
    "condition_latents",
    "audio_condition_latents",
)
ENCODE_BLOCKS = ("text_encoder", "vae_encoder")
_CONDITION_ARGUMENTS = frozenset(
    (*DEFAULT_CAPTURE, "prompt", "image", "last_image", "references")
)


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


@dataclass(frozen=True)
class ConditionEntry:
    """Host-resident values captured from one encoded condition."""

    values: dict

    def to(self, device) -> dict:
        """Copy the captured tensor tree to ``device``."""
        return tree_map(lambda tensor: tensor.to(device), self.values)


class ConditionCache:
    """Thread-safe host LRU with an optional persistent disk tier."""

    def __init__(
        self,
        max_entries: int = 8,
        max_bytes: int | None = None,
        cache_dir=None,
        capture: tuple[str, ...] = DEFAULT_CAPTURE,
        required: tuple[str, ...] = ("prompt_embeds",),
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least one")
        if not set(required).issubset(capture):
            raise ValueError("required names must be included in capture")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.capture = tuple(capture)
        self.required = tuple(required)
        self._entries: OrderedDict[str, ConditionEntry] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pt"

    @staticmethod
    def _host_entry(values: dict) -> ConditionEntry:
        host_values = tree_map(
            lambda tensor: tensor.detach().to("cpu").clone(), values
        )
        return ConditionEntry(host_values)

    def _remember(self, key: str, entry: ConditionEntry) -> None:
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= tree_nbytes(previous.values)
        self._entries[key] = entry
        self._bytes += tree_nbytes(entry.values)
        while len(self._entries) > self.max_entries or (
            self.max_bytes is not None and self._bytes > self.max_bytes
        ):
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= tree_nbytes(evicted.values)

    def _write_disk(self, key: str, entry: ConditionEntry) -> None:
        if self.cache_dir is None:
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.cache_dir,
                prefix=f".{key}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                torch.save(entry.values, handle)
            os.replace(temporary, self._path(key))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def get(self, key: str) -> ConditionEntry | None:
        """Return ``key`` from memory or disk and update hit statistics."""
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._entries[key] = entry
                self._hits += 1
                return entry

            if self.cache_dir is not None:
                try:
                    values = torch.load(self._path(key), weights_only=True)
                    if not isinstance(values, dict):
                        raise TypeError("condition entry must be a mapping")
                    if not all(name in values for name in self.required):
                        raise ValueError(
                            "condition entry is missing required values"
                        )
                    entry = self._host_entry(values)
                except Exception:
                    entry = None
                if entry is not None:
                    self._remember(key, entry)
                    self._hits += 1
                    return entry

            self._misses += 1
            return None

    def put(self, key: str, values) -> ConditionEntry | None:
        """Capture configured non-``None`` values under ``key``."""
        selected = {
            name: values[name]
            for name in self.capture
            if name in values and values[name] is not None
        }
        if not all(name in selected for name in self.required):
            return None
        with self._lock:
            entry = self._host_entry(selected)
            self._remember(key, entry)
            self._write_disk(key, entry)
            return entry

    def stats(self) -> dict:
        """Return a snapshot of cache counters and host residency."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "entries": len(self._entries),
                "bytes": self._bytes,
            }


def build_conditioned_pipeline(pipeline, skip_blocks=ENCODE_BLOCKS):
    """Return a pipeline view without condition encoders, or ``None``."""
    try:
        blocks = pipeline.blocks
        sub_blocks = blocks.sub_blocks
        if SequentialPipelineBlocks is None:
            return None
        retained = {
            name: block
            for name, block in sub_blocks.items()
            if name not in skip_blocks
        }
        reduced_blocks = SequentialPipelineBlocks.from_blocks_dict(retained)
        reduced = reduced_blocks.init_pipeline()
        shared = {
            name: component
            for name, component in pipeline.components.items()
            if component is not None
        }
        reduced.update_components(**shared)
        return reduced
    except Exception:
        return None


@dataclass(frozen=True)
class ConditionReplay:
    """One fail-open condition-cache decision for a generation call."""

    hit: bool
    pipeline: object | None
    _kwargs: dict
    _observer: Callable[[object], None]

    def call_kwargs(self) -> dict:
        """Return kwargs for the selected pipeline."""
        return self._kwargs

    def observe(self, state) -> None:
        """Capture a miss result, or do nothing for a hit/fail-open call."""
        self._observer(state)


def _noop(_state) -> None:
    return None


def _miss(call_kwargs: dict, observer: Callable = _noop) -> ConditionReplay:
    return ConditionReplay(False, None, call_kwargs, observer)


def _input_names(pipeline) -> set[str]:
    inputs = getattr(getattr(pipeline, "blocks", None), "inputs", ())
    return {
        spec.name
        for spec in inputs
        if isinstance(getattr(spec, "name", None), str)
    }


def prepare(
    pipeline,
    cache: ConditionCache,
    *,
    namespace: str,
    prompt: str,
    media: tuple,
    height: int,
    width: int,
    num_frames: int,
    call_kwargs: dict,
) -> ConditionReplay:
    """Prepare an exact replay, failing open to normal encoding."""
    try:
        key = condition_key(
            namespace,
            prompt,
            media,
            height=height,
            width=width,
            num_frames=num_frames,
        )
    except TypeError:
        return _miss(call_kwargs)

    entry = cache.get(key)
    if entry is None:

        def observe(state) -> None:
            values = {}
            for name in cache.capture:
                if isinstance(state, Mapping):
                    value = state.get(name)
                else:
                    value = getattr(state, name, None)
                values[name] = value
            cache.put(key, values)

        return _miss(call_kwargs, observe)

    reduced = build_conditioned_pipeline(pipeline)
    if reduced is None:
        return _miss(call_kwargs)

    device = getattr(pipeline, "_execution_device", None) or "cpu"
    declared_inputs = _input_names(reduced)
    cached = {
        name: value
        for name, value in entry.to(device).items()
        if name in declared_inputs
    }
    replay_kwargs = {
        name: value
        for name, value in call_kwargs.items()
        if name not in _CONDITION_ARGUMENTS
    }
    replay_kwargs.update(cached)
    return ConditionReplay(True, reduced, replay_kwargs, _noop)


def _optimization_spec():
    from omni_infinity import registry

    enabled = MappingProxyType({"condition_cache": True})
    return registry.OptimizationSpec(
        name="condition-cache",
        description="Replay an exact whole condition without re-encoding.",
        supported_archs=("h3-dense", "vdn-hybrid"),
        runner_kwargs_by_arch=MappingProxyType(
            {"h3-dense": enabled, "vdn-hybrid": enabled}
        ),
    )


OPTIMIZATION = _optimization_spec()
