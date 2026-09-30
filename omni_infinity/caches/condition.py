# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Exact whole-condition cache for H3 generation."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

from omni_infinity.caches._tensor_tree import (
    tree_map,
    tree_nbytes,
    update_hash_for_value,
)

CACHE_SCHEMA_VERSION = 1
DEFAULT_CAPTURE = (
    "prompt_embeds",
    "text_token_tags",
    "condition_latents",
    "audio_condition_latents",
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
                dir=self.cache_dir, prefix=f".{key}.", suffix=".tmp", delete=False
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
                        raise ValueError("condition entry is missing required values")
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
