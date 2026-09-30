# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Whole-call caching for vision-tower embeddings."""

from __future__ import annotations

import threading
from collections import OrderedDict

from omni_infinity.caches._tensor_tree import tree_nbytes


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
                "bytes": sum(tree_nbytes(value) for value in self._entries.values()),
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
