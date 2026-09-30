# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""C1 — exact cross-request condition cache (issue #24).

Keeps the layer-50 ``prompt_embeds``, ``text_token_tags``, and the
keyframe/reference VAE ``condition_latents`` keyed by the hash of the
prompt string plus the condition image bytes, **in that order**. This is
the VDN prompt ``.pt`` file lifted into the runner: the hit condition is
that the *entire* condition matches — a shared ``[Shot 1]`` opener is not
a hit, and no prefix semantics exist anywhere in the key.

Why this is exact for the H3 modular pipeline (diffusers 0.40):

- the Qwen3-VL encode runs with ``use_cache=False`` and no sampling, so
  ``prompt_embeds`` / ``text_token_tags`` are pure functions of the
  condition;
- the keyframe/reference VAE posterior is sampled under a *fresh*
  generator seeded independently of the request
  (``components.keyframe_encode_seed``), so ``condition_latents`` are a
  pure function of the image bytes too;
- neither encode block consumes the request ``generator``, so skipping
  them leaves the denoise RNG stream untouched.

A hit therefore replays bitwise — but the cache stays opt-in and off the
parity gates regardless, per the issue #24 contract.

The cross-request analogue in vLLM-Omni is the AR stage-output prefix
cache (block-hashed CPU mirrors); we deliberately copy only its *storage*
posture (host-resident, exact identity) and not the prefix semantics,
because H3's fixed shot order makes a partial prefix worthless (#23).
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from omni_infinity.caches._tensor_tree import (
    tree_map,
    tree_nbytes,
    tree_tensors,
    update_hash_for_value,
)

logger = logging.getLogger(__name__)

# Bump when the on-disk entry layout changes; part of every key.
CACHE_SCHEMA_VERSION = 1

# Intermediates captured from the modular state after a miss. The names
# are the H3 modular blocks' own intermediate outputs (text_encoder ->
# prompt_embeds/text_token_tags, vae_encoder -> condition latents).
DEFAULT_CAPTURE = (
    "prompt_embeds",
    "text_token_tags",
    "condition_latents",
    "audio_condition_latents",
)
# An entry without these is not worth storing.
DEFAULT_REQUIRED = ("prompt_embeds",)

# Blocks a hit skips. Both are condition encoders whose outputs the
# denoise block re-declares as inputs; ``before_encode`` (cheap CPU image
# normalization) stays, because the denoise block also needs its
# ``normalized_references``/``keyframe_anchors`` intermediates.
ENCODE_BLOCKS = ("text_encoder", "vae_encoder")


def condition_key(
    namespace: str,
    prompt: str,
    media: Sequence[Any] = (),
) -> str:
    """Hash of the whole condition: prompt string then image bytes, in
    order. ``media`` slots keep their position (a ``None`` first frame is
    hashed as an empty slot), so ``image=None, last_image=x`` and
    ``image=x, last_image=None`` never collide."""
    hasher = hashlib.sha256()
    update_hash_for_value(hasher, f"omni-condition-v{CACHE_SCHEMA_VERSION}")
    update_hash_for_value(hasher, namespace)
    update_hash_for_value(hasher, prompt)
    update_hash_for_value(hasher, tuple(media))
    return hasher.hexdigest()


class ConditionEntry:
    """One cached condition: named tensor trees, host-resident."""

    __slots__ = ("values", "nbytes")

    def __init__(self, values: dict[str, Any]):
        self.values = values
        self.nbytes = sum(tree_nbytes(value) for value in values.values())

    def to(self, device) -> dict[str, Any]:
        if device is None:
            return dict(self.values)
        return {
            name: tree_map(lambda t: t.to(device), value)
            for name, value in self.values.items()
        }


class ConditionCache:
    """LRU memory tier plus an optional ``.pt`` disk tier.

    The disk tier is the VDN prompt-cache file made runner-managed: one
    ``<key>.pt`` per condition, written atomically, loaded with
    ``weights_only=True``. Entries always live on the CPU.
    """

    def __init__(
        self,
        *,
        max_entries: int = 8,
        max_bytes: int | None = None,
        cache_dir: str | os.PathLike | None = None,
        capture: Sequence[str] = DEFAULT_CAPTURE,
        required: Sequence[str] = DEFAULT_REQUIRED,
    ):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        missing = set(required) - set(capture)
        if missing:
            raise ValueError(
                f"required names {sorted(missing)} are not captured"
            )
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.capture = tuple(capture)
        self.required = tuple(required)
        self.hits = 0
        self.misses = 0
        self._entries: OrderedDict[str, ConditionEntry] = OrderedDict()
        self._lock = threading.Lock()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> ConditionEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return entry
        entry = self._load_from_disk(key)
        with self._lock:
            if entry is not None:
                self.hits += 1
                self._insert(key, entry)
            else:
                self.misses += 1
        return entry

    def put(self, key: str, values: dict[str, Any]) -> ConditionEntry | None:
        """Store the captured intermediates; drop ``None`` values.

        Returns ``None`` (and stores nothing) when a required name is
        missing — e.g. a pipeline that does not expose the intermediate.
        """
        kept = {
            name: tree_map(lambda t: t.detach().to("cpu"), values[name])
            for name in self.capture
            if values.get(name) is not None
        }
        if not self._valid_values(kept):
            return None
        entry = ConditionEntry(kept)
        with self._lock:
            self._insert(key, entry)
        self._save_to_disk(key, kept)
        return entry

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "entries": len(self._entries),
                "bytes": sum(e.nbytes for e in self._entries.values()),
            }

    def _insert(self, key: str, entry: ConditionEntry) -> None:
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        if self.max_bytes is not None:
            while (
                len(self._entries) > 1
                and sum(e.nbytes for e in self._entries.values())
                > self.max_bytes
            ):
                self._entries.popitem(last=False)

    def _disk_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}.pt"

    def _load_from_disk(self, key: str) -> ConditionEntry | None:
        path = self._disk_path(key)
        if path is None or not path.is_file():
            return None
        try:
            values = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            logger.warning("dropping unreadable cache entry %s", path)
            return None
        if not isinstance(values, dict):
            return None
        kept = {
            name: values[name]
            for name in self.capture
            if values.get(name) is not None
        }
        if not self._valid_values(kept):
            return None
        return ConditionEntry(kept)

    def _valid_values(self, values: dict[str, Any]) -> bool:
        if any(name not in values for name in self.required):
            return False
        return all(tree_tensors(value) for value in values.values())

    def _save_to_disk(self, key: str, values: dict[str, Any]) -> None:
        path = self._disk_path(key)
        if path is None:
            return
        tmp_name = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self.cache_dir), suffix=".tmp"
            )
            with os.fdopen(fd, "wb") as handle:
                torch.save(values, handle)
            os.replace(tmp_name, path)
        except Exception:
            logger.warning("could not persist cache entry %s", path)
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass


def declared_inputs(pipeline) -> frozenset[str] | None:
    """The input names a modular pipeline's blocks declare, or ``None``
    when the pipeline exposes no block specs (fail-open)."""
    blocks = getattr(pipeline, "blocks", None)
    sub_blocks = getattr(blocks, "sub_blocks", None)
    if sub_blocks is None:
        return None
    names = set()
    try:
        for block in sub_blocks.values():
            for spec in getattr(block, "inputs", ()) or ():
                name = getattr(spec, "name", None)
                if name:
                    names.add(name)
    except Exception:
        return None
    return frozenset(names) or None


def build_conditioned_pipeline(pipeline, skip_blocks=ENCODE_BLOCKS):
    """A second view of *pipeline* without its condition-encoder blocks.

    Uses the modular-diffusers split-blocks pattern: rebuild the
    sequential blocks minus ``skip_blocks``, then share the already
    loaded component objects. Returns ``None`` (fail-open) when the
    pipeline's blocks cannot be reduced — the caller then keeps encoding.
    """
    blocks = getattr(pipeline, "blocks", None)
    sub_blocks = getattr(blocks, "sub_blocks", None)
    if not sub_blocks or not any(name in sub_blocks for name in skip_blocks):
        return None
    try:
        from diffusers.modular_pipelines import SequentialPipelineBlocks

        reduced = SequentialPipelineBlocks.from_blocks_dict(
            {
                name: block
                for name, block in sub_blocks.items()
                if name not in skip_blocks
            }
        )
        conditioned = reduced.init_pipeline()
        shared = {
            name: component
            for name, component in pipeline.components.items()
            if name in conditioned.components and component is not None
        }
        conditioned.update_components(**shared)
        return conditioned
    except Exception:
        logger.warning(
            "could not build the conditioned (encoder-less) pipeline; "
            "the condition cache will keep re-encoding",
            exc_info=True,
        )
        return None
