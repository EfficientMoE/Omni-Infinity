# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Hashing and traversal helpers for tensor trees."""

from __future__ import annotations

import struct
from io import BytesIO

import torch
from PIL import Image


def _update_bytes(hasher, tag: bytes, payload: bytes) -> None:
    hasher.update(tag)
    hasher.update(struct.pack("<Q", len(payload)))
    hasher.update(payload)


def update_hash_for_value(hasher, value) -> None:
    """Update ``hasher`` with a type-preserving representation of ``value``."""
    if value is None:
        _update_bytes(hasher, b"none", b"")
    elif isinstance(value, bool):
        _update_bytes(hasher, b"bool", b"1" if value else b"0")
    elif isinstance(value, int):
        _update_bytes(hasher, b"int", str(value).encode("ascii"))
    elif isinstance(value, float):
        _update_bytes(hasher, b"float", struct.pack("<d", float(value)))
    elif isinstance(value, str):
        _update_bytes(hasher, b"str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        _update_bytes(hasher, b"bytes", value)
    elif isinstance(value, (list, tuple)):
        hasher.update(b"list" if isinstance(value, list) else b"tuple")
        hasher.update(struct.pack("<Q", len(value)))
        for item in value:
            update_hash_for_value(hasher, item)
    elif isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("dictionary keys must be strings")
        hasher.update(b"dict")
        hasher.update(struct.pack("<Q", len(value)))
        for key in sorted(value):
            update_hash_for_value(hasher, key)
            update_hash_for_value(hasher, value[key])
    elif isinstance(value, torch.Tensor):
        hasher.update(b"tensor")
        update_hash_for_value(hasher, str(value.dtype))
        update_hash_for_value(hasher, tuple(value.shape))
        payload = value.detach().cpu().contiguous().numpy().tobytes()
        _update_bytes(hasher, b"data", payload)
    elif isinstance(value, Image.Image):
        buffer = BytesIO()
        value.save(buffer, format="PNG")
        _update_bytes(hasher, b"image", buffer.getvalue())
    else:
        raise TypeError(f"unsupported hash value: {type(value).__name__}")


def tree_map(fn, value):
    """Apply ``fn`` to tensor leaves while preserving the tree structure."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, list):
        return [tree_map(fn, item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_map(fn, item) for item in value)
    if isinstance(value, dict):
        return {key: tree_map(fn, item) for key, item in value.items()}
    return value


def tree_tensors(value) -> list[torch.Tensor]:
    """Return tensor leaves in tree walk order."""
    tensors = []

    def collect(tensor):
        tensors.append(tensor)
        return tensor

    tree_map(collect, value)
    return tensors


def tree_nbytes(value) -> int:
    """Return the total storage size of tensor leaves."""
    return sum(tensor.nbytes for tensor in tree_tensors(value))
