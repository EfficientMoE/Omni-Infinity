# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Structure-preserving helpers over nested tensor containers.

The cache entries in this package hold whatever the pipeline hands back:
a bare tensor, a tuple/list of tensors, or a ``ModelOutput``-style mapping
(``transformers`` outputs are ``OrderedDict`` subclasses reconstructable
from keyword arguments). These helpers map/measure/hash the tensor leaves
without flattening away the structure, so a cached value replays with the
exact type the caller produced.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from typing import Any, Callable

import torch


def tree_map(fn: Callable[[torch.Tensor], torch.Tensor], value: Any) -> Any:
    """Apply *fn* to every tensor leaf, rebuilding the same container type.

    Mappings are rebuilt via ``type(value)(**mapped)`` (the ``ModelOutput``
    contract); non-tensor leaves are passed through by reference.
    """
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, Mapping):
        return type(value)(
            **{key: tree_map(fn, item) for key, item in value.items()}
        )
    if isinstance(value, tuple):
        mapped = (tree_map(fn, item) for item in value)
        if hasattr(value, "_fields"):  # namedtuple
            return type(value)(*mapped)
        return tuple(mapped)
    if isinstance(value, list):
        return [tree_map(fn, item) for item in value]
    return value


def tree_tensors(value: Any) -> list[torch.Tensor]:
    """Collect the tensor leaves in deterministic traversal order."""
    leaves: list[torch.Tensor] = []

    def _collect(tensor: torch.Tensor) -> torch.Tensor:
        leaves.append(tensor)
        return tensor

    tree_map(_collect, value)
    return leaves


def tree_nbytes(value: Any) -> int:
    return sum(
        leaf.untyped_storage().nbytes()
        if leaf.is_contiguous()
        else leaf.numel() * leaf.element_size()
        for leaf in tree_tensors(value)
    )


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Canonical content bytes of a tensor (dtype/shape tagged)."""
    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    if flat.numel() == 0:
        data = b""
    else:
        data = flat.view(torch.uint8).numpy().tobytes()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
    return header + data


def update_hash_for_value(hasher: "hashlib._Hash", value: Any) -> None:
    """Feed a canonical, type-tagged serialization of *value* into *hasher*.

    Every part is length-prefixed so adjacent values cannot collide by
    shifting bytes across a boundary.
    """

    def _update(part: bytes) -> None:
        hasher.update(len(part).to_bytes(8, "little"))
        hasher.update(part)

    if value is None:
        _update(b"none:")
    elif isinstance(value, torch.Tensor):
        _update(b"tensor:" + tensor_bytes(value))
    elif isinstance(value, bytes):
        _update(b"bytes:" + value)
    elif isinstance(value, str):
        _update(b"str:" + value.encode("utf-8"))
    elif isinstance(value, (bool, int, float)):
        _update(f"scalar:{value!r}".encode())
    elif isinstance(value, Mapping):
        _update(b"mapping:")
        for key in value:
            update_hash_for_value(hasher, str(key))
            update_hash_for_value(hasher, value[key])
    elif isinstance(value, (tuple, list)):
        _update(b"sequence:")
        for item in value:
            update_hash_for_value(hasher, item)
    elif hasattr(value, "tobytes") and hasattr(value, "mode"):
        # PIL image (duck-typed): pixel content plus geometry and mode.
        _update(
            b"image:" + f"{value.mode}|{value.size}|".encode() + value.tobytes()
        )
    elif hasattr(value, "tobytes") and hasattr(value, "dtype"):
        # numpy array (duck-typed): dtype/shape-tagged content bytes.
        _update(
            b"array:"
            + f"{value.dtype}|{getattr(value, 'shape', ())}|".encode()
            + value.tobytes()
        )
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        # e.g. MiniMaxH3ImageReference: class identity plus every field.
        _update(b"dataclass:" + type(value).__name__.encode())
        for field in dataclasses.fields(value):
            update_hash_for_value(hasher, field.name)
            update_hash_for_value(hasher, getattr(value, field.name))
    else:
        raise TypeError(
            f"cannot canonically hash a {type(value).__name__}; pass "
            "bytes, str, scalars, tensors, PIL images, or containers "
            "of those"
        )
