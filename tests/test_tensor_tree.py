# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest
import torch
from PIL import Image

from omni_infinity.caches._tensor_tree import (
    tree_map,
    tree_nbytes,
    tree_tensors,
    update_hash_for_value,
)


def _digest(value):
    hasher = hashlib.sha256()
    update_hash_for_value(hasher, value)
    return hasher.digest()


def test_hash_tags_keep_string_and_int_apart():
    assert _digest("1") != _digest(1)
    assert _digest({"b": 1, "a": 2}) == _digest({"a": 2, "b": 1})


def test_none_slots_do_not_shift():
    assert _digest((None, "x")) != _digest(("x", None))


def test_tree_map_moves_tensor_leaves_only():
    first = torch.zeros(1)
    second = torch.ones(2)
    value = {"a": first, "b": None, "c": (second, "leaf")}

    moved = tree_map(lambda tensor: tensor + 1, value)

    assert moved["b"] is None
    assert int(moved["a"]) == 1
    assert moved["c"][1] == "leaf"
    assert tree_tensors(value) == [first, second]


def test_tensor_hash_and_nbytes_include_all_bytes():
    tensor = torch.tensor([1, 2, 3], dtype=torch.uint8)
    changed = tensor.clone()
    changed[-1] = 4

    assert _digest(tensor) != _digest(changed)
    assert tree_nbytes(tensor) == tensor.nbytes


def test_pil_images_hash_as_png_bytes():
    first = Image.new("RGB", (2, 2), "red")
    same = Image.new("RGB", (2, 2), "red")
    recolored = Image.new("RGB", (2, 2), "blue")

    assert _digest(first) == _digest(same)
    assert _digest(first) != _digest(recolored)


def test_unhashable_value_raises_type_error():
    with pytest.raises(TypeError):
        _digest(object())
