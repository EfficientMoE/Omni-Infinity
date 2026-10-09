# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Rung C boundary protection: ``--fp8-protect-blocks first:N,last:M``.

P1+P4 plan rung C: FP8 error compounds through H3's transformer, and the
survey says residual pathways self-correct mid-blocks. Keeping the entry
and exit blocks bf16 caps the latent deviation; these tests cover the
spec parser, the index-set builder, and the skip wiring into
``apply_scaled_fp8_casting``.
"""

import pytest
import torch
import torch.nn as nn

from omni_infinity.fp8 import (
    ScaledFp8Linear,
    apply_scaled_fp8_casting,
    parse_protect_blocks,
    protect_block_indices,
)


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("first:2,last:3", (2, 3)),
        ("last:3,first:2", (2, 3)),
        ("first:2", (2, 0)),
        ("last:3", (0, 3)),
        (" first:1 , last:1 ", (1, 1)),
        ("first:0,last:0", (0, 0)),
    ],
)
def test_parse_protect_blocks_valid(spec, expected):
    assert parse_protect_blocks(spec) == expected


@pytest.mark.parametrize(
    "spec",
    ["", "first", "first:", "first:-1", "mid:2", "first:2,first:3", "2,3"],
)
def test_parse_protect_blocks_invalid(spec):
    with pytest.raises(ValueError):
        parse_protect_blocks(spec)


def test_protect_block_indices_boundaries():
    assert protect_block_indices(50, 2, 3) == frozenset({0, 1, 47, 48, 49})
    assert protect_block_indices(50, 0, 0) == frozenset()
    assert protect_block_indices(4, 0, 2) == frozenset({2, 3})


def test_protect_block_indices_overlap_covers_all():
    assert protect_block_indices(4, 3, 3) == frozenset({0, 1, 2, 3})


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _TinyBlockStack(nn.Module):
    def __init__(self, blocks: int = 6):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_Block() for _ in range(blocks)]
        )


def test_apply_casting_with_protect_indices_keeps_boundaries_bf16():
    model = _TinyBlockStack(6)
    skip = protect_block_indices(6, 2, 1)
    replaced = apply_scaled_fp8_casting(model, torch.bfloat16, skip_blocks=skip)
    assert replaced == 3
    kept = [
        i
        for i, b in enumerate(model.transformer_blocks)
        if not isinstance(b.proj, ScaledFp8Linear)
    ]
    assert kept == [0, 1, 5]
