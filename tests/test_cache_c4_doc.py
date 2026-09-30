# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_c4_note_states_the_reuse_rule():
    text = Path("docs/caches_c4_shape.md").read_text()
    for sentence in (
        (
            "Flex BlockMask, gather-index, and delta-rule backend caches "
            "stay in third_party/vdn-minimax-h3."
        ),
        (
            "Caption content is not keyed directly, but its derived layout "
            "fields, including `seq_len` and `video_start`, are part of the "
            "BlockMask key."
        ),
        (
            "A new caption length changes those fields and misses the "
            "inference FLASH compile."
        ),
        "One seq_len is compiled per process.",
        "This issue does not add a shape cache.",
        "This plan does not modify third_party.",
    ):
        assert sentence in text


def test_c4_adds_no_shape_module():
    assert not Path("omni_infinity/caches/shape.py").exists()
