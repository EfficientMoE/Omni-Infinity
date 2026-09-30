# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_encoder_prefix_cache_is_not_implemented():
    assert not Path("omni_infinity/caches/prefix.py").exists()
    for path in Path("omni_infinity").rglob("*.py"):
        text = path.read_text()
        assert "use_cache=True" not in text
        assert "use_cache = True" not in text


def test_c2_note_states_the_reuse_rule():
    text = Path("docs/caches_c2_prefix.md").read_text()
    for sentence in (
        "Deferred until the Qwen3-VL tower stays resident.",
        "A hit is a complete leading block only.",
        "Vision tokens in front of the text are part of the prefix.",
        "Picture 1 matches only when the image bytes match and it is first.",
        "Do not reorder [Shot N] blocks.",
        "Do not install ContextPilot.",
        "def prefix_blocks(token_ids: tuple[int, ...], block_size: int) -> "
        "list[bytes]:",
        "This plan does not implement prefix_blocks.",
    ):
        assert sentence in text
