# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import pytest

from omni_infinity.runner import _h3_component_class
from omni_infinity.store import _block_index


def test_block_index_reads_the_transformer_block_number():
    assert (
        _block_index("transformer.transformer_blocks.12.attn.to_q.weight") == 12
    )
    with pytest.raises(ValueError, match="no transformer block index"):
        _block_index("transformer.adaln.weight")


def test_unknown_component_has_no_store_loadable_class():
    pytest.importorskip("diffusers")
    assert _h3_component_class("vae").__name__ == "AutoencoderKLMiniMaxH3"
    assert (
        _h3_component_class("audio_vae").__name__
        == "AutoencoderKLMiniMaxH3Audio"
    )
    with pytest.raises(ValueError, match="component 'tokenizer'"):
        _h3_component_class("tokenizer")
