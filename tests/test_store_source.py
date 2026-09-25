# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""StoreComponentSource: one-read group fetches from a moe-store v2 store.

The synthetic round-trip converts a tiny H3-like modular pipeline with
moe-store and asserts byte-exact state-dict reconstruction per component
plus the AdaLN bundle group split. The real-store test runs only where
``OMNI_H3_STORE`` points at a converted MiniMax-H3 store (dev host).
"""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from omni_infinity.store import StoreComponentSource, _block_index

HIDDEN = 16


def _write_tiny_h3_pipeline(root: Path) -> dict[str, dict[str, torch.Tensor]]:
    root.mkdir()
    (root / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"})
    )
    torch.manual_seed(11)
    states: dict[str, dict[str, torch.Tensor]] = {}

    for component, blocks in (("transformer", 2), ("transformer_ref", 1)):
        transformer = root / component
        transformer.mkdir()
        (transformer / "config.json").write_text(
            json.dumps(
                {
                    "_class_name": "MiniMaxH3Transformer3DModel",
                    "_diffusers_version": "0.40.0",
                }
            )
        )
        state = {}
        if component == "transformer":
            state["proj_in.weight"] = torch.randn(
                HIDDEN, 8, dtype=torch.bfloat16
            )
        for block in range(blocks):
            prefix = f"transformer_blocks.{block}"
            state[f"{prefix}.adaln_proj.linear.weight"] = torch.randn(
                6 * HIDDEN, HIDDEN, dtype=torch.bfloat16
            )
            state[f"{prefix}.adaln_proj.linear.bias"] = torch.randn(
                6 * HIDDEN, dtype=torch.bfloat16
            )
            attention_name = (
                "attn.to_q.weight"
                if component == "transformer"
                else "attn.weight"
            )
            state[f"{prefix}.{attention_name}"] = torch.randn(
                HIDDEN, HIDDEN, dtype=torch.bfloat16
            )
        save_file(
            state, str(transformer / "diffusion_pytorch_model.safetensors")
        )
        states[component] = state

    vae = root / "vae"
    vae.mkdir()
    (vae / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "AutoencoderKLMiniMaxH3",
                "_diffusers_version": "0.40.0",
            }
        )
    )
    vae_state = {
        "encoder.conv_in.weight": torch.randn(HIDDEN, 3, dtype=torch.bfloat16)
    }
    save_file(vae_state, str(vae / "diffusion_pytorch_model.safetensors"))
    states["vae"] = vae_state
    return states


def test_synthetic_store_round_trips_byte_exact(tmp_path):
    pytest.importorskip("moe_store")
    from moe_store.convert.convert import convert_checkpoint

    root = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    states = _write_tiny_h3_pipeline(root)
    convert_checkpoint(str(root), str(store_dir))

    source = StoreComponentSource(store_dir)
    assert set(source.components()) == {"transformer", "transformer_ref", "vae"}

    for component, expected in states.items():
        loaded = source.load_component_state_dict(component)
        assert set(loaded) == set(expected)
        for name, tensor in expected.items():
            assert torch.equal(loaded[name], tensor), f"{component}.{name}"

    adaln = source.adaln_groups("transformer")
    assert len(adaln) == 2
    for group in adaln:
        names = set(source.read_group(group))
        assert all("adaln_proj" in name for name in names)


def test_read_adaln_cache_round_trips_byte_exact(tmp_path):
    pytest.importorskip("moe_store")
    from moe_store.convert.convert import convert_checkpoint

    root = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    states = _write_tiny_h3_pipeline(root)
    convert_checkpoint(str(root), str(store_dir))

    source = StoreComponentSource(store_dir)
    cache = source.read_adaln_cache("transformer")

    assert set(cache) == {0, 1}
    expected = states["transformer"]
    for block, entry in cache.items():
        prefix = f"transformer_blocks.{block}.adaln_proj.linear"
        assert torch.equal(entry.weight, expected[f"{prefix}.weight"])
        assert torch.equal(entry.bias, expected[f"{prefix}.bias"])
        assert entry.scale is None


def test_transformer_ref_groups_are_isolated_and_strip_one_prefix(tmp_path):
    pytest.importorskip("moe_store")
    from moe_store.convert.convert import convert_checkpoint

    root = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_tiny_h3_pipeline(root)
    convert_checkpoint(str(root), str(store_dir))

    source = StoreComponentSource(store_dir)
    source.index.groups[:] = sorted(
        source.index.groups,
        key=lambda group: source.stage_name(group).endswith(".adaln"),
    )
    groups = source.groups_for("transformer_ref")
    assert all(
        source.stage_name(g).startswith("transformer_ref") for g in groups
    )
    assert len(source.adaln_groups("transformer_ref")) == 1
    assert set(source.read_group(groups[0])) == {
        "transformer_blocks.0.attn.weight"
    }


def test_group_fetch_is_single_read(tmp_path, monkeypatch):
    pytest.importorskip("moe_store")
    from moe_store.convert.convert import convert_checkpoint

    root = tmp_path / "ckpt"
    store_dir = tmp_path / "store"
    _write_tiny_h3_pipeline(root)
    convert_checkpoint(str(root), str(store_dir))

    source = StoreComponentSource(store_dir)
    group = source.adaln_groups("transformer")[0]

    reads = []
    original_open = open

    def counting_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if str(path).startswith(str(store_dir)) and "store_data" in str(path):
            original_read = handle.read

            def counted_read(*read_args):
                reads.append(read_args)
                return original_read(*read_args)

            handle.read = counted_read
        return handle

    monkeypatch.setattr("builtins.open", counting_open)
    source.read_group(group)
    assert len(reads) == 1
    assert reads[0] == (group.total_size,)


_REAL_STORE = os.environ.get("OMNI_H3_STORE")


@pytest.mark.skipif(
    not (_REAL_STORE and Path(_REAL_STORE).is_dir()),
    reason="OMNI_H3_STORE not set",
)
def test_real_h3_store_structure_and_audio_vae_bytes():
    source = StoreComponentSource(_REAL_STORE)
    assert set(source.components()) >= {
        "transformer",
        "text_encoder",
        "vae",
        "audio_vae",
    }
    assert len(source.adaln_groups("transformer")) == 50

    # The store leaves group.layer_id == -1, so the cache keys blocks by the
    # index parsed from the member name; assert that covers all 50 blocks
    # (metadata only — reading the 26 GB of bundles is covered elsewhere).
    blocks = {
        _block_index(group.members[0].name)
        for group in source.adaln_groups("transformer")
    }
    assert blocks == set(range(50))

    state = source.load_component_state_dict("audio_vae")
    assert state

    from safetensors import safe_open

    snapshot = Path(os.environ["OMNI_H3_SNAPSHOT"]) / "audio_vae"
    shard = next(snapshot.glob("*.safetensors"))
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            source_tensor = handle.get_tensor(name)
            # The converter's CastPolicy writes fp32 source params as the
            # store compute dtype (bf16), so compare against the cast value.
            assert torch.equal(
                state[name], source_tensor.to(state[name].dtype)
            ), name
