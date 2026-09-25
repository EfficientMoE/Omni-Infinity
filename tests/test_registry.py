# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the model-arch / optimization category registry."""

import pytest

from omni_infinity import registry


def test_categories_are_disjoint_and_complete():
    arch_names = set(registry.ARCHS)
    opt_names = set(registry.OPTIMIZATIONS)
    assert arch_names == {"h3-dense", "vdn-hybrid"}
    assert opt_names == {
        "adaln-host-cache",
        "fp8",
        "block-stream",
        "text-encoder-stream",
    }
    assert not arch_names & opt_names


def test_arch_specs_point_at_importable_runners():
    for spec in registry.ARCHS.values():
        cls = registry.runner_class(spec)
        assert hasattr(cls, "from_pretrained")


def test_optimizations_declare_supported_archs():
    for spec in registry.OPTIMIZATIONS.values():
        assert spec.supported_archs
        for arch in spec.supported_archs:
            assert arch in registry.ARCHS


def test_adaln_host_cache_is_h3_only():
    # The AdaLN host cache reads a moe-store; the VDN diffusers path
    # has no store, so the optimization must not claim vdn-hybrid.
    spec = registry.OPTIMIZATIONS["adaln-host-cache"]
    assert spec.supported_archs == ("h3-dense",)


def test_runner_kwargs_for_merges_optimizations():
    kwargs = registry.runner_kwargs_for("vdn-hybrid", ["fp8", "block-stream"])
    assert kwargs["fp8"] is True
    assert kwargs["block_stream_blocks_per_group"] == 1
    assert kwargs["offload"] is True


def test_runner_kwargs_for_rejects_unsupported_pair():
    with pytest.raises(ValueError, match="adaln-host-cache"):
        registry.runner_kwargs_for("vdn-hybrid", ["adaln-host-cache"])


def test_runner_kwargs_for_uses_arch_specific_fp8_names():
    dense = registry.runner_kwargs_for("h3-dense", ["fp8"])
    hybrid = registry.runner_kwargs_for("vdn-hybrid", ["fp8"])
    assert dense == {"transformer_fp8": True}
    assert hybrid == {"fp8": True}


def test_resolve_profile_returns_runner_checkpoint_and_merged_kwargs():
    profile = registry.resolve_profile(
        "h3-dense", ["adaln-host-cache", "block-stream"]
    )
    assert profile.runner is registry.runner_class(registry.ARCHS["h3-dense"])
    assert profile.checkpoint == "MiniMaxAI/MiniMax-H3"
    assert profile.runner_kwargs == {
        "adaln_host_cache": True,
        "offload": True,
        "block_stream_blocks_per_group": 1,
    }
