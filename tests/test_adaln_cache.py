# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Host-resident AdaLN cache: drop-in parity and host-residency.

`HostResidentAdaLN` must be a byte-for-byte replacement for the reference
`MiniMaxH3AdaLayerNormModulation` (Task 2 increment 3, issue #2) — the whole
point is to move the ~26 GB of AdaLN weights off the GPU without perturbing
the latents. These tests run on CPU with no store, so they gate the mechanism
in CI; the real-store GPU bitwise parity is covered by
`tests/test_reference_parity.py` with `OMNI_H3_ADALN_CACHE=1`.
"""

import torch

from omni_infinity.adaln import AdaLNEntry, HostResidentAdaLN

HIDDEN = 4
TIME_EMBED = 8


def _reference_adaln():
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3AdaLayerNormModulation,
    )

    torch.manual_seed(3)
    module = MiniMaxH3AdaLayerNormModulation(
        time_embed_dim=TIME_EMBED, hidden_size=HIDDEN
    )
    return module.to(torch.bfloat16).eval()


def test_host_resident_adaln_matches_reference_bitwise():
    import pytest

    pytest.importorskip("diffusers")
    reference = _reference_adaln()
    entry = AdaLNEntry(
        reference.linear.weight.detach().clone(),
        reference.linear.bias.detach().clone(),
    )
    host = HostResidentAdaLN(entry, HIDDEN).eval()

    temb = torch.randn(5, TIME_EMBED)
    with torch.no_grad():
        expected = reference(temb)
        actual = host(temb)

    assert len(actual) == len(expected) == 6
    for index, (a, e) in enumerate(zip(actual, expected)):
        assert torch.equal(a, e), index


def test_host_resident_adaln_is_invisible_to_module_to():
    weight = torch.randn(6 * HIDDEN * 3, TIME_EMBED, dtype=torch.bfloat16)
    bias = torch.randn(6 * HIDDEN * 3, dtype=torch.bfloat16)
    host = HostResidentAdaLN(AdaLNEntry(weight, bias), HIDDEN)

    host.to(torch.float16)

    assert host._entry.weight.dtype == torch.bfloat16
    assert host._entry.weight.device.type == "cpu"
    assert list(host.state_dict()) == []
    assert list(host.named_parameters()) == []
    assert list(host.named_buffers()) == []


def test_adaln_entry_materialize_bf16_is_byte_exact():
    weight = torch.randn(12, TIME_EMBED, dtype=torch.bfloat16)
    bias = torch.randn(12, dtype=torch.bfloat16)
    entry = AdaLNEntry(weight, bias)

    materialized_weight, materialized_bias = entry.materialize(
        torch.device("cpu")
    )

    assert torch.equal(materialized_weight, weight)
    assert torch.equal(materialized_bias, bias)
    assert materialized_weight.dtype == torch.bfloat16
