# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Block-streaming composition (Task 2 increment 5).

Group offloading must stream only real parameters and skip the param-less
HostResidentAdaLN, and its enabled-state must be detectable (the invariant the
ComponentsManager offload hook reads to avoid moving a group-offloaded
transformer). CPU-only with use_stream=False so no CUDA is required; the
real-store GPU bitwise gate lives in tests/test_reference_parity.py.
"""

import pytest
import torch
import torch.nn as nn

from omni_infinity.adaln import AdaLNEntry, HostResidentAdaLN


@pytest.fixture
def cpu_accelerator(monkeypatch):
    """Present a CPU device when no real accelerator is available.

    diffusers' ``ModuleGroup.__init__`` resolves
    ``torch.accelerator.current_accelerator().type`` at construction time,
    which is ``None`` on CPU-only runners (the CI ``test`` job installs CPU
    torch) and raises ``AttributeError``. These tests never onload/forward
    (``use_stream=False``), so the accelerator module is only needed for
    construction, never at runtime. On GPU hosts ``current_accelerator()`` is
    already a device, so this fixture is a no-op.
    """
    accel = getattr(torch, "accelerator", None)
    if accel is not None and accel.current_accelerator() is None:
        monkeypatch.setattr(
            accel, "current_accelerator", lambda *a, **k: torch.device("cpu")
        )


class _Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden, hidden, bias=False)
        weight = torch.randn(6 * hidden * 3, hidden, dtype=torch.bfloat16)
        bias = torch.randn(6 * hidden * 3, dtype=torch.bfloat16)
        self.adaln_proj = HostResidentAdaLN(AdaLNEntry(weight, bias), hidden)


class _Stack(nn.Module):
    _supports_group_offloading = True

    def __init__(self, hidden=4, blocks=3):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_Block(hidden) for _ in range(blocks)]
        )


def test_group_offload_skips_param_less_adaln(cpu_accelerator):
    from diffusers.hooks import apply_group_offloading

    model = _Stack().to(torch.bfloat16)
    entries = [b.adaln_proj._entry.weight for b in model.transformer_blocks]

    apply_group_offloading(
        module=model,
        onload_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=False,
    )

    for block, original in zip(model.transformer_blocks, entries):
        assert block.adaln_proj._entry.weight is original
        assert block.adaln_proj._entry.weight.device.type == "cpu"
        assert list(block.adaln_proj.parameters()) == []


def test_group_offload_state_is_detectable_for_manager_compose(cpu_accelerator):
    from diffusers.hooks import apply_group_offloading
    from diffusers.hooks.group_offloading import _is_group_offload_enabled

    model = _Stack().to(torch.bfloat16)
    assert not _is_group_offload_enabled(model)
    apply_group_offloading(
        module=model,
        onload_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=False,
    )
    # The predicate ComponentsManager's CustomOffloadHook reads to skip moving a
    # group-offloaded transformer -> confirms the two offloaders compose.
    assert _is_group_offload_enabled(model)


class _FakeDecoder(nn.Module):
    def __init__(self, layers=4, hidden=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers)]
        )


class _FakeEncoder(nn.Module):
    _supports_group_offloading = True

    def __init__(self):
        super().__init__()
        self.model = _FakeDecoder()


def test_group_offload_targets_nested_decoder_layers(cpu_accelerator):
    from diffusers.hooks import apply_group_offloading
    from diffusers.hooks.group_offloading import _is_group_offload_enabled

    encoder = _FakeEncoder().to(torch.bfloat16)
    apply_group_offloading(
        module=encoder.model,
        onload_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=False,
    )
    assert _is_group_offload_enabled(encoder.model)
