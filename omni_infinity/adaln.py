# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Host-resident AdaLN branch cache for the MiniMax-H3 transformer.

Task 2 increment 3 (issue #2): the 50 per-block AdaLN modulation projections
(``transformer_blocks.N.adaln_proj``, ~13B params / 26 GB bf16 — ~40% of the
33B transformer) are the largest cacheable slice of the block stack. Instead
of keeping them GPU-resident, each block's weight/bias is held in host memory
as a plain (non-``Parameter``, non-buffer) attribute — so ``Module.to(device)``
leaves it on the host — and materialized onto the compute device on demand
inside the block's forward pass, then dropped as it falls out of scope.

``HostResidentAdaLN.forward`` is a byte-for-byte reproduction of
``diffusers.MiniMaxH3AdaLayerNormModulation.forward`` (SiLU at ``temb``'s
incoming precision, cast to the projection dtype, linear, split into the six
``shift/scale/gate`` tensors), so store-sourced latents stay bitwise-identical
to the full-resident reference. The compute dtype is pinned independent of the
storage dtype, so increment 4's FP8 groups dequantize inside
``AdaLNEntry.materialize`` with the forward pass unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLNEntry:
    """Host-resident weight/bias for one block's AdaLN projection.

    ``weight``/``bias`` are the store tensors kept on the host.
    :meth:`materialize` returns them on ``device``. In the bf16 path
    (``scale is None``) it is a byte-exact host-to-device copy; the increment 4
    FP8 path attaches ``scale`` and dequantizes to ``compute_dtype`` here, so
    :class:`HostResidentAdaLN` never sees the storage dtype.
    """

    __slots__ = ("weight", "bias", "compute_dtype", "scale", "block_scaled")

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        *,
        compute_dtype: torch.dtype = torch.bfloat16,
        scale: torch.Tensor | None = None,
        block_scaled: bool = False,
    ):
        self.weight = weight
        self.bias = bias
        self.compute_dtype = compute_dtype
        self.scale = scale
        self.block_scaled = block_scaled

    def materialize(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.weight.to(device, non_blocking=True)
        bias = self.bias.to(device, non_blocking=True)
        if self.scale is None:
            # bf16 path: the store bytes are the compute weights untouched, so
            # the projection is bitwise-identical to the resident reference.
            return weight, bias
        # Increment 4 hook (FP8): dequantize on the compute device. Only this
        # branch changes; the forward pass below is dtype-agnostic.
        scale = self.scale.to(device, non_blocking=True)
        weight = (weight.to(torch.float32) * scale).to(self.compute_dtype)
        return weight, bias

    def materialize_fp8(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Block-scaled FP8 path: return (weight_fp8, block_scale, bias) on
        ``device`` WITHOUT dequantizing — the fused kernel dequantizes inside
        the GEMM. Transfers half the bytes of the bf16 weight (fp8 weight +
        tiny scale)."""
        assert self.scale is not None and self.block_scaled, (
            "materialize_fp8 requires a block-scaled fp8 entry"
        )
        return (
            self.weight.to(device, non_blocking=True),
            self.scale.to(device, non_blocking=True),
            self.bias.to(device, non_blocking=True),
        )


class HostResidentAdaLN(nn.Module):
    """Drop-in replacement for ``MiniMaxH3AdaLayerNormModulation``.

    Holds the projection weights host-resident (via :class:`AdaLNEntry`, a
    plain attribute invisible to ``Module.to``) and materializes them on the
    compute device per call. The forward pass reproduces the reference exactly:
    SiLU at ``temb``'s incoming precision, cast to the projection dtype, the
    linear projection, then the ``view(-1, 6 * hidden_size).chunk(6)`` split
    into ``shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp``.
    """

    def __init__(self, entry: AdaLNEntry, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Plain attribute — NOT register_parameter / register_buffer / a
        # submodule — so ``Module.to(device)`` and ``.to(dtype)`` skip it and
        # the weight stays on the host until forward materializes it.
        self._entry = entry

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        activated = F.silu(temb)  # at temb's incoming precision (ref-exact)
        if self._entry.scale is None:
            # bf16 path — byte-exact host-to-device copy, unchanged
            weight, bias = self._entry.materialize(temb.device)
            temb = F.linear(activated.to(weight.dtype), weight, bias)
        elif not self._entry.block_scaled:
            # legacy per-row fp8 A/B path — materialize() dequantizes per-row
            weight, bias = self._entry.materialize(temb.device)
            temb = F.linear(activated.to(weight.dtype), weight, bias)
        else:
            # block-scaled fp8 — fused weight-only GEMM (no bf16 materialize)
            from omni_infinity.kernels import fused_fp8_gemm
            weight_fp8, scale, bias = self._entry.materialize_fp8(temb.device)
            temb = fused_fp8_gemm(
                activated.to(torch.bfloat16), weight_fp8, scale, bias
            )
        temb = temb.view(-1, 6 * self.hidden_size)
        return temb.chunk(6, dim=-1)
