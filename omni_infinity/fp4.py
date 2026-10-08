# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""MXFP4 (E2M1) weight storage for the H3 transformer's dense Linears.

P5 phase 1 (issue #42). Mirrors the FP8 path in ``omni_infinity/fp8.py``
one rung down the quality ladder: weights are stored as packed 4-bit
E2M1 pairs with per-32-element E8M0 scales and dequantized inside the
fused weight-only GEMM (bf16 MMA). Halves weight bytes vs FP8; it is an
accuracy-gated, opt-in memory/bandwidth tradeoff and never the default.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from omni_infinity.kernels import MXFP4_BLOCK, fused_mxfp4_gemm, quantize_mxfp4

_SKIP_PATTERNS = ("norm", "pos_embed", "patch_embed")

_BLOCK_RE = re.compile(r"transformer_blocks\.(\d+)\.")


class ScaledFp4Linear(nn.Module):
    """nn.Linear with a packed MXFP4 weight, fused-dequantized per call.

    ``mode='mxfp4'`` (the only mode) stores the weight as [N, K//2] uint8
    E2M1 pairs plus [N, K//32] uint8 E8M0 scales and runs the fused
    weight-only GEMM WITHOUT materializing the bf16 weight. Both ride
    ``.to(device)`` as buffers, so the layer stays ~quarter its bf16 size
    resident.
    """

    def __init__(
        self, linear: nn.Linear, compute_dtype: torch.dtype, mode: str = "mxfp4"
    ):
        super().__init__()
        assert mode == "mxfp4", mode
        self.mode = mode
        packed, scales = quantize_mxfp4(linear.weight.detach())
        self.register_buffer("weight_packed", packed)
        self.register_buffer("weight_scales", scales)
        self.bias = linear.bias
        self.compute_dtype = compute_dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return fused_mxfp4_gemm(
            hidden_states.to(torch.bfloat16),
            self.weight_packed,
            self.weight_scales,
            self.bias,
        )


def apply_scaled_fp4_casting(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    skip_blocks: frozenset[int] = frozenset(),
    mode: str = "mxfp4",
) -> int:
    skip = set(_SKIP_PATTERNS)
    skip.update(getattr(model, "_keep_in_fp32_modules", None) or ())
    skip.update(getattr(model, "_skip_layerwise_casting_patterns", None) or ())

    def _skipped(name: str) -> bool:
        if any(re.search(pattern, name) for pattern in skip):
            return True
        block = _BLOCK_RE.search(name)
        return block is not None and int(block.group(1)) in skip_blocks

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and not _skipped(name)
        and module.in_features % MXFP4_BLOCK == 0
    ]
    for name, module in targets:
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf, ScaledFp4Linear(module, compute_dtype, mode=mode))
    return len(targets)
