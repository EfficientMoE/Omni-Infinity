# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Per-output-channel scaled FP8 storage for the H3 transformer's dense Linears.

Task 2 increment 4 (issue #2). diffusers' ``enable_layerwise_casting`` stores
float8 *unscaled*; on H3's ~1e-2 attn/ff weights that quantizes far too
coarsely (measured ~17-35% output error, well past the rtol=2e-2 gate). This
module instead stores each eligible ``nn.Linear`` weight as ``float8_e4m3fn``
with a per-output-channel (row) scale and upcasts ``w = fp8 * scale`` to the
compute dtype inside forward -- the same scaled scheme ``AdaLNEntry`` uses --
keeping the transformer ~20 GB resident at rtol=2e-2 accuracy. Weights that
must stay high precision (``_keep_in_fp32_modules``, norms, patch/pos embeds)
are matched by name and left untouched, mirroring diffusers' skip set.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from omni_infinity.kernels import fused_fp8_gemm, quantize_block_fp8

_SKIP_PATTERNS = ("norm", "pos_embed", "patch_embed")


def quantize_per_row_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    w = weight.to(torch.float32)
    # One scale per output channel (row): keeps a row's small weights off the
    # coarse end of the float8 grid instead of letting a global max crush them.
    scale = (w.abs().amax(dim=1, keepdim=True) / fp8_max).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    quantized = (w / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return quantized, scale


class ScaledFp8Linear(nn.Module):
    """nn.Linear with a float8 weight, fused-dequantized per call.

    ``mode='block'`` (default) stores a block-wise (128x128) scale and runs the
    fused weight-only GEMM WITHOUT materializing the bf16 weight (activations
    stay bf16 -> accuracy-preserving vs W8A8). ``mode='per_row'`` keeps the
    legacy per-output-channel scale + dequant-materialize ``F.linear`` path,
    retained for the inc-8 accuracy A/B. Both store the float8 weight + scale as
    buffers (they ride ``.to(device)`` without a dtype change), so the layer
    stays ~half its bf16 size resident.
    """

    def __init__(
        self, linear: nn.Linear, compute_dtype: torch.dtype, mode: str = "block"
    ):
        super().__init__()
        assert mode in ("block", "per_row"), mode
        self.mode = mode
        if mode == "block":
            quantized, scale = quantize_block_fp8(linear.weight.detach())
        else:
            quantized, scale = quantize_per_row_fp8(linear.weight.detach())
        self.register_buffer("weight_fp8", quantized)
        self.register_buffer("weight_scale", scale)
        self.bias = linear.bias
        self.compute_dtype = compute_dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.mode == "block":
            return fused_fp8_gemm(
                hidden_states.to(torch.bfloat16),
                self.weight_fp8,
                self.weight_scale,
                self.bias,
            )
        weight = (self.weight_fp8.to(torch.float32) * self.weight_scale).to(
            self.compute_dtype
        )
        return F.linear(hidden_states, weight, self.bias)


_BLOCK_RE = re.compile(r"transformer_blocks\.(\d+)\.")


def apply_scaled_fp8_casting(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    skip_blocks: frozenset[int] = frozenset(),
    mode: str = "block",
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
        if isinstance(module, nn.Linear) and not _skipped(name)
    ]
    for name, module in targets:
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf, ScaledFp8Linear(module, compute_dtype, mode=mode))
    return len(targets)
