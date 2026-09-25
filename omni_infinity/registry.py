# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Category registry: model architectures x optimizations.

Two orthogonal category axes describe every supported configuration:

- **model-arch** -- which transformer/pipeline family runs: the dense
  H3-Base reference (``h3-dense``) or the VDN-Minimax-H3 hybrid-attention
  derivative (``vdn-hybrid``).
- **optimization** -- a memory/speed technique layered on top: AdaLN host
  cache, FP8 weights, transformer block streaming, text-encoder streaming.

The registry is data, not behavior. Each entry maps a stable name to the
runner keyword arguments that enable it and the archs it supports; the
ablation harness (``benchmarks/ablation_vdn.py``) iterates these entries
to build its sweep grid, and the smoke CLIs resolve flags through it.
"""

from __future__ import annotations

import dataclasses
import importlib
from types import MappingProxyType

CATEGORY_MODEL_ARCH = "model-arch"
CATEGORY_OPTIMIZATION = "optimization"


@dataclasses.dataclass(frozen=True)
class ArchSpec:
    """A model architecture: which runner loads which checkpoint."""

    name: str
    runner: str  # "module.path:ClassName"
    default_checkpoint: str
    description: str


@dataclasses.dataclass(frozen=True)
class OptimizationSpec:
    """An optimization: runner kwargs that switch it on, per arch."""

    name: str
    description: str
    supported_archs: tuple[str, ...]
    runner_kwargs: MappingProxyType


def _kw(**kwargs) -> MappingProxyType:
    return MappingProxyType(kwargs)


ARCHS = {
    "h3-dense": ArchSpec(
        name="h3-dense",
        runner="omni_infinity.runner:ReferenceRunner",
        default_checkpoint="MiniMaxAI/MiniMax-H3",
        description="Dense MiniMax-H3-Base via MiniMaxH3ModularPipeline.",
    ),
    "vdn-hybrid": ArchSpec(
        name="vdn-hybrid",
        runner="omni_infinity.arch.vdn:VdnRunner",
        default_checkpoint="OpenVDN/vdn-minimax-h3",
        description=(
            "VDN-Minimax-H3 hybrid attention (frame-wise linear branch "
            "+ window softmax + merged LoRAs) via diffusers remote code."
        ),
    ),
}

OPTIMIZATIONS = {
    "adaln-host-cache": OptimizationSpec(
        name="adaln-host-cache",
        description="Host-resident AdaLN branch cache from a moe-store.",
        supported_archs=("h3-dense",),
        runner_kwargs=_kw(adaln_host_cache=True),
    ),
    "fp8": OptimizationSpec(
        name="fp8",
        description=(
            "FP8 weights on wide Linears (h3-dense: block-scaled "
            "store path; vdn-hybrid: upstream torchao preset)."
        ),
        supported_archs=("h3-dense", "vdn-hybrid"),
        runner_kwargs=_kw(fp8=True),
    ),
    "block-stream": OptimizationSpec(
        name="block-stream",
        description=(
            "Stream transformer blocks through the GPU one group at "
            "a time (diffusers block_level group offload)."
        ),
        supported_archs=("h3-dense", "vdn-hybrid"),
        runner_kwargs=_kw(offload=True, block_stream_blocks_per_group=1),
    ),
    "text-encoder-stream": OptimizationSpec(
        name="text-encoder-stream",
        description=("Leaf-level streaming of the Qwen3-VL text encoder."),
        supported_archs=("h3-dense", "vdn-hybrid"),
        runner_kwargs=_kw(offload=True, stream_text_encoder=True),
    ),
}


def runner_class(spec: ArchSpec):
    module_name, _, class_name = spec.runner.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def runner_kwargs_for(arch: str, optimizations) -> dict:
    """Merge the runner kwargs of *optimizations* for *arch*.

    Raises ValueError for unknown names or unsupported (arch, opt)
    pairs so a sweep fails loudly instead of silently no-op'ing.
    """
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}")
    merged: dict = {}
    for name in optimizations:
        try:
            spec = OPTIMIZATIONS[name]
        except KeyError:
            raise ValueError(f"unknown optimization {name!r}") from None
        if arch not in spec.supported_archs:
            raise ValueError(
                f"optimization {name!r} does not support arch {arch!r}"
            )
        merged.update(spec.runner_kwargs)
    return merged
