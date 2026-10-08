# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Split-stage execution over the diffusers modular block groups.

The MiniMaxH3 modular pipeline is a ``SequentialPipelineBlocks`` of five
top-level steps (``before_encode``, ``text_encoder``, ``vae_encoder``,
``denoise``, ``decode``).  Role-split serving runs a contiguous subset
of those steps and hands the ``PipelineState`` to the next role, so a
same-device split run is data-movement-only relative to the one-shot
``pipeline(**kwargs)`` call (bitwise-identical latents).
"""

from __future__ import annotations

from typing import Any

import torch

from omni_infinity.serve.roles import Role

STAGE_BLOCKS: dict[Role, tuple[str, ...]] = {
    Role.ENCODER: ("before_encode", "text_encoder", "vae_encoder"),
    Role.DENOISER: ("denoise",),
    Role.DECODER: ("decode",),
}

STAGE_ORDER = (Role.ENCODER, Role.DENOISER, Role.DECODER)


def role_for_block(name: str) -> Role:
    """Map a top-level block key to its role.

    Workflow-pruned pipelines (``from_pretrained(workflow=...)``) carry
    flattened keys like ``denoise.prepare_latents`` or ``decode.video``
    (``get_workflow`` resolves conditional blocks via
    ``get_execution_blocks``); unpruned pipelines carry the plain group
    names.  Both spellings map by prefix.
    """

    for role, prefixes in STAGE_BLOCKS.items():
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                return role
    raise ValueError(f"block {name!r} does not belong to any stage role")


def _blocks_of(pipeline: Any) -> Any:
    blocks = getattr(pipeline, "_blocks", None)
    if blocks is None:
        blocks = getattr(pipeline, "blocks", None)
    if blocks is None or not hasattr(blocks, "sub_blocks"):
        raise TypeError(
            "pipeline does not expose modular sub-blocks; split-stage "
            "execution requires a diffusers ModularPipeline"
        )
    return blocks


def prepare_state(pipeline: Any, **kwargs: Any) -> Any:
    """Build the initial ``PipelineState`` exactly as ``__call__`` does.

    Mirrors the kwargs/default population in
    ``ModularPipeline.__call__`` (diffusers modular_pipeline.py) without
    running any block, so the encoder role can start from the same
    state a one-shot call would see.
    """

    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    blocks = _blocks_of(pipeline)
    state = PipelineState()
    passed = dict(kwargs)
    for param in blocks.inputs:
        name = param.name
        kwargs_type = param.kwargs_type
        if name in passed:
            state.set(name, passed.pop(name), kwargs_type)
        elif kwargs_type is not None and kwargs_type in passed:
            for key, value in passed.pop(kwargs_type).items():
                state.set(key, value, kwargs_type)
        elif name is not None and name not in state.values:
            state.set(name, param.default, kwargs_type)
    if passed:
        raise ValueError(f"unexpected pipeline inputs: {sorted(passed.keys())}")
    return state


def run_stage(pipeline: Any, role: Role, state: Any) -> Any:
    """Run one role's block subset in place and return the state."""

    if role not in STAGE_BLOCKS:
        raise ValueError(f"role {role.value!r} has no stage blocks")
    blocks = _blocks_of(pipeline)
    names = [name for name in blocks.sub_blocks if role_for_block(name) is role]
    if not names:
        raise ValueError(
            f"pipeline has no blocks for role {role.value!r}; "
            f"available: {list(blocks.sub_blocks)}"
        )
    with torch.no_grad():
        for name in names:
            block = blocks.sub_blocks[name]
            pipeline, state = block(pipeline, state)
    return state


def run_all_stages(pipeline: Any, **kwargs: Any) -> Any:
    """Split-run every stage sequentially (parity path for role=all)."""

    state = prepare_state(pipeline, **kwargs)
    for role in STAGE_ORDER:
        state = run_stage(pipeline, role, state)
    return state
