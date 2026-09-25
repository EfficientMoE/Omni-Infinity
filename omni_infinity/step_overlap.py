# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Cross-step adapter for Diffusers 0.40.x group-offloading hooks.

This module is intentionally the only Omni-Infinity code that imports the
private Diffusers symbols ``_GROUP_OFFLOADING``, ``GroupOffloadingHook``, and
``ModuleGroup``. Diffusers 0.40.x is pinned because the adapter depends on
``GroupOffloadingHook.next_group`` and ``ModuleGroup.onload_`` semantics.
Topology drift fails closed instead of silently reverting to synchronous H2D.
"""

from __future__ import annotations

import dataclasses

import torch

_OMNI_STEP_OVERLAP = "omni_step_overlap"


def _private_api():
    import diffusers
    from diffusers.hooks import HookRegistry, ModelHook
    from diffusers.hooks.group_offloading import (
        _GROUP_OFFLOADING,
        GroupOffloadingHook,
        ModuleGroup,
    )

    if not diffusers.__version__.startswith("0.40."):
        raise RuntimeError(
            "step overlap supports diffusers 0.40.x private hooks; "
            f"found {diffusers.__version__}"
        )
    return (
        HookRegistry,
        ModelHook,
        _GROUP_OFFLOADING,
        GroupOffloadingHook,
        ModuleGroup,
    )


def _discover_group_hooks(transformer):
    (
        HookRegistry,
        _ModelHook,
        group_offloading_key,
        GroupOffloadingHook,
        ModuleGroup,
    ) = _private_api()
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None or len(blocks) < 2:
        raise RuntimeError(
            "step overlap requires at least two transformer blocks"
        )

    hooks = []
    seen_groups = set()
    for index, block in enumerate(blocks):
        registry = HookRegistry.check_if_exists_or_initialize(block)
        hook = registry.get_hook(group_offloading_key)
        if not isinstance(hook, GroupOffloadingHook):
            raise RuntimeError(
                f"transformer_blocks[{index}] has no Diffusers group hook"
            )
        if not isinstance(hook.group, ModuleGroup):
            raise RuntimeError(
                f"transformer_blocks[{index}] has an invalid ModuleGroup"
            )
        if id(hook.group) in seen_groups:
            raise RuntimeError(
                "step overlap requires one unique group per transformer block"
            )
        seen_groups.add(id(hook.group))
        if hook.group.modules != [block]:
            raise RuntimeError(
                "transformer block groups are not in sequential one-block order"
            )
        stream = hook.group.stream
        if stream is None or not isinstance(stream, torch.cuda.Stream):
            raise RuntimeError(
                f"transformer_blocks[{index}] does not use a CUDA copy stream"
            )
        hooks.append(hook)
    return hooks


def _top_level_group_hook(transformer):
    (
        HookRegistry,
        _ModelHook,
        group_offloading_key,
        GroupOffloadingHook,
        _ModuleGroup,
    ) = _private_api()
    registry = HookRegistry.check_if_exists_or_initialize(transformer)
    hook = registry.get_hook(group_offloading_key)
    if not isinstance(hook, GroupOffloadingHook):
        raise RuntimeError("transformer has no top-level Diffusers group hook")
    return registry, hook


@dataclasses.dataclass
class StepOverlapController:
    transformer: object
    registry: object
    first_group: object
    last_hook: object
    top_hook: object
    original_first_onload_self: bool
    original_first_non_blocking: bool
    original_last_next_group: object
    original_top_next_group: object
    _closed: bool = False

    def _reassert(self) -> None:
        self.top_hook.next_group = None
        self.first_group.onload_self = False
        self.first_group.non_blocking = True
        self.last_hook.next_group = self.first_group

    def close(self) -> None:
        if self._closed:
            return
        self.registry.remove_hook(_OMNI_STEP_OVERLAP, recurse=False)
        self.last_hook.next_group = self.original_last_next_group
        self.first_group.onload_self = self.original_first_onload_self
        self.first_group.non_blocking = self.original_first_non_blocking
        self.top_hook.next_group = self.original_top_next_group
        self._closed = True


def enable_step_overlap(transformer) -> StepOverlapController:
    """Close Diffusers' native group chain from the last block to the first."""

    HookRegistry, ModelHook, *_private_symbols = _private_api()
    block_hooks = _discover_group_hooks(transformer)
    registry, top_hook = _top_level_group_hook(transformer)
    if registry.get_hook(_OMNI_STEP_OVERLAP) is not None:
        raise RuntimeError("step overlap is already enabled")

    first_group = block_hooks[0].group
    last_hook = block_hooks[-1]
    controller = StepOverlapController(
        transformer=transformer,
        registry=registry,
        first_group=first_group,
        last_hook=last_hook,
        top_hook=top_hook,
        original_first_onload_self=first_group.onload_self,
        original_first_non_blocking=first_group.non_blocking,
        original_last_next_group=last_hook.next_group,
        original_top_next_group=top_hook.next_group,
    )

    class _ReassertStepOverlapHook(ModelHook):
        def post_forward(self, module, output):
            controller._reassert()
            return output

    first_group.onload_self = False
    first_group.non_blocking = True
    last_hook.next_group = first_group
    registry.register_hook(_ReassertStepOverlapHook(), _OMNI_STEP_OVERLAP)
    return controller
