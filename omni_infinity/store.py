# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Component weights from a moe-store v2 store with one-read group fetches.

Task 2 increment 1 (issue #2): each moe-store group (a transformer block's
non-AdaLN weights, an AdaLN branch bundle, or a whole VAE/encoder shard) is
fetched with a single contiguous read at ``(offset, total_size)`` and sliced
into member tensors in memory — the store-side contract that component
streaming and the AdaLN cache build on. Tensor names are returned without
the ``<component>.`` prefix so they load directly into the diffusers
component's ``state_dict`` namespace.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

_BLOCK_INDEX_RE = re.compile(r"transformer_blocks\.(\d+)\.")


def _data_file(store_dir: Path, file_id: int) -> Path:
    try:
        from moe_store.index import data_file_name

        return store_dir / data_file_name(file_id)
    except ImportError:
        return store_dir / f"store_data_{file_id}"


def _block_index(tensor_name: str) -> int:
    match = _BLOCK_INDEX_RE.search(tensor_name)
    if match is None:
        raise ValueError(
            f"no transformer block index in tensor name {tensor_name!r}"
        )
    return int(match.group(1))


class StoreComponentSource:
    def __init__(self, store_dir: str | Path):
        from moe_store.index import read_index

        self.store_dir = Path(store_dir)
        self.index = read_index(str(store_dir))
        self._stage_names = [stage.name for stage in self.index.stages]

    def stage_name(self, group) -> str:
        return self._stage_names[group.group_id & 0xFFFFFFFF]

    def components(self) -> list[str]:
        seen: dict[str, None] = {}
        for name in self._stage_names:
            seen.setdefault(name.split(".", 1)[0])
        return list(seen)

    def groups_for(self, component: str) -> list:
        prefix = f"{component}."
        return [
            group
            for group in self.index.groups
            if self.stage_name(group) == component
            or self.stage_name(group).startswith(prefix)
        ]

    def adaln_groups(self, component: str = "transformer") -> list:
        return [
            group
            for group in self.groups_for(component)
            if self.stage_name(group).endswith(".adaln")
        ]

    def read_adaln_cache(
        self,
        component: str = "transformer",
        *,
        fp8: bool = False,
        fp8_mode: str = "block",
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        from omni_infinity.adaln import AdaLNEntry
        from omni_infinity.fp8 import quantize_per_row_fp8
        from omni_infinity.kernels import quantize_block_fp8

        cache: dict[int, AdaLNEntry] = {}
        for group in self.adaln_groups(component):
            weight = bias = block = None
            for name, tensor in self.read_group(group).items():
                block = _block_index(name)
                if name.endswith(".weight"):
                    weight = tensor
                elif name.endswith(".bias"):
                    bias = tensor
            if block is None or weight is None or bias is None:
                raise ValueError(
                    f"adaln group {group.group_id:#x} is not a "
                    "{weight, bias} projection bundle"
                )
            if fp8:
                if fp8_mode == "block":
                    quantized, scale = quantize_block_fp8(weight)
                    cache[block] = AdaLNEntry(
                        quantized,
                        bias,
                        compute_dtype=compute_dtype,
                        scale=scale,
                        block_scaled=True,
                    )
                else:
                    quantized, scale = quantize_per_row_fp8(weight)
                    cache[block] = AdaLNEntry(
                        quantized,
                        bias,
                        compute_dtype=compute_dtype,
                        scale=scale,
                        block_scaled=False,
                    )
            else:
                cache[block] = AdaLNEntry(weight, bias)
        return cache

    def read_group(self, group) -> dict[str, torch.Tensor]:
        path = _data_file(self.store_dir, group.file_id)
        with open(path, "rb") as handle:
            handle.seek(group.offset)
            payload = handle.read(group.total_size)
        tensors: dict[str, torch.Tensor] = {}
        for member in group.members:
            raw = bytearray(
                payload[member.rel_offset : member.rel_offset + member.size]
            )
            tensor = torch.frombuffer(
                raw, dtype=getattr(torch, member.dtype)
            ).reshape(tuple(member.shape))
            name = member.name.split(".", 1)[1]
            tensors[name] = tensor
        return tensors

    def load_component_state_dict(
        self, component: str, *, exclude_adaln: bool = False
    ) -> dict[str, torch.Tensor]:
        groups = self.groups_for(component)
        if exclude_adaln:
            groups = [
                group
                for group in groups
                if not self.stage_name(group).endswith(".adaln")
            ]
        state: dict[str, torch.Tensor] = {}
        for group in groups:
            state.update(self.read_group(group))
        return state


def load_diffusers_component(
    component_cls,
    checkpoint: str,
    component: str,
    source: StoreComponentSource,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    config = component_cls.load_config(checkpoint, subfolder=component)
    model = component_cls.from_config(config)
    state = source.load_component_state_dict(component)
    _load_state_dict_strict_unexpected(model, state, component)
    model = _cast_preserving_fp32(model, component_cls, torch_dtype)
    return model.eval()


def load_transformer_with_adaln_cache(
    component_cls,
    checkpoint: str,
    source: StoreComponentSource,
    torch_dtype: torch.dtype = torch.bfloat16,
    component: str = "transformer",
    *,
    fp8: bool = False,
    fp8_skip_last_blocks: int = 0,
    adaln_fp8: bool = False,
    fp8_mode: str = "block",
):
    from omni_infinity.adaln import HostResidentAdaLN

    config = component_cls.load_config(checkpoint, subfolder=component)
    model = component_cls.from_config(config)
    # Swap the AdaLN projections for host-resident modules before the
    # state-dict load and the compute-dtype cast: the ~26 GB of AdaLN weights
    # never become GPU-movable Parameters (so pipeline.to(device) leaves them
    # on the host) and the discarded fp32 init Linears are freed here rather
    # than surviving the load.
    cache = source.read_adaln_cache(
        component, fp8=adaln_fp8, fp8_mode=fp8_mode, compute_dtype=torch_dtype
    )
    for index, block in enumerate(model.transformer_blocks):
        entry = cache.get(index)
        if entry is None:
            raise ValueError(f"{component}: no AdaLN group for block {index}")
        block.adaln_proj = HostResidentAdaLN(
            entry, block.adaln_proj.hidden_size
        )
    state = source.load_component_state_dict(component, exclude_adaln=True)
    _load_state_dict_strict_unexpected(model, state, component)
    model = _cast_preserving_fp32(model, component_cls, torch_dtype)
    if fp8:
        # Non-adaln FP8: store the attn/ff Linears as per-row-scaled float8 and
        # upcast per forward (scaled, not diffusers' unscaled layerwise which
        # is ~17-35% off on H3's ~1e-2 weights). FP8 error compounds across the
        # 50 blocks, so fp8_skip_last_blocks keeps the last N blocks bf16 to cap
        # the output deviation. Applied on the CPU model so weights are float8
        # before .to(device) -- no 40 GB bf16 spike.
        from omni_infinity.fp8 import apply_scaled_fp8_casting

        total_blocks = len(model.transformer_blocks)
        skip_blocks = (
            frozenset(range(total_blocks - fp8_skip_last_blocks, total_blocks))
            if fp8_skip_last_blocks > 0
            else frozenset()
        )
        apply_scaled_fp8_casting(
            model,
            compute_dtype=torch_dtype,
            skip_blocks=skip_blocks,
            mode=fp8_mode,
        )
    return model.eval()


def _load_state_dict_strict_unexpected(model, state, component: str) -> None:
    _missing, unexpected = model.load_state_dict(
        state, strict=False, assign=True
    )
    if unexpected:
        raise ValueError(
            f"{component}: store tensors not accepted by the model: "
            f"{sorted(unexpected)[:5]}"
        )


def _cast_preserving_fp32(model, component_cls, torch_dtype: torch.dtype):
    fp32_modules = getattr(component_cls, "_keep_in_fp32_modules", None) or []

    def _keep_fp32(tensor_name: str) -> bool:
        return any(pattern in tensor_name for pattern in fp32_modules)

    # Match diffusers' mixed-precision load: _keep_in_fp32_modules stay fp32
    # (params from the dtype-preserving store AND init-computed buffers such
    # as rope.inv_freq), everything else is the compute dtype. Snapshot the
    # fp32 tensors before the blanket cast so a bf16 round-trip cannot
    # irrecoverably truncate them.
    named = {
        **dict(model.named_parameters()),
        **dict(model.named_buffers()),
    }
    preserved = {
        name: tensor.clone()
        for name, tensor in named.items()
        if _keep_fp32(name)
    }
    model = model.to(torch_dtype)
    for name, tensor in preserved.items():
        _assign_param(model, name, tensor)
    return model


def _assign_param(model, dotted_name: str, tensor: torch.Tensor) -> None:
    module_path, _, leaf = dotted_name.rpartition(".")
    module = model.get_submodule(module_path)
    if leaf in dict(module.named_parameters(recurse=False)):
        module._parameters[leaf] = torch.nn.Parameter(
            tensor, requires_grad=False
        )
    elif leaf in dict(module.named_buffers(recurse=False)):
        module._buffers[leaf] = tensor
