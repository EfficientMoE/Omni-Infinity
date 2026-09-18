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

from pathlib import Path

import torch


def _data_file(store_dir: Path, file_id: int) -> Path:
    try:
        from moe_store.index import data_file_name

        return store_dir / data_file_name(file_id)
    except ImportError:
        return store_dir / f"store_data_{file_id}"


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
        self, component: str
    ) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for group in self.groups_for(component):
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
    missing, unexpected = model.load_state_dict(
        state, strict=False, assign=True
    )
    if unexpected:
        raise ValueError(
            f"{component}: store tensors not accepted by the model: "
            f"{sorted(unexpected)[:5]}"
        )
    return model.to(torch_dtype).eval()
