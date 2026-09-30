# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Generation-local denoise-step caching."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenoiseCacheConfig:
    """Configuration requiring model-specific calibrated coefficients."""

    coefficients: tuple[float, ...]
    threshold: float
    mode: str = "output"
    signal_name: str = "hidden_states"
    io_names: tuple[str, ...] = ("hidden_states",)
    calls_per_step: int = 1
    warmup_steps: int = 1
    final_steps: int = 1

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError(
                "calibrated coefficients are required for MiniMax-H3"
            )
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.mode not in {"output", "residual"}:
            raise ValueError("mode must be output or residual")
        if self.warmup_steps < 1 or self.final_steps < 1:
            raise ValueError("the first and last steps must always compute")
        if self.calls_per_step < 1:
            raise ValueError("calls_per_step must be positive")
