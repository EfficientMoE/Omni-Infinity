# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import pytest

from omni_infinity.caches.denoise import DenoiseCacheConfig


def test_config_requires_calibrated_coefficients():
    with pytest.raises(ValueError, match="MiniMax-H3"):
        DenoiseCacheConfig(coefficients=(), threshold=0.2)


@pytest.mark.parametrize("threshold", [0, -0.1])
def test_config_requires_positive_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        DenoiseCacheConfig(coefficients=(1.0, 0.0), threshold=threshold)


def test_config_rejects_unsupported_mode():
    with pytest.raises(ValueError, match="output"):
        DenoiseCacheConfig(
            coefficients=(1.0, 0.0), threshold=0.2, mode="kv"
        )


@pytest.mark.parametrize("field", ["warmup_steps", "final_steps"])
def test_config_requires_boundary_steps(field):
    kwargs = {field: 0}
    with pytest.raises(ValueError, match="first and last"):
        DenoiseCacheConfig(
            coefficients=(1.0, 0.0), threshold=0.2, **kwargs
        )


def test_config_requires_positive_calls_per_step():
    with pytest.raises(ValueError):
        DenoiseCacheConfig(
            coefficients=(1.0, 0.0), threshold=0.2, calls_per_step=0
        )


def test_config_is_constructible_and_frozen():
    config = DenoiseCacheConfig(coefficients=(1.0, 0.0), threshold=0.2)

    assert config.coefficients == (1.0, 0.0)
    assert config.threshold == 0.2
    with pytest.raises(FrozenInstanceError):
        config.threshold = 0.3
