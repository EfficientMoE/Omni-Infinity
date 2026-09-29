# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Issue #24 C5 — denoise-step feature cache (TeaCache-style, opt-in).

Approximate by construction, so the contract under test is the decision
machinery and its guardrails: no default coefficients exist for
MiniMax-H3 (enabling without a calibration must refuse loudly), the
first and last steps always compute, skipped steps replay the cached
result, and the wrapper leaves no trace on the transformer afterwards.
The parity gates never enable this cache.
"""

import sys
import types
from collections import OrderedDict

import pytest
import torch

from omni_infinity.caches.denoise import (
    DenoiseCacheConfig,
    denoise_step_cache,
    enable_cache_dit,
)
from omni_infinity.runner import GenerationResult, ReferenceRunner

# Identity rescale (poly1d order: highest degree first) and a threshold
# that permits skipping while inputs stay identical.
IDENTITY = DenoiseCacheConfig(coefficients=(1.0, 0.0), threshold=0.5)


class ResidualModel(torch.nn.Module):
    """Output shape == input shape, so residual mode applies."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return hidden_states * 2.0


class H3LikeModel(torch.nn.Module):
    """MiniMax-H3 contract: per-modality projections whose shapes do NOT
    match the packed input rows — only mode='output' applies."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, audio_hidden_states, timestep):
        self.calls += 1
        return OrderedDict(
            sample=hidden_states[:, :2] + timestep,
            audio_sample=audio_hidden_states[:, :1] + timestep,
        )


def test_uncalibrated_config_refuses():
    # No published TeaCache/cache-dit table covers H3: an empty
    # calibration must fail at construction, not silently no-op.
    with pytest.raises(ValueError, match="calibrated"):
        DenoiseCacheConfig(coefficients=(), threshold=0.1)


def test_invalid_parameters_refuse():
    with pytest.raises(ValueError, match="threshold"):
        DenoiseCacheConfig(coefficients=(1.0,), threshold=0.0)
    with pytest.raises(ValueError, match="mode"):
        DenoiseCacheConfig(coefficients=(1.0,), threshold=0.1, mode="prefix")
    with pytest.raises(ValueError, match="always compute"):
        DenoiseCacheConfig(coefficients=(1.0,), threshold=0.1, warmup_steps=0)


def test_first_and_last_steps_always_compute():
    model = ResidualModel()
    x = torch.ones(2, 3)
    with denoise_step_cache(model, IDENTITY, total_steps=4) as stats:
        for _ in range(4):
            model(hidden_states=x)
    # Steps 0 and 3 compute; identical inputs let steps 1-2 skip.
    assert model.calls == 2
    assert stats.computed == 2 and stats.skipped == 2


def test_output_mode_replays_the_previous_prediction():
    model = ResidualModel()
    x = torch.ones(2, 3)
    with denoise_step_cache(model, IDENTITY, total_steps=3):
        first = model(hidden_states=x)
        second = model(hidden_states=x)  # skipped
        third = model(hidden_states=x)  # final: dense
    assert model.calls == 2
    assert torch.equal(second, first)
    assert torch.equal(third, x * 2.0)


def test_residual_mode_applies_the_cached_residual():
    config = DenoiseCacheConfig(
        coefficients=(1.0, 0.0), threshold=0.5, mode="residual"
    )
    model = ResidualModel()
    with denoise_step_cache(model, config, total_steps=3):
        model(hidden_states=torch.ones(2, 3))
        # Residual from step 0 is +1s; the (slightly moved) input gets
        # it added instead of a fresh forward.
        drifted = torch.full((2, 3), 1.01)
        replay = model(hidden_states=drifted)
        model(hidden_states=drifted)
    assert model.calls == 2
    assert torch.allclose(replay, drifted + torch.ones(2, 3))


def test_accumulated_distance_forces_a_compute():
    # poly = identity; step-to-step distance 0.2 with threshold 0.3:
    # one skip is allowed, the accumulated second exceeds it.
    config = DenoiseCacheConfig(coefficients=(1.0, 0.0), threshold=0.3)
    model = ResidualModel()
    with denoise_step_cache(model, config, total_steps=5) as stats:
        signal = torch.ones(4)
        for _ in range(5):
            model(hidden_states=signal)
            signal = signal * 1.2
    # Steps 0 and 4 are boundary computes; among steps 1-3 the
    # accumulator (~0.2, then ~0.4) forces a compute at step 2.
    assert stats.computed == 3 and stats.skipped == 2


def test_polynomial_rescale_is_applied():
    # A polynomial that maps any distance above the threshold disables
    # skipping entirely, even for tiny raw distances.
    config = DenoiseCacheConfig(coefficients=(0.0, 99.0), threshold=0.5)
    model = ResidualModel()
    with denoise_step_cache(model, config, total_steps=4) as stats:
        x = torch.ones(2)
        for _ in range(4):
            model(hidden_states=x)
            x = x * 1.001
    assert stats.skipped == 0 and model.calls == 4


def test_h3_like_output_mode_round_trip():
    config = DenoiseCacheConfig(
        coefficients=(1.0, 0.0),
        threshold=0.5,
        mode="output",
        signal_name="hidden_states",
    )
    model = H3LikeModel()
    video = torch.ones(1, 4)
    audio = torch.ones(1, 2)
    with denoise_step_cache(model, config, total_steps=3) as stats:
        first = model(
            hidden_states=video,
            audio_hidden_states=audio,
            timestep=torch.tensor(1.0),
        )
        second = model(
            hidden_states=video,
            audio_hidden_states=audio,
            timestep=torch.tensor(0.5),
        )
        model(
            hidden_states=video,
            audio_hidden_states=audio,
            timestep=torch.tensor(0.1),
        )
    assert model.calls == 2 and stats.skipped == 1
    assert torch.equal(second["sample"], first["sample"])
    assert torch.equal(second["audio_sample"], first["audio_sample"])


def test_residual_mode_rejects_h3_shaped_outputs():
    config = DenoiseCacheConfig(
        coefficients=(1.0, 0.0),
        threshold=0.5,
        mode="residual",
        io_names=("hidden_states", "audio_hidden_states"),
    )
    model = H3LikeModel()
    with denoise_step_cache(model, config, total_steps=3):
        with pytest.raises(RuntimeError, match="mode='output'"):
            model(
                hidden_states=torch.ones(1, 4),
                audio_hidden_states=torch.ones(1, 2),
                timestep=torch.tensor(1.0),
            )


def test_cfg_slots_are_independent():
    # SGLang keeps separate positive/negative residual slots; with
    # calls_per_step=2 each intra-step call tracks its own signal.
    config = DenoiseCacheConfig(
        coefficients=(1.0, 0.0), threshold=0.5, calls_per_step=2
    )
    model = ResidualModel()
    positive, negative = torch.ones(2), torch.zeros(2) + 5.0
    with denoise_step_cache(model, config, total_steps=3) as stats:
        for _ in range(3):
            model(hidden_states=positive)
            model(hidden_states=negative)
    # Step 0 and 2 compute both branches; step 1 skips both.
    assert stats.computed == 4 and stats.skipped == 2


def test_positional_signal_resolution():
    model = ResidualModel()
    x = torch.ones(3)
    with denoise_step_cache(model, IDENTITY, total_steps=3) as stats:
        model(x)
        model(x)
        model(x)
    assert stats.skipped == 1


def test_forward_is_restored_and_reentry_refused():
    model = ResidualModel()
    with denoise_step_cache(model, IDENTITY, total_steps=2):
        with pytest.raises(RuntimeError, match="already enabled"):
            with denoise_step_cache(model, IDENTITY, total_steps=2):
                pass
    assert "forward" not in model.__dict__
    model(hidden_states=torch.ones(1))
    model(hidden_states=torch.ones(1))
    assert model.calls == 2


def test_progress_hooks_still_fire_on_skipped_steps():
    # The job server counts transformer forwards via Module hooks; a
    # skipped step must still report progress.
    model = ResidualModel()
    fired = []
    handle = model.register_forward_hook(
        lambda module, args, output: fired.append(1)
    )
    try:
        with denoise_step_cache(model, IDENTITY, total_steps=3):
            x = torch.ones(2)
            for _ in range(3):
                model(hidden_states=x)
    finally:
        handle.remove()
    assert model.calls == 2 and len(fired) == 3


def test_runner_generate_threads_the_config():
    class FakeTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return hidden_states * 2.0

    class FakePipeline:
        def __init__(self):
            self.transformer = FakeTransformer()

        def __call__(self, **kwargs):
            for _ in range(kwargs["num_inference_steps"]):
                self.transformer(hidden_states=torch.ones(2))
            return type("State", (), {"values": {}})()

    pipeline = FakePipeline()
    result = ReferenceRunner(pipeline).generate(
        "prompt", num_inference_steps=4, denoise_cache=IDENTITY
    )
    assert isinstance(result, GenerationResult)
    assert pipeline.transformer.calls == 2
    # And the wrapper is gone after the generation.
    assert "forward" not in pipeline.transformer.__dict__


def test_enable_cache_dit_delegates_to_the_library(monkeypatch):
    recorded = {}
    fake = types.ModuleType("cache_dit")

    def enable_cache(target, **kwargs):
        recorded["target"] = target
        recorded["kwargs"] = kwargs
        return "adapter"

    fake.enable_cache = enable_cache
    monkeypatch.setitem(sys.modules, "cache_dit", fake)
    assert enable_cache_dit("transformer", Fn=8) == "adapter"
    assert recorded == {"target": "transformer", "kwargs": {"Fn": 8}}


def test_enable_cache_dit_reports_missing_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "cache_dit", None)
    with pytest.raises(ImportError, match="cache-dit"):
        enable_cache_dit("transformer")
