# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import inspect
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

import omni_infinity.caches.denoise as denoise_module
from omni_infinity import registry
from omni_infinity.caches.denoise import (
    DenoiseCacheConfig,
    denoise_step_cache,
    enable_cache_dit,
)


def test_config_requires_calibrated_coefficients():
    with pytest.raises(ValueError, match="MiniMax-H3"):
        DenoiseCacheConfig(coefficients=(), threshold=0.2)


@pytest.mark.parametrize("threshold", [0, -0.1])
def test_config_requires_positive_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        DenoiseCacheConfig(coefficients=(1.0, 0.0), threshold=threshold)


@pytest.mark.parametrize(
    ("coefficients", "threshold"),
    [
        ((float("nan"), 0.0), 0.2),
        ((float("inf"), 0.0), 0.2),
        ((1.0, 0.0), float("nan")),
        ((1.0, 0.0), float("inf")),
    ],
)
def test_config_rejects_non_finite_calibration(coefficients, threshold):
    with pytest.raises(ValueError, match="finite"):
        DenoiseCacheConfig(coefficients=coefficients, threshold=threshold)


def test_config_rejects_unsupported_mode():
    with pytest.raises(ValueError, match="output"):
        DenoiseCacheConfig(coefficients=(1.0, 0.0), threshold=0.2, mode="kv")


def test_config_rejects_unsupported_indicator():
    with pytest.raises(ValueError, match="indicator"):
        DenoiseCacheConfig(
            coefficients=(1.0, 0.0), threshold=0.2, indicator="unknown"
        )


def test_config_rejects_unsupported_approximator():
    with pytest.raises(ValueError, match="approximator"):
        DenoiseCacheConfig(
            coefficients=(1.0, 0.0),
            threshold=0.2,
            approximator="quadratic",
        )


@pytest.mark.parametrize("field", ["warmup_steps", "final_steps"])
def test_config_requires_boundary_steps(field):
    kwargs = {field: 0}
    with pytest.raises(ValueError, match="first and last"):
        DenoiseCacheConfig(coefficients=(1.0, 0.0), threshold=0.2, **kwargs)


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


class _AddOne(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return hidden_states + 1


def _config(**kwargs):
    values = {"coefficients": (1.0, 0.0), "threshold": 0.2}
    values.update(kwargs)
    return DenoiseCacheConfig(**values)


def test_output_mode_skips_middle_steps():
    module = _AddOne()

    with denoise_step_cache(module, _config(), total_steps=4) as stats:
        outputs = [module(torch.ones(4)) for _ in range(4)]

    assert module.calls == 2
    assert stats.computed == 2
    assert stats.skipped == 2
    assert torch.equal(outputs[1], outputs[0])


class _ScriptedOutput(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = iter(outputs)
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return next(self.outputs).clone()


def test_taylor1_output_extrapolates_from_two_computed_values():
    module = _ScriptedOutput(
        [torch.tensor([10.0]), torch.tensor([14.0]), torch.tensor([99.0])]
    )
    config = _config(approximator="taylor1", warmup_steps=2)

    with denoise_step_cache(module, config, total_steps=5):
        outputs = [module(torch.ones(1)) for _ in range(4)]

    assert module.calls == 2
    torch.testing.assert_close(outputs[2], torch.tensor([18.0]))
    torch.testing.assert_close(outputs[3], torch.tensor([22.0]))


def test_taylor1_first_skip_falls_back_to_reuse():
    module = _ScriptedOutput([torch.tensor([10.0]), torch.tensor([99.0])])
    config = _config(approximator="taylor1")

    with denoise_step_cache(module, config, total_steps=3):
        first = module(torch.ones(1))
        skipped = module(torch.ones(1))

    torch.testing.assert_close(skipped, first)


class _ScriptedResidual(torch.nn.Module):
    def __init__(self, residuals):
        super().__init__()
        self.residuals = iter(residuals)

    def forward(self, hidden_states):
        return hidden_states + next(self.residuals)


def test_taylor1_residual_extrapolates_against_current_input():
    module = _ScriptedResidual(
        [torch.tensor([2.0]), torch.tensor([4.0]), torch.tensor([99.0])]
    )
    config = _config(
        mode="residual",
        approximator="taylor1",
        warmup_steps=2,
        threshold=1.0,
    )

    with denoise_step_cache(module, config, total_steps=4):
        module(torch.tensor([1.0]))
        module(torch.tensor([2.0]))
        skipped = module(torch.tensor([3.0]))

    torch.testing.assert_close(skipped, torch.tensor([9.0]))


def test_taylor1_shape_change_resets_derivative_and_reuses_latest_value():
    module = _ScriptedOutput(
        [torch.tensor([1.0]), torch.tensor([2.0, 3.0]), torch.tensor([99.0])]
    )
    config = _config(approximator="taylor1", warmup_steps=2)

    with denoise_step_cache(module, config, total_steps=4):
        module(torch.ones(1))
        latest = module(torch.ones(1))
        skipped = module(torch.ones(1))

    torch.testing.assert_close(skipped, latest)


def test_taylor1_calls_per_step_keeps_derivatives_slot_local():
    module = _ScriptedOutput(
        [
            torch.tensor([10.0]),
            torch.tensor([100.0]),
            torch.tensor([14.0]),
            torch.tensor([106.0]),
        ]
    )
    config = _config(approximator="taylor1", warmup_steps=2, calls_per_step=2)

    with denoise_step_cache(module, config, total_steps=4):
        outputs = [module(torch.ones(1)) for _ in range(6)]

    torch.testing.assert_close(outputs[4], torch.tensor([18.0]))
    torch.testing.assert_close(outputs[5], torch.tensor([112.0]))


def test_polynomial_can_force_middle_steps_to_compute():
    module = _AddOne()
    config = _config(coefficients=(100.0, 0.0))

    with denoise_step_cache(module, config, total_steps=4) as stats:
        for scale in (1.0, 1.01, 1.02, 1.03):
            module(torch.full((4,), scale))

    assert module.calls == 4
    assert stats.computed == 4
    assert stats.skipped == 0


def test_v2_accumulates_across_skipped_steps_and_resets_on_compute(
    monkeypatch,
):
    module = _AddOne()
    config = _config(indicator="teacache", accumulate=True, threshold=0.25)
    signals = (1.0, 1.1, 1.21, 1.331, 1.39755, 1.4674275)
    monkeypatch.setattr(
        denoise_module,
        "_teacache_signal",
        lambda _module, args, _kwargs, _config: args[0],
    )

    with denoise_step_cache(module, config, total_steps=6) as stats:
        outputs = [module(torch.full((4,), value)) for value in signals]

    assert module.calls == 3
    assert stats.computed == 3
    assert stats.skipped == 3
    assert torch.equal(outputs[1], outputs[0])
    assert torch.equal(outputs[2], outputs[0])
    assert not torch.equal(outputs[3], outputs[0])
    assert torch.equal(outputs[4], outputs[3])


def test_v2_forced_boundary_computes_reset_accumulator(monkeypatch):
    module = _AddOne()
    config = _config(
        indicator="teacache",
        accumulate=True,
        threshold=0.2,
        warmup_steps=2,
    )
    monkeypatch.setattr(
        denoise_module,
        "_teacache_signal",
        lambda _module, args, _kwargs, _config: args[0],
    )

    with denoise_step_cache(module, config, total_steps=4) as stats:
        for value in (1.0, 1.15, 1.3225, 1.520875, 1.6729625):
            module(torch.full((4,), value))

    assert module.calls == 3
    assert stats.computed == 3
    assert stats.skipped == 2


def test_old_config_keeps_per_call_threshold_behavior():
    module = _AddOne()

    with denoise_step_cache(module, _config(threshold=0.2), total_steps=4):
        for value in (1.0, 1.15, 1.3225, 1.520875):
            module(torch.full((4,), value))

    assert module.calls == 2


@pytest.mark.parametrize("distance", [float("nan"), float("inf")])
def test_non_finite_distance_forces_middle_step_compute(monkeypatch, distance):
    module = _AddOne()
    monkeypatch.setattr(
        denoise_module, "_relative_l1", lambda _current, _previous: distance
    )

    with denoise_step_cache(module, _config(), total_steps=3) as stats:
        outputs = [module(torch.ones(4)) for _ in range(3)]

    assert module.calls == 3
    assert stats.computed == 3
    assert stats.skipped == 0
    assert outputs[1] is not outputs[0]


def test_calls_per_step_keeps_slots_independent():
    module = _AddOne()
    config = _config(calls_per_step=2)

    with denoise_step_cache(module, config, total_steps=3):
        outputs = []
        for _ in range(3):
            outputs.append(module(torch.ones(4)))
            outputs.append(module(torch.full((4,), 2.0)))

    assert module.calls == 4
    assert torch.equal(outputs[3], outputs[1])
    assert not torch.equal(outputs[3], outputs[0])


def test_residual_mode_rejects_shape_mismatch_on_replay():
    class WrongShape(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states[:2]

    module = WrongShape()
    config = _config(mode="residual")

    with denoise_step_cache(module, config, total_steps=3):
        module(torch.ones(3))
        with pytest.raises(ValueError, match="shape"):
            module(torch.ones(3))


def test_residual_mode_replays_delta_against_current_input():
    module = _AddOne()
    config = _config(mode="residual")

    with denoise_step_cache(module, config, total_steps=3):
        first = module(torch.ones(3))
        second = module(torch.full((3,), 1.1))

    assert torch.equal(first, torch.full((3,), 2.0))
    assert torch.equal(second, torch.full((3,), 2.1))


def test_context_rejects_reentry_and_restores_forward():
    module = _AddOne()

    with denoise_step_cache(module, _config(), total_steps=3):
        with pytest.raises(RuntimeError, match="already enabled"):
            with denoise_step_cache(module, _config(), total_steps=3):
                pass

    module(torch.ones(1))
    assert module.calls == 1
    assert "forward" not in module.__dict__


def test_context_restores_existing_instance_forward():
    module = _AddOne()

    def instance_forward(hidden_states):
        return hidden_states + 5

    module.forward = instance_forward

    with denoise_step_cache(module, _config(), total_steps=1):
        assert torch.equal(module(torch.ones(1)), torch.full((1,), 6.0))

    assert module.forward is instance_forward


def test_context_preserves_forward_signature():
    module = _AddOne()

    with denoise_step_cache(module, _config(), total_steps=1):
        assert tuple(inspect.signature(module.forward).parameters) == (
            "hidden_states",
        )


def test_context_survives_forward_rebinding():
    module = _AddOne()
    original_forward = module.forward

    def evicting_forward(hidden_states):
        module.__dict__["forward"] = original_forward
        return original_forward(hidden_states)

    module.forward = evicting_forward
    with denoise_step_cache(module, _config(), total_steps=4) as stats:
        outputs = [module(torch.ones(4)) for _ in range(4)]

    assert stats.computed == 2
    assert stats.skipped == 2
    assert module.calls == 2
    assert torch.equal(outputs[1], outputs[0])
    assert module.forward is evicting_forward


class _SyntheticAdaLN(torch.nn.Module):
    def forward(self, temb):
        shift_msa = torch.stack((temb, temb + 1, temb + 2), dim=1).flatten(0, 1)
        scale_msa = shift_msa * 0.5
        zeros = torch.zeros_like(shift_msa)
        return shift_msa, scale_msa, zeros, zeros, zeros, zeros


class _SyntheticBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.RMSNorm(2, eps=1e-5)
        self.adaln_proj = _SyntheticAdaLN()


class _SyntheticH3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = torch.nn.Identity()
        self.time_proj = torch.nn.Identity()
        self.time_embedder = torch.nn.Identity()
        self.transformer_blocks = torch.nn.ModuleList([_SyntheticBlock()])


class _SyntheticFBCacheBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.Identity()
        self.adaln_proj = torch.nn.Identity()

    def forward(
        self,
        hidden_states,
        temb,
        adaln_indices,
        rotary_emb,
        attention_mask=None,
    ):
        del temb, adaln_indices, rotary_emb, attention_mask
        return hidden_states + hidden_states * 0.1


class _SyntheticFBCacheH3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = torch.nn.Identity()
        self.audio_proj_in = torch.nn.Identity()
        self.context_embedder = torch.nn.Identity()
        self.token_refiner = torch.nn.Identity()
        self.time_proj = torch.nn.Identity()
        self.time_embedder = torch.nn.Identity()
        self.rope = lambda position_ids: (position_ids, -position_ids)
        self.transformer_blocks = torch.nn.ModuleList(
            [_SyntheticFBCacheBlock()]
        )
        self.calls = 0

    def forward(self, hidden_states, **_kwargs):
        self.calls += 1
        return hidden_states + 1


def _fbcache_kwargs(video_value):
    return {
        "hidden_states": torch.full((1, 2, 2), video_value),
        "audio_hidden_states": torch.empty((1, 0, 2)),
        "encoder_hidden_states": torch.empty((1, 0, 2)),
        "timestep": torch.tensor([[0.25, 0.5]]),
        "timestep_indices": torch.tensor([0, 0]),
        "token_tags": torch.tensor([0, 0]),
        "position_ids": torch.zeros((2, 3)),
        "video_indices": torch.tensor([0, 1]),
        "audio_indices": torch.empty(0, dtype=torch.long),
        "text_indices": torch.empty(0, dtype=torch.long),
    }


def test_fbcache_signal_matches_first_block_residual():
    module = _SyntheticFBCacheH3()
    kwargs = _fbcache_kwargs(3.0)

    actual = denoise_module._fbcache_signal(
        module, (), kwargs, _config(indicator="fbcache")
    )

    torch.testing.assert_close(actual, kwargs["hidden_states"] * 0.1)


def test_fbcache_indicator_decision_responds_to_first_block_residual_delta():
    module = _SyntheticFBCacheH3()
    config = _config(indicator="fbcache", threshold=0.2)

    with denoise_step_cache(module, config, total_steps=4) as stats:
        for value in (1.0, 1.01, 2.0):
            module(**_fbcache_kwargs(value))

    assert module.calls == 2
    assert stats.computed == 2
    assert stats.skipped == 1


def test_teacache_signal_matches_h3_first_block_adaln_formula():
    module = _SyntheticH3()
    hidden_states = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])
    timestep = torch.tensor([[0.25, 0.5]])
    timestep_indices = torch.tensor([0, 0])
    token_tags = torch.tensor([0, 0])
    video_indices = torch.tensor([0, 1])
    kwargs = {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "video_indices": video_indices,
    }

    actual = denoise_module._teacache_signal(
        module, (), kwargs, _config(indicator="teacache", accumulate=True)
    )

    shift_msa, scale_msa, *_ = module.transformer_blocks[0].adaln_proj(timestep)
    adaln_indices = (timestep_indices * 3 + token_tags).index_select(
        0, video_indices
    )
    expected = module.transformer_blocks[0].norm1(hidden_states)
    expected = expected * (
        1.0 + scale_msa.index_select(0, adaln_indices)
    ) + shift_msa.index_select(0, adaln_indices)

    torch.testing.assert_close(actual, expected)


class _ParamAdaLN(_SyntheticAdaLN):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.received_dtype = None

    def forward(self, temb):
        self.received_dtype = temb.dtype
        return super().forward(self.proj(temb.to(self.proj.weight.dtype)))


class _ProjectedH3(_SyntheticH3):
    def __init__(self, dtype):
        super().__init__()
        self.proj_in = torch.nn.Linear(2, 2, dtype=dtype)
        self.transformer_blocks[0].adaln_proj = _ParamAdaLN().to(dtype)


def test_teacache_signal_follows_module_weight_dtype():
    module = _ProjectedH3(torch.float64)
    kwargs = {
        "hidden_states": torch.tensor([[[3.0, 4.0], [5.0, 12.0]]]),
        "timestep": torch.tensor([[0.25, 0.5]]),
        "timestep_indices": torch.tensor([0, 0]),
        "token_tags": torch.tensor([0, 0]),
        "video_indices": torch.tensor([0, 1]),
    }

    signal = denoise_module._teacache_signal(
        module, (), kwargs, _config(indicator="teacache", accumulate=True)
    )

    assert signal.dtype == module.transformer_blocks[0].norm1.weight.dtype


@pytest.mark.gpu
def test_teacache_signal_bridges_offloaded_weights_and_cuda_inputs():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    module = _ProjectedH3(torch.float32)
    kwargs = {
        "hidden_states": torch.tensor(
            [[[3.0, 4.0], [5.0, 12.0]]], device="cuda"
        ),
        "timestep": torch.tensor([[0.25, 0.5]], device="cuda"),
        "timestep_indices": torch.tensor([0, 0], device="cuda"),
        "token_tags": torch.tensor([0, 0], device="cuda"),
        "video_indices": torch.tensor([0, 1], device="cuda"),
    }

    signal = denoise_module._teacache_signal(
        module, (), kwargs, _config(indicator="teacache", accumulate=True)
    )

    assert signal.device.type == "cpu"


def test_adaln_receives_temb_at_incoming_precision():
    module = _ProjectedH3(torch.float64)
    kwargs = {
        "hidden_states": torch.tensor([[[3.0, 4.0], [5.0, 12.0]]]),
        "timestep": torch.tensor([[0.25, 0.5]]),
        "timestep_indices": torch.tensor([0, 0]),
        "token_tags": torch.tensor([0, 0]),
        "video_indices": torch.tensor([0, 1]),
    }

    denoise_module._teacache_signal(
        module, (), kwargs, _config(indicator="teacache", accumulate=True)
    )

    adaln = module.transformer_blocks[0].adaln_proj
    assert adaln.received_dtype == torch.float32


class _BufferRope(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.register_buffer(
            "inv_freq", torch.ones(2, device=device), persistent=False
        )

    def forward(self, position_ids):
        return (position_ids, -position_ids)


@pytest.mark.gpu
def test_fbcache_signal_bridges_gpu_rope_buffer_and_cpu_block():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    module = _SyntheticFBCacheH3()
    module.rope = _BufferRope("cuda")
    kwargs = {
        name: value.to("cuda") for name, value in _fbcache_kwargs(3.0).items()
    }

    signal = denoise_module._fbcache_signal(
        module, (), kwargs, _config(indicator="fbcache")
    )

    torch.testing.assert_close(
        signal.cpu(), _fbcache_kwargs(3.0)["hidden_states"] * 0.1
    )


def test_cache_decision_wrapper_remains_plain_python():
    assert inspect.isfunction(denoise_module._cache_decision)
    assert inspect.isfunction(inspect.unwrap(denoise_module._cache_decision))


def test_forward_hook_runs_for_skipped_call():
    module = _AddOne()
    hook_calls = 0

    def count_hook(_module, _inputs, _output):
        nonlocal hook_calls
        hook_calls += 1

    handle = module.register_forward_hook(count_hook)
    try:
        with denoise_step_cache(module, _config(), total_steps=3):
            module(torch.ones(1))
            module(torch.ones(1))
    finally:
        handle.remove()

    assert module.calls == 1
    assert hook_calls == 2


def test_enable_cache_dit_fails_closed_when_package_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "cache_dit", None)

    with pytest.raises(ImportError, match="cache-dit is not installed"):
        enable_cache_dit(object())


def test_enable_cache_dit_reraises_missing_transitive_dependency(monkeypatch):
    error = ModuleNotFoundError(
        "No module named 'cache_dit_dependency'",
        name="cache_dit_dependency",
    )

    def import_cache_dit(_name):
        raise error

    monkeypatch.setattr(
        denoise_module.importlib, "import_module", import_cache_dit
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        enable_cache_dit(object())

    assert exc_info.value is error


def test_enable_cache_dit_calls_optional_package(monkeypatch):
    calls = []
    module = object()
    fake = types.ModuleType("cache_dit")
    fake.enable_cache = lambda target, **kwargs: calls.append((target, kwargs))
    monkeypatch.setitem(sys.modules, "cache_dit", fake)

    enable_cache_dit(module, threshold=0.2)

    assert calls == [(module, {"threshold": 0.2})]


def test_denoise_cache_stays_off_registry():
    assert "denoise-cache" not in registry.OPTIMIZATIONS
    with pytest.raises(ValueError):
        registry.runner_kwargs_for("h3-dense", ["denoise-cache"])


def test_documentation_records_safety_boundaries():
    text = Path("docs/caches_c5_denoise.md").read_text()
    for sentence in (
        "No published TeaCache or cache-dit table covers MiniMax-H3.",
        "This plan does not add a CLIP, SSIM, or PSNR gate.",
        "Never publish H3 text K/V from a denoise step.",
        'mode="output" is the H3 default.',
    ):
        assert sentence in text
