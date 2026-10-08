# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU test: the probe hook records one row per transformer call."""

import torch

import benchmarks.caches.denoise_probe as denoise_probe
from benchmarks.caches.denoise_probe import probe_forward


class _Toy(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 1.0


class _TensorlessToy(torch.nn.Module):
    def forward(self, hidden_states):
        return {"metadata": "no tensor output"}


def test_probe_records_relative_distances():
    module = _Toy()
    with probe_forward(module) as records:
        for step in range(4):
            module(hidden_states=torch.full((2, 4), float(step)))
    assert len(records) == 4
    assert records[0]["input_rel_l1"] is None
    assert records[1]["input_rel_l1"] is not None
    assert all("output_rel_l1" in record for record in records)


def test_probe_records_v2_indicator_distances_from_full_kwargs(monkeypatch):
    seen = []

    def signal(_module, _args, kwargs, _config):
        seen.append(kwargs["timestep"])
        return kwargs["hidden_states"] * kwargs["scale"]

    monkeypatch.setattr(denoise_probe, "_teacache_signal", signal)
    monkeypatch.setattr(denoise_probe, "_fbcache_signal", signal)

    class SignalToy(torch.nn.Module):
        def forward(self, hidden_states, *, timestep, scale):
            return hidden_states + timestep

    module = SignalToy()
    with probe_forward(module) as records:
        module(hidden_states=torch.ones(2, 4), timestep=1.0, scale=2.0)
        module(hidden_states=torch.full((2, 4), 2.0), timestep=2.0, scale=2.0)

    assert seen == [1.0, 1.0, 2.0, 2.0]
    assert records[0]["teacache_rel_l1"] is None
    assert records[0]["fbcache_rel_l1"] is None
    assert records[1]["teacache_rel_l1"] == 1.0
    assert records[1]["fbcache_rel_l1"] == 1.0
    assert records[1]["teacache_error"] is None
    assert records[1]["fbcache_error"] is None


def test_probe_records_indicator_failure_without_crashing(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise AttributeError("missing block attribute")

    monkeypatch.setattr(denoise_probe, "_teacache_signal", unavailable)
    monkeypatch.setattr(denoise_probe, "_fbcache_signal", unavailable)
    module = _Toy()
    with probe_forward(module) as records:
        module(hidden_states=torch.ones(2, 4))

    assert records[0]["teacache_rel_l1"] is None
    assert records[0]["fbcache_rel_l1"] is None
    assert "missing block attribute" in records[0]["teacache_error"]
    assert "missing block attribute" in records[0]["fbcache_error"]


def test_probe_records_tensorless_output_without_advancing_output_state():
    module = _TensorlessToy()
    with probe_forward(module) as records:
        module(hidden_states=torch.zeros(2, 4))
        module(hidden_states=torch.ones(2, 4))
    assert [record["output_rel_l1"] for record in records] == [None, None]


def test_probe_preserves_the_forward_signature():
    import inspect

    module = _Toy()
    with probe_forward(module):
        parameters = inspect.signature(module.forward).parameters
        assert "hidden_states" in parameters
        module(hidden_states=torch.zeros(2))
