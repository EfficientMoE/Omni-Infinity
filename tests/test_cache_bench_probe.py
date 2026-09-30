# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU test: the probe hook records one row per transformer call."""

import torch

from benchmarks.caches.denoise_probe import probe_forward


class _Toy(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 1.0


def test_probe_records_relative_distances():
    module = _Toy()
    with probe_forward(module) as records:
        for step in range(1, 5):
            module(hidden_states=torch.full((2, 4), float(step)))
    assert len(records) == 4
    assert records[0]["input_rel_l1"] is None  # no previous step
    assert records[1]["input_rel_l1"] is not None
    assert all("output_rel_l1" in r for r in records)


def test_probe_restores_preexisting_instance_forward():
    module = _Toy()

    def instance_forward(hidden_states):
        return hidden_states + 2.0

    module.forward = instance_forward
    with probe_forward(module):
        module(hidden_states=torch.ones(1))
    assert module.__dict__["forward"] is instance_forward
