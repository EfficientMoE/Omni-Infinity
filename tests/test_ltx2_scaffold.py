# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Scaffold tests for the WIP LTX-2.5 arch (issue #54).

These lock the registry wiring and the runner surface. The real
loading/generation gates (golden parity, two-stage decode, audio mux)
arrive with the integration and are marked ``skip`` until then.
"""

import pytest

from omni_infinity import registry
from omni_infinity.arch.ltx2 import Ltx2Runner


def test_ltx2_registered_in_arch_registry():
    profile = registry.resolve_profile("ltx-2.5", [])
    assert profile.model_arch == "ltx-2.5"
    assert profile.runner is Ltx2Runner
    assert profile.checkpoint == "Lightricks/LTX-2.5"
    assert profile.runner_kwargs == {}


def test_ltx2_runner_exposes_the_runner_surface():
    runner = Ltx2Runner()
    assert runner.pipeline is None
    assert runner.default_steps == 8
    assert hasattr(Ltx2Runner, "from_pretrained")
    assert hasattr(runner, "generate")


def test_ltx2_from_pretrained_is_not_implemented_yet():
    with pytest.raises(NotImplementedError, match="issues/54"):
        Ltx2Runner.from_pretrained()


def test_ltx2_generate_is_not_implemented_yet():
    runner = Ltx2Runner()
    with pytest.raises(NotImplementedError, match="not implemented"):
        runner.generate("a red ball bouncing")


@pytest.mark.skip(reason="LTX-2.5 generation pending integration (#54)")
def test_ltx2_golden_parity():
    raise AssertionError("placeholder: real e2e parity gate lands with #54")
