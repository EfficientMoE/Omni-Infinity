# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from omni_infinity.demo.pipeline import (
    SimClock,
    ready_stages,
    run_pipeline,
)


def test_later_encoder_waits_for_its_second():
    done = {("encoder", 0), ("backbone", 0), ("decoder", 0)}
    times = [None, 4]

    assert ("encoder", 1) not in ready_stages(done, times, now=3)
    assert ("encoder", 1) in ready_stages(done, times, now=4)

    clock = SimClock()
    stages = run_pipeline(2, [None, 4], clock=clock)
    assert stages.index(("decoder", 0)) < stages.index(("encoder", 1))
    assert clock.now() == 4


def test_next_encoder_runs_before_previous_decoder():
    stages = run_pipeline(3, [None, 0, 0])
    assert stages.index(("encoder", 2)) < stages.index(("decoder", 1))
    assert stages.index(("backbone", 2)) > stages.index(("decoder", 1))
