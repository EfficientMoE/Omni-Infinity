# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from omni_infinity.demo.models import DemoRequest


def test_prompt_count_rejects_short_lists():
    # duration 15s requires 3 prompts
    with pytest.raises(ValidationError) as e1:
        DemoRequest(prompts=["a", "b"], duration="15s")
    assert "duration 15s requires 3 prompts, got 2" in str(e1.value)

    # duration 2min requires 24 prompts
    with pytest.raises(ValidationError) as e2:
        DemoRequest(prompts=["a"] * 23, duration="2min")
    assert "duration 2min requires 24 prompts, got 23" in str(e2.value)

    # duration 5min requires 59 prompts
    with pytest.raises(ValidationError) as e3:
        DemoRequest(prompts=["a"] * 58, duration="5min")
    assert "duration 5min requires 59 prompts, got 58" in str(e3.value)

    # This one should pass
    d = DemoRequest(prompts=["a"] * 3, duration="15s")
    assert d.duration == "15s"


def test_schedule_range_rejects_reversed():
    with pytest.raises(ValidationError) as e:
        DemoRequest(
            prompts=["a"] * 3,
            duration="15s",
            schedule_start=5,
            schedule_end=2,
        )
    assert "schedule end is before schedule start" in str(e.value)

    d = DemoRequest(
        prompts=["a"] * 3,
        duration="15s",
        schedule_start=3,
        schedule_end=3,
    )
    assert d.schedule_start == d.schedule_end == 3


def test_duplicate_optimizations_rejected():
    with pytest.raises(ValidationError) as e:
        DemoRequest(
            prompts=["a"] * 3, duration="15s", optimizations=["fp8", "fp8"]
        )
    assert "optimization names must be unique" in str(e.value)
