# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the source-free VDN profiling shim."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SHIM = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "vdn_prof_shim"
    / "sitecustomize.py"
)


def _load_shim():
    spec = importlib.util.spec_from_file_location(
        "vdn_prof_sitecustomize",
        SHIM,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shim_imports_without_upstream_package(monkeypatch):
    monkeypatch.delitem(sys.modules, "src", raising=False)

    module = _load_shim()

    assert module.UPSTREAM_PATCH_TARGETS
    assert "src" not in sys.modules
    assert module._module_range_name("MiniMaxH3Attention") == "dense_attn"


def test_range_accounting_sums_mock_cuda_events():
    module = _load_shim()
    timestamps = iter((0.0, 2.0, 8.0, 10.0))
    synchronizations = []

    class FakeEvent:
        def record(self):
            self.timestamp = next(timestamps)

        def elapsed_time(self, other):
            return other.timestamp - self.timestamp

    accounting = module.RangeAccounting(
        event_factory=FakeEvent,
        synchronize=lambda: synchronizations.append(True),
    )

    with accounting.measure("step_total"):
        with accounting.measure("dense_attn"):
            pass

    assert synchronizations == [True]
    assert accounting.steps == [
        {
            "dense_attn": {"total_ms": 6.0, "calls": 1},
            "step_total": {"total_ms": 10.0, "calls": 1},
        }
    ]


def test_linear_branch_accounting_subtracts_only_its_own_gate():
    module = _load_shim()
    timestamps = iter((0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0))

    class FakeEvent:
        def record(self):
            self.timestamp = next(timestamps)

        def elapsed_time(self, other):
            return other.timestamp - self.timestamp

    accounting = module.RangeAccounting(FakeEvent, lambda: None)
    with accounting.measure("step_total"):
        with accounting.measure("linear_branch"):
            with accounting.measure("gates"):
                pass
        with accounting.measure("linear_branch"):
            with accounting.measure("gates"):
                pass

    assert accounting.steps[0]["linear_branch"] == {
        "total_ms": 6.0,
        "calls": 2,
    }


def test_driver_rejects_busy_gpu():
    driver_path = (
        Path(__file__).parents[1] / "benchmarks" / "attribution_vdn.py"
    )
    spec = importlib.util.spec_from_file_location(
        "attribution_vdn",
        driver_path,
    )
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)

    assert driver._gpu_is_idle("3, 0")
    assert not driver._gpu_is_idle("90000, 99")
    assert driver.DEFAULT_CKPTS == driver.DEFAULT_VDN / "ckpts"


def test_profile_summary_drops_warmups_and_derives_residual():
    driver_path = (
        Path(__file__).parents[1] / "benchmarks" / "attribution_vdn.py"
    )
    spec = importlib.util.spec_from_file_location(
        "attribution_vdn_summary",
        driver_path,
    )
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    step = {
        "step_total": {"total_ms": 1000.0, "calls": 1},
        "dense_attn": {"total_ms": 800.0, "calls": 52},
        "linear_calls": {"total_ms": 0.0, "calls": 370},
    }

    summary = driver._summarize_steps([step, step, step], [1.02], warmups=2)

    assert summary["step_total_ms"] == 1000.0
    assert summary["other_ms"] == 200.0
    assert summary["ratios"] == [1000.0 / 1020.0]
