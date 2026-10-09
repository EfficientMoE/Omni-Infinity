# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for fitting C5 calibration-probe records."""

import importlib

import numpy as np
import pytest


def _fit_module():
    return importlib.import_module("benchmarks.caches.denoise_fit")


def _records(coefficients, field="input_rel_l1"):
    xs = np.linspace(0.01, 0.5, 20)
    return [
        {field: float(x), "output_rel_l1": float(np.polyval(coefficients, x))}
        for x in xs
    ]


def test_fit_coefficients_recovers_known_polynomial():
    fit = _fit_module()
    expected = (2.0, -0.5, 0.25)
    actual = fit.fit_coefficients(_records(expected), "raw", degree=2)
    assert np.allclose(actual, expected)


def test_fit_coefficients_selects_requested_signal_and_ignores_invalid_rows():
    fit = _fit_module()
    expected = (1.5, 0.2)
    records = _records(expected, field="teacache_rel_l1")
    records.extend(
        [
            {"teacache_rel_l1": None, "output_rel_l1": 1.0},
            {"teacache_rel_l1": 1.0, "output_rel_l1": float("inf")},
        ]
    )
    actual = fit.fit_coefficients(records, "teacache", degree=1)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("degree", [-1, 20])
def test_fit_coefficients_validates_degree_and_sample_count(degree):
    fit = _fit_module()
    with pytest.raises(ValueError):
        fit.fit_coefficients(_records((1.0, 0.0)), "raw", degree=degree)


def test_fit_coefficients_rejects_unknown_signal():
    fit = _fit_module()
    with pytest.raises(ValueError, match="signal"):
        fit.fit_coefficients(_records((1.0, 0.0)), "unknown", degree=1)


def test_rescaled_values_preserve_negative_fitted_deltas():
    fit = _fit_module()
    records = [
        {"input_rel_l1": 0.1, "output_rel_l1": -0.1},
        {"input_rel_l1": 0.2, "output_rel_l1": -0.2},
        {"input_rel_l1": 0.3, "output_rel_l1": -0.3},
    ]
    coefficients = fit.fit_coefficients(records, "raw", degree=1)

    values = fit.rescaled_values(records, "raw", coefficients)

    assert all(value < 0 for value in values)


def test_accumulated_skip_fraction_resets_on_forced_boundary_calls():
    fit = _fit_module()
    values = (0.05, 0.05, 0.05, 0.05)

    unbounded = fit.accumulated_skip_fraction(
        values, 1.0, warmup_calls=0, final_calls=0
    )
    assert unbounded == 1.0

    with_final = fit.accumulated_skip_fraction(values, 1.0)
    assert with_final == 0.75

    with_warmup = fit.accumulated_skip_fraction(
        values, 1.0, warmup_calls=2, final_calls=1
    )
    assert with_warmup == 0.25


def test_threshold_suggestions_are_ordered_and_reach_skip_targets():
    fit = _fit_module()
    records = _records((1.0, 0.0))
    coefficients = fit.fit_coefficients(records, "raw", degree=1)
    suggestions = fit.suggest_thresholds(records, "raw", coefficients)

    assert list(suggestions) == [10, 25, 50]
    assert list(suggestions.values()) == sorted(suggestions.values())
    values = fit.rescaled_values(records, "raw", coefficients)
    for percent, threshold in suggestions.items():
        assert fit.accumulated_skip_fraction(values, threshold) >= percent / 100
