# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Fit MiniMax-H3 C5 polynomial coefficients from a denoise probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

_SIGNAL_FIELDS = {
    "raw": "input_rel_l1",
    "teacache": "teacache_rel_l1",
    "fbcache": "fbcache_rel_l1",
}


def _pairs(records: list[dict], signal: str) -> list[tuple[float, float]]:
    try:
        field = _SIGNAL_FIELDS[signal]
    except KeyError:
        choices = ", ".join(_SIGNAL_FIELDS)
        raise ValueError(f"signal must be one of: {choices}") from None

    pairs = []
    for record in records:
        x = record.get(field)
        y = record.get("output_rel_l1")
        if x is None or y is None:
            continue
        x = float(x)
        y = float(y)
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    return pairs


def fit_coefficients(
    records: list[dict], signal: str, degree: int
) -> tuple[float, ...]:
    """Fit highest-degree-first coefficients for ``DenoiseCacheConfig``."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    pairs = _pairs(records, signal)
    if len(pairs) <= degree:
        raise ValueError(
            f"degree {degree} requires at least {degree + 1} finite records"
        )
    xs, ys = zip(*pairs, strict=True)
    if len(set(xs)) <= degree:
        raise ValueError(
            f"degree {degree} requires at least {degree + 1} distinct signals"
        )
    return tuple(float(value) for value in np.polyfit(xs, ys, degree))


def rescaled_values(
    records: list[dict], signal: str, coefficients: tuple[float, ...]
) -> tuple[float, ...]:
    """Return finite, non-negative fitted deltas in probe-call order."""
    values = []
    for x, _ in _pairs(records, signal):
        value = float(np.polyval(coefficients, x))
        if math.isfinite(value):
            values.append(max(0.0, value))
    return tuple(values)


def accumulated_skip_fraction(
    values: tuple[float, ...], threshold: float
) -> float:
    """Simulate C5's accumulate-until-threshold rule for fitted deltas."""
    if not values:
        return 0.0
    accumulated = 0.0
    skipped = 0
    for value in values:
        accumulated += value
        if accumulated < threshold:
            skipped += 1
        else:
            accumulated = 0.0
    return skipped / len(values)


def suggest_thresholds(
    records: list[dict],
    signal: str,
    coefficients: tuple[float, ...],
    percents: tuple[int, ...] = (10, 25, 50),
) -> dict[int, float]:
    """Find thresholds that reach each requested accumulated skip rate."""
    values = rescaled_values(records, signal, coefficients)
    if not values:
        raise ValueError("no finite probe records available for thresholds")
    if any(percent <= 0 or percent >= 100 for percent in percents):
        raise ValueError("threshold percentages must be between 0 and 100")

    upper = sum(values) + max(values, default=0.0) + 1.0
    suggestions = {}
    for percent in percents:
        target = percent / 100.0
        low = 0.0
        high = upper
        for _ in range(80):
            midpoint = (low + high) / 2.0
            if accumulated_skip_fraction(values, midpoint) >= target:
                high = midpoint
            else:
                low = midpoint
        suggestions[percent] = high
    return suggestions


def _coefficient_text(coefficients: tuple[float, ...]) -> str:
    return ",".join(format(value, ".17g") for value in coefficients)


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument(
        "--signal", choices=tuple(_SIGNAL_FIELDS), default="raw"
    )
    parser.add_argument("--degree", type=int, default=4)
    args = parser.parse_args()

    payload = json.loads(args.probe.read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    coefficients = fit_coefficients(records, args.signal, args.degree)
    thresholds = suggest_thresholds(records, args.signal, coefficients)
    print(f"--denoise-cache-coefficients {_coefficient_text(coefficients)}")
    for percent, threshold in thresholds.items():
        print(
            f"skip {percent}% (accumulate): "
            f"--denoise-cache-threshold {format(threshold, '.17g')}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
