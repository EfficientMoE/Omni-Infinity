# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Make the repo root importable so tests can import `benchmarks.*`."""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
