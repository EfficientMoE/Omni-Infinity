# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Model-arch category: one runner per supported architecture.

- ``h3-dense``  -> :class:`omni_infinity.runner.ReferenceRunner`
- ``vdn-hybrid`` -> :class:`omni_infinity.arch.vdn.VdnRunner`
- ``ltx-2.5``   -> :class:`omni_infinity.arch.ltx2.Ltx2Runner` (WIP #54)
"""

from omni_infinity.arch.ltx2 import Ltx2Runner
from omni_infinity.arch.vdn import VdnRunner
from omni_infinity.runner import ReferenceRunner

__all__ = ["Ltx2Runner", "ReferenceRunner", "VdnRunner"]
