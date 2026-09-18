# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

try:
    from importlib.metadata import version

    __version__ = version("omni-infinity")
except Exception:
    __version__ = "0.0.0"

__all__ = ["__version__"]
