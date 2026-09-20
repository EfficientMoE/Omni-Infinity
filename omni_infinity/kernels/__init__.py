# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Omni-Infinity compute kernels package.

Task 1 lands the pure-torch reference (``_reference.py``) and the block-wise
FP8 quantizer (``_quant.py``) as importable submodules. The op-centric public
facade (``fused_fp8_gemm`` and friends) is populated in Task 3.
"""
from __future__ import annotations
