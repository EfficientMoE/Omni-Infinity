# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Lazy JIT loader for the vendored CUTLASS SM120 extension."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path


def _cutlass_include_dir() -> Path:
    cutlass_dir = os.environ.get("OMO_CUTLASS_DIR")
    if not cutlass_dir:
        raise RuntimeError(
            "cutlass_sm120 backend requires OMO_CUTLASS_DIR to point to "
            "a CUTLASS v4.4.2 checkout"
        )
    include_dir = Path(cutlass_dir).expanduser().resolve() / "include"
    if not (include_dir / "cutlass" / "cutlass.h").is_file():
        raise RuntimeError(
            "cutlass_sm120 backend could not find CUTLASS headers under "
            f"OMO_CUTLASS_DIR={cutlass_dir!r}"
        )
    return include_dir


@lru_cache(maxsize=1)
def load_extension():
    """Compile and load the extension on first explicit backend use."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("cutlass_sm120 backend requires nvcc on PATH")
    cuda_home = Path(nvcc).resolve().parent.parent
    cuda_include = cuda_home / "include"
    cuda_lib = cuda_home / "lib64"
    if not (cuda_include / "cuda_runtime.h").is_file():
        raise RuntimeError(
            f"cutlass_sm120 backend found nvcc at {nvcc!r} but no "
            f"CUDA headers under {cuda_include}"
        )

    from torch.utils import cpp_extension

    # ``nvcc`` may be exposed through /usr/bin while its companion tools
    # (cudafe++, cicc) live beside the resolved binary. Point cpp_extension at
    # the real toolkit so both generated paths and nvcc subprocesses are valid.
    cpp_extension.CUDA_HOME = str(cuda_home)

    source_dir = Path(__file__).resolve().parent
    cutlass_include = _cutlass_include_dir()
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{cuda_home / 'bin'}{os.pathsep}{old_path}"
    try:
        return cpp_extension.load(
            name="omni_infinity_cutlass_sm120",
            sources=[str(source_dir / "scaled_mm_blockwise_sm120_fp8.cu")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "-gencode",
                "arch=compute_120f,code=sm_120f",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-DNDEBUG",
                "--threads",
                "4",
            ],
            extra_ldflags=[f"-L{cuda_lib}"],
            extra_include_paths=[
                str(source_dir),
                str(cutlass_include),
                str(cutlass_include.parent / "tools" / "util" / "include"),
                str(cuda_include),
                str(cuda_include / "cccl"),
            ],
            with_cuda=True,
            verbose=os.environ.get("OMO_CUTLASS_VERBOSE") == "1",
        )
    finally:
        os.environ["PATH"] = old_path
