// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM csrc/cutlass_extensions/common.hpp.
#pragma once

#include <cstdio>
#include <utility>

#include "cutlass/cutlass.h"

#define OMO_CUTLASS_CHECK(status)                                      \
  do {                                                                 \
    cutlass::Status error = (status);                                  \
    TORCH_CHECK(error == cutlass::Status::kSuccess,                    \
                cutlassGetStatusString(error));                        \
  } while (false)

template <typename Kernel>
struct enable_sm120_family : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    Kernel::operator()(std::forward<Args>(args)...);
#else
    printf("This kernel only supports the SM12x family.\n");
    asm("trap;");
#endif
#endif
  }
};
