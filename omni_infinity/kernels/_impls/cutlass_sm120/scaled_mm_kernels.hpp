// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM's scaled_mm_kernels.hpp.
#pragma once

#include <ATen/ATen.h>

namespace omni_infinity::cutlass_sm120 {

void scaled_mm_blockwise_sm120_fp8(at::Tensor& out, at::Tensor const& a,
                                   at::Tensor const& b,
                                   at::Tensor const& a_scales,
                                   at::Tensor const& b_scales);

}  // namespace omni_infinity::cutlass_sm120
