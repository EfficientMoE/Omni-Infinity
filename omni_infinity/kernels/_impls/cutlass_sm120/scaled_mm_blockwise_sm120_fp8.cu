// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM's CUTLASS SM120 blockwise FP8 GEMM.

#include <torch/extension.h>

#include "scaled_mm_blockwise_sm120_fp8_dispatch.cuh"
#include "scaled_mm_kernels.hpp"

namespace omni_infinity::cutlass_sm120 {

void scaled_mm_blockwise_sm120_fp8(at::Tensor& out, at::Tensor const& a,
                                   at::Tensor const& b,
                                   at::Tensor const& a_scales,
                                   at::Tensor const& b_scales) {
  TORCH_CHECK(out.is_cuda() && a.is_cuda() && b.is_cuda() &&
                  a_scales.is_cuda() && b_scales.is_cuda(),
              "all cutlass_sm120 tensors must be CUDA tensors");
  TORCH_CHECK(out.get_device() == a.get_device() &&
                  a.get_device() == b.get_device() &&
                  a.get_device() == a_scales.get_device() &&
                  a.get_device() == b_scales.get_device(),
              "all cutlass_sm120 tensors must be on the same device");
  TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn &&
                  b.scalar_type() == at::kFloat8_e4m3fn,
              "cutlass_sm120 A and B must be float8_e4m3fn");
  TORCH_CHECK(a_scales.scalar_type() == at::kFloat &&
                  b_scales.scalar_type() == at::kFloat,
              "cutlass_sm120 scales must be float32");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2,
              "cutlass_sm120 operands must be 2D");
  TORCH_CHECK(a.is_contiguous(), "cutlass_sm120 A must be row-major");
  TORCH_CHECK(b.stride(0) == 1 && b.stride(1) == b.size(0),
              "cutlass_sm120 B must be column-major");
  TORCH_CHECK(a_scales.stride(0) == 1 &&
                  a_scales.stride(1) == a_scales.size(0),
              "cutlass_sm120 A scales must be M-major");
  TORCH_CHECK(b_scales.stride(0) == 1 &&
                  b_scales.stride(1) == b_scales.size(0),
              "cutlass_sm120 B scales must be K-major");
  TORCH_CHECK(a.size(1) == b.size(0), "cutlass_sm120 K mismatch");
  TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == b.size(1),
              "cutlass_sm120 output shape mismatch");

  if (out.scalar_type() == at::kBFloat16) {
    cutlass_gemm_blockwise_sm120_fp8_dispatch<cutlass::bfloat16_t>(
        out, a, b, a_scales, b_scales);
  } else {
    TORCH_CHECK(out.scalar_type() == at::kHalf,
                "cutlass_sm120 output must be bfloat16 or float16");
    cutlass_gemm_blockwise_sm120_fp8_dispatch<cutlass::half_t>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace omni_infinity::cutlass_sm120

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("scaled_mm_blockwise_sm120_fp8",
             &omni_infinity::cutlass_sm120::scaled_mm_blockwise_sm120_fp8,
             "CUTLASS SM120 blockwise FP8 GEMM");
}
