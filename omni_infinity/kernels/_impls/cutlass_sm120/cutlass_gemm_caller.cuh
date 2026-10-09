// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM's CUTLASS 3.x GEMM caller for plain at::Tensor JIT builds.
#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "vendor_common.cuh"

namespace omni_infinity::cutlass_sm120 {

template <typename GemmKernel>
void cutlass_gemm_caller(
    at::Tensor const& like,
    cute::Shape<int, int, int, int> problem_shape,
    typename GemmKernel::MainloopArguments mainloop_args,
    typename GemmKernel::EpilogueArguments epilogue_args,
    typename GemmKernel::TileSchedulerArguments scheduler = {}) {
  cutlass::KernelHardwareInfo hardware_info;
  typename GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, problem_shape, mainloop_args,
      epilogue_args, hardware_info, scheduler};

  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  OMO_CUTLASS_CHECK(gemm_op.can_implement(arguments));

  auto workspace = at::empty(
      {static_cast<int64_t>(gemm_op.get_workspace_size(arguments))},
      like.options().dtype(at::kByte));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(like.get_device());
  OMO_CUTLASS_CHECK(
      gemm_op.run(arguments, workspace.data_ptr(), stream));
}

}  // namespace omni_infinity::cutlass_sm120
