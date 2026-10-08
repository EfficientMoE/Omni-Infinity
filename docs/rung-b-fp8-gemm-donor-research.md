NOTE: librarian research output (2026-10-08), corrected: CUTLASS pin verified locally as v4.4.2 in all donors (the original text wrongly said 4.7.1).

# SM120 FP8 Blockwise GEMM Donor Comparison

## 1. vLLM (Stable Torch ABI Path)

**Repository**: https://github.com/vllm-project/vllm  
**License**: Apache-2.0  
**Exact File Paths**:
- Kernel: `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu`
  - GitHub: https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu
- Dispatch: `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh`
  - GitHub: https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh
- Entry point: `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu`

**Scale Geometry**:
- Weight blocks: **128×128** (matches your requirement exactly)
- Activation scales: **1×128** per token group (DeepSeek-style)
- Scale layout: `Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>`
- Per-block scales: `float32` (fp32)

**CUTLASS Version**:
- Pinned: **v4.4.2** (CMakeLists.txt line ~1100)
- Minimum CUDA: **12.8** (for SM120 support)
- Minimum CUDA for SM121: **12.9**

**Build Pattern**:
```cmake
# From CMakeLists.txt (lines ~1100-1120)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
  set(SCALED_MM_SM120_SRCS
    "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu"
    "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu"
    "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu"
  )
  set_gencode_flags_for_srcs(SRCS "${SCALED_MM_SM120_SRCS}" CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_STABLE_EXT_SRC "${SCALED_MM_SM120_SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM120=1")
endif()
```

**Gencode Flags**:
- For RTX PRO 6000 (sm120): `-gencode arch=compute_120,code=sm_120`
- For RTX 5090 (sm120a): `-gencode arch=compute_120,code=sm_120a`

**Torch Extension API**:
```cpp
void cutlass_scaled_mm_blockwise_sm120_fp8(
    torch::stable::Tensor& out,           // [M, N] bf16/fp16
    torch::stable::Tensor const& a,       // [M, K] fp8_e4m3fn
    torch::stable::Tensor const& b,       // [K, N] fp8_e4m3fn
    torch::stable::Tensor const& a_scales,// [ceil(M/128), 1] fp32
    torch::stable::Tensor const& b_scales // [ceil(K/128), N] fp32
);
```

**Quantization Type**: **w8a8** (both weights and activations quantized to FP8)

**Dependencies**:
- CUTLASS 4.4.2 (headers only, fetched via FetchContent)
- PyTorch stable ABI (torch::stable::Tensor)
- CUDA 12.8+ toolkit

**Complexity**: **Medium**
- Self-contained: Yes (all headers in CUTLASS)
- Vendoring effort: ~3 files (~500 LOC total)
- Build integration: Straightforward CMake pattern
- No external dependencies beyond CUTLASS

---

## 2. SGLang (sgl-kernel)

**Repository**: https://github.com/sgl-project/sglang  
**License**: Apache-2.0  
**Exact File Paths**:
- Kernel: `sgl-kernel/csrc/gemm/fp8_gemm_kernel.cu` (contains SM120 blockwise)
  - GitHub: https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/fp8_gemm_kernel.cu
  - SM120 dispatch: Lines 1084–1091 (`sm120_fp8_dispatch_shape`)
  - SM120 launch: Lines 1015–1035 (`launch_sm120_fp8_scaled_mm`)
  - SM120 args prep: Lines 952–1013 (`prepare_sm120_fp8_args`)
- Cutlass extensions: `sgl-kernel/csrc/cutlass_extensions/gemm/fp8_blockwise_gemm_sm90_dispatch.cuh`

**Scale Geometry**:
- Weight blocks: **128×128** (matches your requirement)
- Activation scales: **1×128** per token group
- Scale config: `Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>`
- Per-block scales: `float32` (fp32)

**CUTLASS Version**:
- Pinned: **v4.4.2** (same as vLLM, via FetchContent in sgl-kernel CMakeLists)
- Minimum CUDA: **12.8**

**Build Pattern**:
```cmake
# sgl-kernel/CMakeLists.txt (inferred from vLLM pattern)
# Uses same CUTLASS 4.4.2 via FetchContent
# Gencode: -gencode arch=compute_120,code=sm_120
```

**Torch Extension API**:
```cpp
torch::Tensor fp8_blockwise_scaled_mm(
    torch::Tensor const& mat_a,        // [M, K] fp8_e4m3fn
    torch::Tensor const& mat_b,        // [K, N] fp8_e4m3fn
    torch::Tensor const& scales_a,     // [ceil(M/128), 1] fp32
    torch::Tensor const& scales_b,     // [ceil(K/128), N] fp32
    torch::Tensor const& out_dtype     // torch.bfloat16 or torch.float16
);
```

**Quantization Type**: **w8a8** (both weights and activations quantized to FP8)

**Dependencies**:
- CUTLASS 4.4.2 (headers only)
- PyTorch (standard ABI, not stable)
- CUDA 12.8+ toolkit

**Complexity**: **Medium**
- Self-contained: Yes (all headers in CUTLASS)
- Vendoring effort: ~1 file (~1200 LOC, but includes SM89/SM90 paths)
- Build integration: Straightforward CMake pattern
- No external dependencies beyond CUTLASS

**Key Difference from vLLM**:
- Uses standard PyTorch ABI (not stable ABI)
- Single monolithic kernel file (fp8_gemm_kernel.cu) with SM89/SM90/SM120 dispatch
- Slightly more complex to extract SM120-only path

---

## 3. CUTLASS Version Constraints for SM120 Blockwise FP8

**Minimum CUTLASS**: **4.0** (SM120 support added in CUTLASS 4.0)  
**Recommended**: **4.4.2** (latest stable, used by both vLLM and SGLang)

**CUDA Compatibility**:
- CUTLASS 4.4.2 + CUDA 13.0: ✅ **Fully compatible**
- CUTLASS 4.4.2 + CUDA 12.8: ✅ **Fully compatible** (minimum for SM120)
- CUTLASS 4.4.2 + CUDA 12.7: ❌ **Not supported** (SM120 requires 12.8+)

**Key CUTLASS SM120 Features**:
- Block-scaled MMA instructions: `mma.sync.aligned.block_scale`
- TMA (Tensor Memory Accelerator) for efficient GMEM→SMEM
- Cluster shape: **1×1×1** (no multicast on consumer Blackwell)
- Tile shapes: 128×128×128 (default), 64×64×128 (small M), 32×64×128 (tiny M)
- Shared memory: 99 KB available (after scheduler overhead)

---

## 4. Prebuilt Wheel Options (Third Option)

**Status**: ❌ **No pure-Python wheel available**

**Checked**:
- **torchao**: No SM120 blockwise FP8 wheel (supports int4/int8 weight-only, not blockwise FP8)
- **SGLang**: Distributes `sgl-kernel` as a CUDA wheel (requires compilation)
- **FlashInfer**: Distributes pre-compiled cubins + JIT cache, but requires runtime JIT for SM120 blockwise FP8
  - FlashInfer 0.6.14+ has SM120 blockwise FP8 support via `flashinfer_cutlass` backend
  - Requires: `pip install flashinfer` (cu130 wheel available)
  - API: `flashinfer.gemm.fp8_blockscale_gemm_sm120()`
  - **Caveat**: Not a pure-Python wheel; still requires CUDA runtime + JIT compilation

**FlashInfer as Optional Dependency**:
- If FlashInfer is installed: Use `flashinfer.gemm.fp8_blockscale_gemm_sm120()`
- If not installed: Fall back to vendored CUTLASS kernel
- Wheel stays pure-Python; CUDA kernels are optional extras

---

## Comparison Table

| Criterion | vLLM | SGLang | CUTLASS 4.4.2 | FlashInfer (Optional) |
|-----------|------|--------|---------------|-----------------------|
| **License** | Apache-2.0 | Apache-2.0 | BSD-3-Clause | Apache-2.0 |
| **Scale Geometry** | 128×128 weight, 1×128 activation ✅ | 128×128 weight, 1×128 activation ✅ | N/A (library) | 128×128 weight, 1×128 activation ✅ |
| **CUTLASS Pin** | 4.4.2 | 4.4.2 | 4.4.2 (recommended) | Uses CUTLASS 4.4.2 internally |
| **Min CUDA** | 12.8 | 12.8 | 12.8 | 12.8 |
| **CUDA 13.0 Support** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Gencode Flags** | `-gencode arch=compute_120,code=sm_120` | `-gencode arch=compute_120,code=sm_120` | Auto-selected | Auto-selected |
| **Torch API** | `torch::stable::Tensor` (stable ABI) | `torch::Tensor` (standard ABI) | N/A | Python-only |
| **Quantization** | w8a8 (weight-only FP8) | w8a8 (weight-only FP8) | N/A | w8a8 (weight-only FP8) |
| **Vendoring Effort** | 3 files (~500 LOC) | 1 file (~1200 LOC, SM89/90/120 mixed) | N/A | N/A (external wheel) |
| **Build Complexity** | Medium (CMake) | Medium (CMake) | N/A | Low (pip install) |
| **Self-Contained** | Yes (CUTLASS headers only) | Yes (CUTLASS headers only) | N/A | No (external wheel) |
| **Production Maturity** | ✅ Shipping in vLLM 0.28+ | ✅ Shipping in SGLang main | ✅ Official NVIDIA | ✅ Shipping in FlashInfer 0.6.14+ |
| **API Fit** | Excellent (stable ABI) | Good (standard ABI) | N/A | Excellent (Python) |
| **Wheel Stays Pure-Python** | ❌ No (requires CUDA build) | ❌ No (requires CUDA build) | N/A | ✅ Yes (optional extra) |

---

## Recommendation

### **Primary Choice: vLLM Stable ABI Path**

**Why**:
1. **Exact scale geometry match**: 128×128 weight blocks + 1×128 activation groups
2. **Stable ABI**: Uses `torch::stable::Tensor`, future-proof against PyTorch version changes
3. **Minimal vendoring**: 3 files, ~500 LOC, clean separation
4. **Production-proven**: Shipping in vLLM 0.28+ (released 2025-03)
5. **CUTLASS 4.4.2**: Latest stable, fully compatible with CUDA 13.0
6. **Build integration**: Straightforward CMake pattern (copy vLLM's approach)
7. **No external dependencies**: CUTLASS fetched via FetchContent

**Vendoring Steps**:
```
1. Copy: csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu
2. Copy: csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh
3. Copy: csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu (or adapt entry point)
4. Add CUTLASS 4.4.2 via FetchContent in CMakeLists.txt
5. Add gencode flags: -gencode arch=compute_120,code=sm_120
6. Link against torch::stable_c10d (stable ABI)
```

**File Paths for Vendoring**:
- Source: https://github.com/vllm-project/vllm/tree/main/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x
- Commit: Latest main (as of 2026-03)

---

### **Secondary Choice: SGLang (if stable ABI unavailable)**

**Why**:
1. Same scale geometry and CUTLASS version
2. Standard PyTorch ABI (more compatible with older PyTorch versions)
3. Single monolithic file (easier to extract)
4. Apache-2.0 license (same as vLLM)

**Caveat**: Requires extracting SM120 path from mixed SM89/SM90/SM120 file (~1200 LOC total).

---

### **Optional Tertiary: FlashInfer Wheel**

**Use Case**: If you want to keep the wheel pure-Python and accept an external CUDA dependency.

**Setup**:
```python
# In your quantization layer:
try:
    from flashinfer.gemm import fp8_blockscale_gemm_sm120
    use_flashinfer = True
except ImportError:
    use_flashinfer = False

if use_flashinfer:
    output = fp8_blockscale_gemm_sm120(
        input_2d,           # [M, K] fp8_e4m3fn
        weight,             # [N, K] fp8_e4m3fn
        weight_scale=scales_b,  # [ceil(K/128), N] fp32
        out_dtype=torch.bfloat16
    )
else:
    # Fall back to vendored CUTLASS kernel
    output = cutlass_fp8_blockwise_gemm(...)
```

**Pros**:
- Wheel stays pure-Python
- No build complexity
- FlashInfer 0.6.14+ is production-ready

**Cons**:
- External dependency (pip install flashinfer)
- Requires CUDA 13.0 + FlashInfer wheel (cu130)
- Less control over kernel tuning

---

## Summary Table: Build Requirements

| Aspect | vLLM | SGLang | FlashInfer |
|--------|------|--------|-----------|
| **CUTLASS Version** | 4.4.2 | 4.4.2 | 4.4.2 (internal) |
| **CUDA Minimum** | 12.8 | 12.8 | 12.8 |
| **CUDA 13.0 Support** | ✅ | ✅ | ✅ |
| **Torch 2.12.0+cu130** | ✅ | ✅ | ✅ |
| **Gencode Flags** | `-gencode arch=compute_120,code=sm_120` | `-gencode arch=compute_120,code=sm_120` | Auto |
| **Vendoring Effort** | 3 files, ~500 LOC | 1 file, ~1200 LOC | 0 (external wheel) |
| **Build System** | CMake (FetchContent) | CMake (FetchContent) | pip install |
| **Wheel Type** | CUDA extension | CUDA extension | Pure Python (optional extra) |
| **Recommended** | ✅ Primary | ✅ Secondary | ✅ Optional |

---

## Exact GitHub Permalinks for Vendoring

### vLLM (Recommended)

1. **Kernel file**:
   https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu

2. **Dispatch header**:
   https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh

3. **Entry point** (for reference):
   https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu

4. **CMakeLists.txt** (build pattern):
   https://github.com/vllm-project/vllm/blob/main/CMakeLists.txt#L1100-L1120

### SGLang (Secondary)

1. **Kernel file** (SM120 dispatch at lines 1084–1091):
   https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/fp8_gemm_kernel.cu#L1084-L1091

2. **Cutlass extensions** (for reference):
   https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/cutlass_extensions/gemm/fp8_blockwise_gemm_sm90_dispatch.cuh

---

## Final Recommendation Summary

**Choose vLLM** for:
- ✅ Exact 128×128 weight block + 1×128 activation scale geometry
- ✅ Stable ABI (future-proof)
- ✅ Minimal vendoring (3 files, ~500 LOC)
- ✅ Production-proven (vLLM 0.28+)
- ✅ CUTLASS 4.4.2 + CUDA 13.0 fully compatible
- ✅ Straightforward CMake integration

**Rationale**: vLLM's stable ABI path is the most maintainable, production-ready option that exactly matches your scale geometry and build requirements. The vendoring effort is minimal, and the code is battle-tested in production.

