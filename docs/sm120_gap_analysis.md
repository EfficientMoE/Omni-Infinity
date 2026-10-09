# SM120 optimization gap analysis — vLLM / SGLang vs Omni-Infinity

Research record (2026-02). Survey of the optimization features vLLM and SGLang
ship for omni-modal models, the SM120 (consumer/workstation Blackwell) kernel
ecosystem, and what Omni-Infinity lacks on this hardware. Produced from four
parallel research passes (local codebase audit, vLLM, SGLang, SM120 kernel
ecosystem). Companion survey: [text2video optimization survey](t2v_optimization_survey.md).

> **Verification caveat.** External issue/PR numbers below were collected by
> automated research and should be spot-verified before being cited elsewhere.
> The technical substance was cross-confirmed across independent passes.

## Hardware context

Bench machine: 6× **RTX PRO 6000 Blackwell** (CC **12.0 = sm120**, 96 GB each),
CUDA 13.1. SM120 vs datacenter Blackwell (SM100):

- **No WGMMA, no tcgen05/TMEM** — SM80-era `mma.sync.aligned.m16n8k16` only.
  FA3-style async producer/consumer pipelines cannot be ported.
- **99 KB SMEM** (vs 163 KB usable on SM80/90 class) — tile sizes must shrink.
- **TMA present but single-CTA only**; no multicast clusters.
- **Native FP8 and FP4 (NVFP4) tensor cores** — SM120's one *advantage*;
  4-bit compute is first-class here.

## What vLLM ships for omni models

- Qwen2.5/Qwen3-Omni **thinker only** in mainline (text out); the full
  talker + code2wav speech pipeline lives in **vllm-project/vllm-omni**
  (3-stage pipeline, per-stage CUDA graphs + batching, streaming audio,
  duplex mode; ~12× RTF vs HF transformers).
- Vision-encoder CUDA graphs in V1; audio-encoder graphs only in vllm-omni.
- Multimodal prefix caching covers vision only; audio features are
  re-encoded per request. No chunked prefill for multimodal embeddings.
- Quantization: FP8 W8A8 ✅, NVFP4/MXFP4 via vllm-omni per-stage nested
  quant; AWQ needs the `awq_marlin` path on SM120.

## What SGLang ships for omni models

- Qwen3-Omni thinker in mainline; talker/streaming-audio in the separate
  **sglang-omni** stack (talker model runner + streaming detokenizer).
- RadixAttention prefix caching works with multimodal inputs, but Qwen3-Omni
  is not in the breakable-CUDA-graph allowlist → multimodal encoder runs
  eager.
- On SM120 several FlashInfer paths are gated → Triton attention fallback;
  FA3 is Hopper-only. Multiple SM120 kernel issues open (FP8 scale-format
  quirks, GDN prefill races, NVFP4 serving support).

## SM120 kernel ecosystem (applies to any stack)

| Kernel | SM120 status |
|---|---|
| FlashAttention-2 (CuTe DSL port) | ✅ fwd/bwd/varlen merged; tiles tuned for 99 KB SMEM |
| FlashAttention-3 | ❌ never coming (needs WGMMA) |
| FlashAttention-4 (CuTe DSL) | ⚠️ fwd/bwd/varlen merged; **paged-KV and split-KV open** → blocked for paged LLM serving, but *usable for diffusion* (no KV cache). Compiles to ~FA2-class `mma.sync` perf (geomean ~1.2× FA2 fwd S512–32k) |
| FlashInfer fmha_v2 (HMMA) | ✅; 10–31 % over generic FA2 at S≥512 |
| cuDNN 9.18+ SDPA | ✅ native SM120 paths incl. d>128 (2-CTA MMA), MXFP8 (d=64) |
| CUTLASS FP8 `scaled_mm_sm120` (+128×128 blockwise) | ✅ in vLLM; FP8 grouped/MoE GEMM landed recently (CUTLASS 4.5+) |
| NVFP4 W4A4 GEMM | ✅ native CUTLASS/FlashInfer JIT paths |
| Marlin / GPTQ / AWQ | ✅ (AWQ via `awq_marlin`); often the fastest single-stream W4A16 path |
| FlashInfer top-k sampler | was hanging on SM120/121; fixed, default-off |

Build note: prebuilt wheels frequently miss `compute_120`; build with
`TORCH_CUDA_ARCH_LIST="12.0"`.

## Omni-Infinity: present vs missing on SM120

Already present (see [README](../README.md#key-features)): component
offload, AdaLN host cache (66.28→40.26 GB), bf16 block-streaming,
text-encoder streaming, cross-step prefetch (0.637/0.599 overlap), FP8
weight-only 128×128 block-scaled Triton GEMM, C1/C3/C5 caches,
VDN hybrid (dense→hybrid+fp8 **2.15×** per-NFE on this GPU vs upstream's
2.3–3.2× on H200/B200).

Gaps, ranked by expected impact:

1. **FP8 GEMM is weight-only dequant→bf16 MMA.** SM120 has native FP8
   tensor cores; the FP8 step delivered 1.23× vs upstream's ~1.7–1.9×.
   A true-FP8 compute path (CUTLASS `scaled_mm_sm120`-style or
   `torch._scaled_mm`) is the single largest kernel win.
   **Status (2026-10-08, PR #43):** landed as opt-in facade backends —
   `backend="scaled_mm"` (`torch._scaled_mm` w8a8, per-row scales) and
   `backend="cutlass_sm120"` (vendored vLLM CUTLASS blockwise kernel,
   lazy JIT via `OMO_CUTLASS_DIR`, 128×128 weight + 1×128 activation
   scales). Winner per shape (CUDA-event median, RTX PRO 6000):

   | shape (M,N,K) | bf16 | triton w8a16 | cutlass_sm120 | winner |
   |---|---:|---:|---:|---|
   | 1024,28672,5376 | 2.809 | 1.018 | 0.586 | cutlass (1.74× vs triton) |
   | 8192,28672,5376 | 8.037 | 8.224 | 5.130 | cutlass (1.60×) |
   | 1024,7168,5376 | 0.753 | 0.316 | 0.225 | cutlass (1.40×) |
   | 8192,7168,5376 | 2.135 | 1.992 | 2.191 | triton |
   | 32768,* | 6.7–27.0 | 7.9–32.8 | 8.7–34.3 | bf16 (act-quant bound) |

   w8a8 accuracy: cutlass rel≈0.026, scaled_mm rel≈0.037 vs the block
   reference — both coarser than the weight-only Triton path. Rung C
   boundary protection (`--fp8-protect-blocks`) moved the golden gate
   rms_rel only 0.2135 → 0.1880 (first:2,last:3) → 0.1719
   (first:5,last:10): the FP8 error is distributed across mid-blocks,
   so the `rtol=2e-2` gate still fails and FP8 stays opt-in.
2. **No CUDA graph capture.** The denoise loop is static-shape and repeated
   N times — ideal for graphs; every serving stack treats this as table
   stakes.
3. **No torch.compile.** Regional compile of repeated DiT blocks is the
   documented diffusers fast path (~1.5× on comparable DiTs).
4. **Attention backend pinned to `decomposed` because "no FA4".** FA4-fwd
   is now merged for SM120 and diffusion needs neither paged-KV nor
   split-KV; cuDNN 9.18+ SDPA and fmha_v2 are additional candidates.
   Also: upstream component log misdetects the card as sm100 — pin
   dispatch explicitly.
5. **No SM120 Triton tuning.** Generic BLOCK/num_stages/num_warps; 99 KB
   SMEM budget differs from both SM80 and SM90 — autotune for CC 12.0.
   **Status (2026-10-08, PR #43):** the fused FP8 GEMM now autotunes
   over BLOCK_M/N/K ∈ {32,64,128}, stages ∈ {2,3,4}, warps ∈ {4,8}
   statically pruned to the 99 KB budget (148 configs,
   `cache_results=True`). Best config almost everywhere:
   `BM=128,BN=128,BK=32,stages=3,warps=4` — BK=32 was inexpressible in
   the old {64,128} grid; baseline-normalized gain ≈1.2× at M=1024,
   neutral at M≥8192.
6. **NVFP4 unexploited.** FP4 weight(-only or W4A4) for the 33B DiT halves
   weight traffic again vs FP8; the pipeline is bandwidth/offload-bound,
   so this attacks the actual bottleneck. (Accuracy gating applies — H3
   is not FP8-native either; treat as opt-in like `fp8`.)
7. **C2 encoder prefix cache deferred** (vLLM's analogue covers vision
   only; audio caching is an open gap there too).
8. **Single-GPU, single-worker** — with 6× 96 GB available,
   encoder/denoise/decode stage pipelining and sequence parallelism are
   on the table (see the [T2V survey](t2v_optimization_survey.md)).

Not applicable to this codebase (autoregressive-LLM machinery): paged
attention, radix/KV prefix caching, continuous batching, speculative
decoding.

## Suggested priority

P1 native-FP8 GEMM → P2 CUDA-graph the denoise loop → P3 FA4-fwd /
fmha_v2 / cuDNN for window softmax → P4 SM120 Triton autotune →
P5 NVFP4 experiment → P6 C2 cache + multi-GPU pipelining.
