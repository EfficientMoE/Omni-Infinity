# Text-to-video serving optimization survey (SM120)

Research record (2026-02). Deep investigation of text2video / video-DiT
inference optimizations across the serving ecosystem, scoped to what is
portable to Omni-Infinity (MiniMax-H3 33B video+audio DiT, diffusers
ModularPipeline) on 6× RTX PRO 6000 Blackwell (sm120, 96 GB).
Companion: [SM120 gap analysis](sm120_gap_analysis.md).

> **Verification caveat.** Collected by automated research passes. Repo
> paths, issue numbers, and benchmark figures are leads, not gospel —
> verify before implementation. Items marked ⚠ are single-sourced.

## 1. Framework landscape

| Framework | Relevance | Headline techniques |
|---|---|---|
| **FastVideo** (hao-ai-lab) | **Highest** — ships FastH3 ⚠ (distilled MiniMax-H3) and NVFP4 checkpoints; claims best latency on RTX PRO 6000 | Video Sparse Attention (VSA, ~90 % sparsity tile-top-k), DMD distillation, TeaCache, Ulysses+Ring SP, NVFP4, fused QK-norm+RoPE / SwiGLU kernels, CUDA graphs, precomputed AdaLN modulation |
| **SGLang diffusion** (`multimodal_gen`) | High — modular, Wan/HunyuanVideo serving | Ulysses+Ring SP, cache-dit integration, distributed VAE, CFG parallel, CuTe-DSL fused norms |
| **vLLM-omni** | High — diffusion serving beside the Omni LLM stack | USP, cache-dit (dual-transformer aware), VAE patch parallelism, text-encoder TP |
| **TensorRT-LLM VisualGen** | Production reference | pluggable attention (TRTLLM/cuDNN/FlashInfer/FA4/CuTeDSL), ModelOpt FP8/NVFP4, SAGE + skip-softmax quantized attention, Attention2D CP, TeaCache+cache-dit, parallel VAE |
| **xDiT** | Parallelism reference | USP (Ulysses×Ring), PipeFusion (patch pipeline, NeurIPS'25), CFG parallel, parallel VAE; 6.12× on 8× PCIe L40 |
| **ParaAttention / cache-dit** | Lightweight diffusers add-ons | context parallel + FBCache; FLUX 6.75× on 4 GPUs combining FBCache+FP8+CP |

Notable precedent: upstream "AdaLN precompute for fixed schedules"
(FastVideo) parallels our AdaLN host cache — their variant precomputes the
*modulation outputs*, not just hosting the branch weights.

## 2. Attention (long-seq bidirectional DiT, no KV cache)

Tiered for SM120 (training-free first):

- **cuDNN 9.18+ SDPA** — native SM120 paths (incl. d>128 via 2-CTA MMA);
  ~1.1–1.3× over stock SDPA; 5-minute diffusers backend switch. Start here.
- **SageAttention 2/2++** (thu-ml) — INT8 QK + FP8 PV, training-free
  SDPA monkey-patch; 1.2–1.6× on CogVideoX/HunyuanVideo class. Runs on
  SM120 via Triton backend (CUDA fast paths are SM90-tuned).
  **SageAttention3** (FP4) targets datacenter Blackwell; SM120 support
  unconfirmed ⚠.
- **Radial Attention** (mit-han-lab) — static O(n log n) spatiotemporal
  energy-decay mask over FlashInfer block-sparse; ~1.9× training-free at
  default length, 3.7× at 4× length (needs LoRA). FlashInfer path works
  on SM120. Good fit for the VDN window-softmax branch comparison.
- **Sparse VideoGen 2** — k-means semantic permutation, 1.9–2.3×;
  Triton-only on SM120 (optimized kernels are H100-first) ⚠.
- **FA4 CuTe-DSL fwd (SM120)** — merged; for *non-causal, non-paged* full
  attention it is usable today, but on SM120 it compiles to ~FA2-class
  `mma.sync` perf (geomean ~1.2× vs FA2 at S512–32k; no tcgen05 speed
  path). Benchmark against cuDNN before adopting.
- **STA (sliding tile attention)** — archived by FastVideo in favor of
  VSA; ThunderKittens kernels are H100-only. Skip; evaluate **VSA**
  (tile-level top-k, has Triton fallback) instead.

## 3. Step caching and quantization

Caching (training-free):

- **TeaCache** (ali-vilab, CVPR'25) — timestep-embedding-modulated signal
  + polynomial rescaling; 1.8–2× on HunyuanVideo/CogVideoX class. Our C5
  denoise cache is this family; the polynomial-rescaled indicator is the
  upgrade.
- **FBCache / ParaAttention** — first-block residual gate; merged into
  diffusers as `apply_first_block_cache(..., threshold≈0.06–0.08)`.
  Simpler than TeaCache; needs higher threshold when combined with FP8.
- **cache-dit** (vipshop) — unified DBCache (Fn/Bn block caching) +
  TaylorSeer residual extrapolation; 40+ DiT families via BlockAdapter;
  the integration target used by SGLang/vLLM-omni/TRT-LLM. −18 % to
  −32 % latency at SSIM 0.94–0.96 on Wan2.2 ⚠. **Caveat: data-dependent
  skipping breaks torch.compile fullgraph** — choose cache tech and
  compile strategy together.
- PAB / Delta-DiT / AdaCache / FasterCache / MagCache — mostly paper-ware;
  superseded by the above.

Quantization on SM120:

- **SVDQuant / Nunchaku** (W4A4 NVFP4 + low-rank outlier branch) — ~3×
  vs BF16 and ~3.5× memory on FLUX-12B class, RTX 5090-validated; needs
  deepcompressor calibration + checkpoint repack. The highest-ceiling
  option; accuracy gate required (H3 wasn't FP8-native; expect the same
  fight at FP4).
- **FP8 true-compute** (per-tensor or MXFP8 32-block) — modest standalone
  (1.07–1.15×) but composable and low-risk; `torch._scaled_mm` /
  CUTLASS sm120 paths. **Boundary protection** pattern (keep first ~2 and
  last ~3 blocks BF16) reportedly closes most quality loss ⚠ — directly
  relevant to our failed `rtol=2e-2` FP8 gate.
- torchao float8 / diffusers quant backends for quick experiments.

Compile/graphs: regional compile (`compile_repeated_blocks(fullgraph=True,
dynamic=True)`) is the diffusers-blessed path (~1.5× runtime, 7× faster
cold start vs full-model compile); CUDA-graph the denoise step where
shapes are fixed (FastVideo/TRT-LLM practice).

## 4. Multi-GPU parallelism

- **USP (Ulysses×Ring)** — the default DiT scaling scheme (xDiT
  `xfuser`, arXiv:2405.07719). Ulysses comm shrinks with degree
  (`4O(p·hs)L/N`); Ring's is constant per layer (`2O(p·hs)L`) and hurts
  on PCIe. xDiT: 6.12× on 8× PCIe L40 (Ulysses-2×Ring-2×CFG-2);
  realistic DiT-loop expectation on a 6-GPU PCIe box ≈ 2–2.5×.
  Guidance: Ulysses degree ≤ 4 on PCIe, hybridize beyond that.
- **CFG parallel** — 2-way cond/uncond split; latent-space all-gather
  only, near-free, composable with SP. xDiT marks CFG ❎ for MiniMax-H3 ⚠
  (custom guidance) — verify against our pipeline.
- **PipeFusion** (arXiv:2405.14430) — patch-level pipeline using stale-KV
  temporal redundancy; lowest comm volume (`2O(p·hs)` per *step*), best on
  weak interconnects, but xDiT marks it unsupported for MiniMax-H3's
  attention pattern ⚠ and warmup steps hurt latency.
- **Disaggregated stage serving** (SGLang diffusion roles, vllm-omni) —
  encoder / denoiser / decoder on separate GPU groups with async handoff;
  ~same single-request latency as USP but 3–6× multi-request throughput
  via stage pipelining. Natural extension of our existing
  pipelined multi-prompt demo (`demo/pipeline.py`).
- **TP: avoid** — per-layer activation all-reduce scales with sequence
  length; measured worst scalability on DiT (flat to negative).
- **VAE parallelism** — becomes the bottleneck once the DiT scales.
  Variants: DistVAE overlapping-tile patch parallel; Wan-style spatial
  shard with conv halo exchange (exact, no blend seams); temporal
  chunking for causal video VAEs (LTX `DistributedVideoDecoder`).
- **NCCL/P2P on this hardware**: RTX PRO 6000 Blackwell *does* support
  PCIe P2P (GeForce 5090 does not — driver-gated), but needs: IOMMU +
  ACS disabled in BIOS, `nvidia_uvm uvm_disable_hmm=1`, and the
  `ForceP2P` driver registry override; known first-collective hangs on
  dual-socket boards otherwise. Tuning: `NCCL_P2P_LEVEL=SYS`,
  `NCCL_MIN_NCHANNELS=8` (~41 GB/s tuned vs ~22 GB/s default on Gen5
  x16). Fallback `NCCL_P2P_DISABLE=1` costs ~30 % collective throughput.
  Refs: github.com/voipmonitor/rtx6kpro (nccl-tuning, pcie-bandwidth),
  NVIDIA forums thread 365048, NVIDIA/nccl#1637.

Recommended staging for this repo: (A) USP 2×3 + spatial-shard VAE for
single-request latency → (C) hybrid disaggregated (encoder GPU0 /
denoiser GPUs1-4 USP / decoder GPU5) for multi-request serving.

## 5. Portability to Omni-Infinity — ranked

| # | Item | Expected gain | Effort | Notes |
|---|---|---|---|---|
| 1 | True-FP8 GEMM (native tensor cores, boundary-protected) | recover FP8 step toward ~1.7× class | M | replaces dequant-to-bf16 Triton path |
| 2 | Regional torch.compile + CUDA-graphed denoise step | ~1.3–1.5× | M | decide jointly with cache strategy |
| 3 | Attention backend bake-off: cuDNN SDPA vs FA4-fwd vs fmha_v2 (+SageAttention2 patch) | 1.1–1.5× on attention | S–M | fixes sm100 misdetection too |
| 4 | C5 upgrade → TeaCache-style indicator or cache-dit DBCache | 1.5–2× at calibrated quality | S–M | we already have the calibration harness |
| 5 | CFG parallel + stage pipelining across 6 GPUs | ~2×+ multi-GPU | M–L | first step beyond single-GPU design |
| 6 | USP sequence parallel (Ulysses-first on PCIe) | 3–4× attention scale-out | L | xDiT/para-attn as reference |
| 7 | NVFP4 weight path (SVDQuant-style) | up to ~3× ceiling | L | accuracy-gated opt-in, like `fp8` |
| 8 | VSA / Radial sparse attention on the softmax branch | 1.9–2.3× attention | L | compare against VDN linear branch first — may overlap |

Cross-framework speedup stacking claims (12–25× end-to-end) are
multiplicative marketing arithmetic — treat per-item numbers as ceilings
and re-measure per phase with the existing parity/ablation harness
(`benchmarks/ablation_vdn.py`, golden gates).
