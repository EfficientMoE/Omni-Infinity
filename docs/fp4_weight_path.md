# FP4 (MXFP4) weight path — QA, memory, and the Phase-3 decision

Plan: [P5 — NVFP4 weight path](superpowers/plans/2026-10-07-p5-nvfp4.md),
tracking [#42](https://github.com/EfficientMoE/Omni-Infinity/issues/42),
implementation PR
[#46](https://github.com/EfficientMoE/Omni-Infinity/pull/46). Same QA
contract as the FP8 inc-8 harness: golden latents (256p/120 frames, seed 0,
8 steps) with explicit pass/fail vs `allclose(rtol=2e-2, atol=2e-2)`, plus
memory deltas.

FP4 follows the FP8 messaging exactly: H3 ships bf16 with no quantization
config, so post-hoc 4-bit weights are an **opt-in memory/bandwidth
tradeoff**, never the accuracy-preserving default (that remains bf16
block-streaming, which is bitwise).

## How it is measured

```bash
CUDA_VISIBLE_DEVICES=5 HF_HOME=... HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/fl2va_smoke.py --prompt "a red ball bouncing" \
    --seed 0 --steps 8 --resolution 256p --frames 120 \
    --first-frame tests/fixtures/ref.png \
    --checkpoint <local-H3-snapshot-dir> --offload \
    --store-dir <moe-store> --store-components transformer,vae,audio_vae \
    --adaln-host-cache --transformer-fp4 --fp4-scale mxfp4 \
    --max-vram 90GiB --vram-window full \
    --goldens tests/fixtures/goldens/fl2va_goldens.pt
```

The resident-weight figures below are the transformer's GPU-movable state
bytes (parameters + buffers; the host-resident AdaLN cache is a plain
attribute invisible to `Module.to` and is excluded), the same accounting
behind the inc-3/inc-4 numbers. Microbench latencies are from
`benchmarks/bench_mxfp4_gemm.py` (RTX PRO 6000 Blackwell, SM120,
torch 2.12.0+cu130).

## Transformer resident weights (the deliverable)

| profile | resident weights | vs bf16 | vs fp8 |
|---|---:|---:|---:|
| bf16, full transformer (inc 2) | 66.28 GB | 1.00× | — |
| bf16 + AdaLN host cache (inc 3) | 40.26 GB | 0.61× | — |
| fp8 + AdaLN host cache (inc 4) | 20.19 GB | 0.30× | 1.00× |
| **fp4 (mxfp4) + AdaLN host cache (P5 phase 1)** | **10.79 GB** | **0.16×** | **0.53×** |

FP4 lands below the plan's ~14 GB weights-resident target: packed E2M1
pairs are a quarter of bf16 and the per-32 E8M0 scales add 1/16 of the
packed payload (`weight bytes = 0.53× fp8`, matching the microbench:
N=28672/K=5376 is 308.3 / 154.1 / 81.9 MB for bf16 / fp8 / fp4+scales).

End-to-end peaks for context (97.9 GB dev box, 90 GiB emulated budget, the
64 GB text encoder stays resident because it fits): full pipeline-window
`max_memory_allocated` 82.96 GiB, generate wall-clock 50.8 s. Under a real
22 GiB envelope the streamed bf16 profile (9.96 GiB peak, bitwise) remains
the reference configuration.

## Quality ladder (golden rms_rel, 256p/120f, seed 0, 8 steps)

| rung | rms_rel vs goldens | `allclose(rtol=2e-2, atol=2e-2)` |
|---|---:|:---:|
| bf16 (resident or block-streamed) | 0.0000 (bitwise) | PASS |
| fp8, block 128×128 scales | 0.2135 | FAIL |
| fp8, per-row scales | 0.2373 | FAIL |
| **fp4, mxfp4 (per-32 E8M0)** | **0.4042** | **FAIL** |
| fp4 + low-rank outlier branch (phase 3) | not built — see decision | — |

The FP4 gate failure was expected and is documented exactly like FP8's:
H3 is not FP4-native, so the value is memory/bandwidth, not quality.

## Latency (microbench, median CUDA-event ms)

| shape (M, N, K) | bf16 | fused_fp8 | fused_mxfp4 |
|---|---:|---:|---:|
| 4096, 7168, 5376 | 1.022 | 1.405 | 5.912 |
| 8192, 7168, 5376 | 6.800 | 5.254 | 14.030 |
| 4096, 28672, 5376 | 6.506 | 14.233 | 28.676 |
| 8192, 28672, 5376 | 18.977 | 24.513 | 53.125 |

The Triton MXFP4 dequant-in-kernel GEMM is 3–6× slower than bf16 cuBLAS
(LUT decode + even/odd `tl.dot` split dominates), so phase 1 does not meet
the plan's latency-parity hope. Marlin W4A16 is unusable on SM120 here
(groupsize=128 rejects H3's K dims; groupsize=-1 hits
`cudaErrorIllegalInstruction`), leaving the BatchGen native FP4 SM120
kernel (phase 2, needs P1's build plumbing) as the latency path.

## Phase-3 decision: deferred

Phase 3 (SVDQuant-style low-rank BF16 outlier branch) is **deferred, not
scheduled**:

1. **The gate is implausible for outlier handling.** The measured
   ladder shows even FP8 — with twice the bits per weight — lands 0.21
   rms_rel, ~10× above the 2e-2 gate. An outlier branch on FP4 would at
   best pull 0.4042 toward the FP8 error class, which itself misses the
   gate by an order of magnitude, so gate-passing FP4 via outlier
   correction alone is not a realistic outcome (though not formally
   disproven). The dominant failure mode is H3's quantization-naive
   bf16 weights compounding across 50 blocks, not a handful of outlier
   channels.
2. **The memory goal is already met.** 10.79 GB resident beats the ~14 GB
   target without a calibration pipeline (deepcompressor) or a second
   weight branch that would claw back part of the savings.
3. **Latency, not accuracy, is the binding constraint.** The next unit of
   effort goes to phase 2 (native FP4 tensor-core GEMM at H3 shapes) if
   fp4 streaming profiles need it; an outlier branch would make latency
   worse (extra bf16 GEMM per Linear).

Revisit the decision only if a use case accepts a looser latent tolerance
(≥ ~0.2 rms_rel class) where fp4+outlier vs fp8 at equal bytes becomes the
relevant comparison.

## Store side (moe-store)

The FP4 group format (packed E2M1 member + adjacent `<weight>_scales`
E8M0 member, `fp4_format_version` 1) and the `--quantize-experts fp4`
converter flag land in [moe-store#17](https://github.com/EfficientMoE/moe-store/pull/17)
(branch `feat/quantize-experts-fp4`, commit a698f80, kept
numerics-identical to
`omni_infinity/kernels/_quant.py`). A store→`ScaledFp4Linear` direct-read
path in Omni-Infinity (loading packed payloads without re-quantizing
bf16 at startup) is deliberately left to a later phase.
