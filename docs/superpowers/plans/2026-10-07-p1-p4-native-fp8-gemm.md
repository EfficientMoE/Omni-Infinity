# P1+P4 — Native FP8 tensor-core GEMM + SM120 Triton autotune

> **Tracking PR** — implementation for this plan lands on this branch. Roadmap: #42.

Tracking: [#42](https://github.com/EfficientMoE/Omni-Infinity/issues/42) (P1, P4).
Refs: [sm120 gap analysis](../../sm120_gap_analysis.md), reuse scan in issue #42 comment.

## Objective

Replace the FP8 weight-only dequant→bf16-MMA Triton GEMM
(`omni_infinity/kernels/_fused_fp8_gemm.py`) with a path that runs on SM120's
native FP8 tensor cores, and autotune remaining Triton kernels for CC 12.0
(99 KB SMEM). Target: recover the FP8 step from measured 1.23× toward the
~1.7–1.9× upstream class, without relaxing the parity/QA contract.

## Approach (3 rungs, each independently landable)

1. **Rung A — `torch._scaled_mm` baseline (cheap, pure-torch).**
   Add `fused_fp8_gemm` impl using `torch._scaled_mm` with per-row scales
   (its native granularity); keep 128×128 block path on Triton. Registers in
   the existing op facade as `backend="scaled_mm"`. Establishes the
   true-FP8-compute speed ceiling with zero build complexity.
2. **Rung B — CUTLASS sm120 blockwise kernel (the real target).**
   Vendor BatchGen's `batchgen_kernels/src/moe/fp8_blockwise/fp8_blockwise_gemm.cu`
   (CuTe persistent, per-block scales — matches our 128×128 format) +
   `fp8_blockwise_ops.cu` quant helpers. Build via BatchGen's proven
   `setup.py`/`_jit_registry.py` pattern (`BUILD_ARCH=sm120`,
   `-gencode arch=compute_120,code=sm_120`). This is the README's
   moe-kernels extraction trigger — keep the impl behind
   `omni_infinity/kernels/_impls/` so `git mv` extraction stays trivial.
   Also evaluate vLLM `scaled_mm_blockwise_sm120_fp8.cu` as alternative donor.
3. **Rung C — accuracy recovery via boundary protection.**
   FP8 currently fails `allclose(rtol=2e-2)` (block rms_rel 0.2135). Add a
   `--fp8-protect-blocks first:2,last:3` option keeping entry/exit transformer
   blocks in bf16 (survey: residual pathways self-correct mid-blocks).
   Re-run the QA harness; if the gate passes, FP8 can graduate from
   "memory tradeoff" to a default-candidate.

**P4 autotune:** wrap `_fused_fp8_gemm` (and any kernel kept on Triton) with
`triton.autotune` over a grid modeled on MoE-Infinity `fused_ffn.py` /
BatchGen `v4_cache_utils.py` (BLOCK_M/N/K ∈ {32,64,128}, num_stages ∈ {2,3,4},
num_warps ∈ {4,8}), constrained to 99 KB SMEM; cache best configs per (M,N,K).

## Tasks

- [x] Rung A impl + registry entry + unit parity vs `_reference.py`
- [x] Microbench A vs current Triton path (N=28672 shapes from inc 8)
- [ ] Vendor Rung B kernel + build plumbing (optional extra, wheel stays pure-python)
- [ ] Rung B parity + microbench; pick winner per shape
- [ ] Rung C boundary protection flag + QA gate rerun (256p/120f goldens)
- [ ] P4 autotune sweep; record configs in docs/ablation table
- [ ] Update `docs/sm120_gap_analysis.md` status + issue #42 checkboxes

## Verification

- `pytest tests/ -m "not gpu"` clean; GPU QA: golden-latent gates
  (`rms_rel`, `allclose(rtol=2e-2)`) per README inc-8 methodology.
- CUDA-event microbench: tokens/s + peak MB at M∈{1k,8k,32k} per shape.
- No regression on the bf16 default path (bitwise goldens untouched).

## Risks

- `_scaled_mm` per-row scaling may worsen accuracy vs 128×128 blocks → keep
  both granularities A/B'd (per-row path already exists for comparison).
- CUTLASS pin: BatchGen kernel may need CUTLASS ≥4.x — isolate in optional
  build extra; never a hard dep.
- H3 is not FP8-native: even perfect kernels stay opt-in unless Rung C passes.
