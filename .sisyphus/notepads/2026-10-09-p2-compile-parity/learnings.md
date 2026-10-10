# Learnings — P2 compile-parity follow-up (seed)

## Background: compile-blocks measured NEGATIVE on the pinned stack (P2 Phase 1)
- `compile_repeated_blocks(fullgraph=True, dynamic=True)` fused: compute 507.70->471.05 ms,
  kernels 2487->983, 3 Dynamo graphs, no recompile storm. But wall regressed to 0.939x
  (host-AdaLN H2D dominates).
- Parity diverged: video-latent rms_rel ~= 0.0726177, NOT bitwise, fails allclose 1e-5..2e-2;
  eager control stayed bitwise on the golden GPU model.
- `TORCHINDUCTOR_FORCE_SAME_PRECISION=1` cut error ~7.26% -> ~3.89% (did NOT close);
  `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` did not help.
- In diffusers 0.40.0, `HostResidentAdaLN.forward` runs INSIDE each compiled block, so
  per-block AdaLN materialization is inside the compiled region; only host-cache install +
  C5's outer skip are outside it.
- The specific divergent op has NOT been localized — that is Phase 0's job.

## Integration points (base branch)
- `omni_infinity/registry.py`: `compile-blocks` OptimizationSpec (~L124).
- `omni_infinity/runner.py`: `_compile_repeated_transformer_blocks` (~L119),
  `_validate_compile_blocks` (~L98). compile applied at model load.
- Phase-1 compile parity harness + artifacts under `results/p2_phase1/` (committed JSONs
  `compile-resident.json`, `parity/{resident,compile-resident}.json`); CUDA-event timing in
  `benchmarks/`. Reuse this harness for the upgrade-matrix runs.

## Env (pinned / default — do NOT mutate for Phase 1; use isolated envs)
- python=/home/leyang/anaconda3/bin/python (3.13), torch 2.12.0+cu130, diffusers 0.40.0.
- HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; PYTHONPATH=<worktree>.
- Checkpoint snapshot 42ed227ee7df40d41602854ae760620d6eb651fe; store
  /mnt/raid0nvme0/leyang/h3-store-v2; goldens tests/fixtures/goldens/fl2va_goldens.pt
  (+ tests/fixtures/ref.png). 256p, 120 req/124 eff frames, seed 0, 8 steps.
- Parity provenance: goldens recorded on RTX PRO 6000 Server Edition (GPU 0). Parity on
  GPU 0; timing on GPU 4. CPU test baseline: 430 passed, 13 deselected.

## [2026-10-10] Phase 0 — divergence localized (probe.json, GPU 0, bitwise control)
- Dynamo tracing + AOT decomps BITWISE (backend=eager/aot_eager); divergence is 100%
  Inductor codegen of fused pointwise/reduction chains.
- Named culprit: q/k RMSNorm + rotary fused region in attention (attn_qknorm_rope:
  3.98e-2 of block-49's 4.06e-2); secondary: RMSNorm+AdaLN modulate chains, gated
  residual adds. GEMMs + SDPA + HostResidentAdaLN all BITWISE under Inductor.
- Per-block error grows with depth (block0 4.9e-3 -> block49 4.1e-2); chained
  within-forward peaks 1.2e-1 @ block 42.
- e2e config sweep ALL FAIL allclose 2e-2: default .0726 / FSP .0389 / EMULATE .0900 /
  EMULATE+FSP .0900 (bit-identical to emulate). emulate is 30x BETTER per-block but
  WORSE e2e -> chaotic trajectory amplification; only near-bitwise per-forward passes.
- Phase 1 bar: a newer stack passes ONLY if its Inductor emits bitwise-or-near-bitwise
  fused norm/rotary kernels vs eager bf16 op-by-op rounding. Low prior; measure anyway.

## [2026-10-10] Phase 0 corrections (review 377f3a1) + GPU-0 uniform e2e sweep
- CORRECTION: ff piece (whole SwiGLU FeedForward incl. fused activation) is NOT bitwise
  (<=7.3e-4); only attn_qkv/attn_out GEMMs, SDPA, and HostResidentAdaLN are bitwise.
  ff linears were never isolated. Chained within-forward peak is 1.32e-1 @ block 43.
- GPU-0 fresh-cache e2e sweep (results/p2_parity_followup/phase0/e2e-*): default 0.0411,
  FSP 0.0664, emulate 0.0900, emulate+FSP 0.0900 (bit-identical) — ALL fail 2e-2.
- Prior values (GPU-3 artifact default 0.0726; P2 GPU-2 FSP note ~0.0389, no artifact)
  do NOT reproduce on GPU 0: compiled-path parity error is unstable across cards/runs
  (0.04-0.09 band) while eager stays bitwise. "FSP helps" was noise, not signal.
- Phase-1 implication unchanged: pass requires near-bitwise per-forward Inductor codegen.

## [2026-10-10] Phase 1 — upgrade matrix NEGATIVE, abort gate honored
- Isolated venvs /tmp/opencode/env-{t2141,t216n} (pinned anaconda3 untouched).
- diffusers 0.41.0 BREAKS the pipeline on both torches: decode.video dtype mismatch
  (float vs bf16) in MiniMaxH3VideoDecodeStep -> both rows pinned d0.40.0.
- sm120 + opt-in fp8 smoke PASS on both stacks (arch list has sm_120; fp8 ~2.4e-3).
- torch 2.14.1: eager-vs-golden 0.0471 (eager drift!), compiled-vs-eager 0.0745 FAIL.
- torch 2.16.0.dev20261009: eager-vs-golden 0.1008, compiled-vs-eager 0.0998 FAIL.
- Eager run-to-run bitwise on both stacks (determinism control OK).
- Conclusion: Inductor fused-codegen divergence persists on newer stacks (0.04-0.10
  band); nightly worst. Negative result stands; compile-blocks stays opt-in non-gating.
