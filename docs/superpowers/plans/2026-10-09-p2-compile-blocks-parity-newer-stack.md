# P2 follow-up — `compile-blocks` parity on a newer torch/diffusers stack

> **Tracking PR** — implementation for this plan lands on its own branch. Roadmap: #42.

Tracking: [#42](https://github.com/EfficientMoE/Omni-Infinity/issues/42) (P2 follow-up).
Builds on [P2](2026-10-07-p2-cudagraph-compile.md) and its measured `compile-blocks`
negative result. Refs: [sm120 gap analysis](../../sm120_gap_analysis.md) §gaps,
[t2v survey](../../t2v_optimization_survey.md) §3 compile/graphs.

## Objective

Determine, **with evidence**, whether a newer torch/diffusers stack closes the
`compile-blocks` golden-parity gap measured on torch 2.12.0+cu130 / diffusers 0.40.0
(video-latent `rms_rel ≈ 0.0726`; fails bitwise and allclose `2e-2`), **without**
disturbing the pinned default stack or the recorded goldens unless the evidence
justifies a bump.

Success is either:
- (a) a named newer stack on which `compile-blocks` passes the golden gate at a
  documented parity tier, with no CPU-suite regression, no timing regression, intact
  sm120 + opt-in fp8, and a provenance-clean golden re-baseline; or
- (b) a documented, reproducible **negative** result across the tested stacks that
  keeps `compile-blocks` opt-in and non-gating.

## Background (measured on the pinned stack)

- `compile_repeated_blocks(fullgraph=True, dynamic=True)` fused as intended: compute
  `507.70 → 471.05 ms`, kernels `2487 → 983`, 3 Dynamo graphs, no recompile storm.
  End-to-end wall still regressed to `0.939×` because variable host-AdaLN H2D dominates.
- Parity diverged: video-latent `rms_rel ≈ 0.0726177`, not bitwise, fails allclose at
  `1e-5 … 2e-2`; the eager control stayed bitwise on the golden GPU model.
- `TORCHINDUCTOR_FORCE_SAME_PRECISION=1` cut error ~`7.26% → ~3.89%` but did not close
  it; `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` did not help.
- In diffusers 0.40.0, `HostResidentAdaLN.forward` runs **inside** each compiled block,
  so per-block AdaLN materialization is inside the compiled region; only host-cache
  install and C5's outer skip decision are outside it.
- The specific op responsible for the divergence has **not** been localized yet.

## Constraints that shape the design

- The pinned stack (torch 2.12.0+cu130, diffusers 0.40.0) is the default and the
  goldens' provenance. Any upgrade is **quarantined to isolated envs** until proven.
- Goldens were recorded on the RTX PRO 6000 Blackwell **Server Edition (GPU 0)**.
  Parity comparisons MUST run on GPU 0 with matching provenance; timing uses GPU 4.
- sm120 (no WGMMA/tcgen05, 99 KB SMEM): a newer torch must still support sm120 and the
  opt-in fp8 path, or the upgrade is a non-starter for this repo.
- GPU etiquette: physical GPUs 1,2,3 are used by other sessions; pin exactly one GPU
  per run via `CUDA_VISIBLE_DEVICES`; never kill others' pids; leave the MPS server.
- Never silently replace goldens; a re-baseline is its own reviewed change.

## Approach (phased)

1. **Phase 0 — localize the divergence (in-stack, no upgrade).** Per-block and per-op
   parity probes vs eager to find the dominant error source (matmul accumulation,
   attention softmax, RMS/layernorm, AdaLN modulation). Use selective compilation
   (compile a subset of blocks / exclude a submodule) and `TORCH_LOGS` to attribute the
   error. Deliverable: the named culprit op(s) and whether any in-stack Inductor config
   (precision flags, autotune off, a specific lowering) closes parity on the pinned
   stack. If an in-stack mitigation closes it → record it and skip the upgrade.
2. **Phase 1 — isolated upgrade-matrix spike.** Build 1–2 candidate stacks (newer torch
   stable + nightly with a cu13x wheel, matching diffusers) in isolated venvs, each with
   a fresh `TORCHINDUCTOR_CACHE_DIR`. Run the committed compile-parity harness
   (`results/p2_phase1` path) on GPU 0. Record `rms_rel` + allclose tiers + an sm120/fp8
   smoke per stack. **ABORT GATE:** if no candidate stack closes parity to a documented
   tier, STOP → `compile-blocks` stays a documented negative result; write the matrix
   and return nonzero on the gate.
3. **Phase 2 — adopt (only if Phase 1 closes parity).** On the winning stack: run the
   full CPU suite, the GPU parity + timing ablation, and a provenance-clean golden
   re-baseline (its own reviewed change). Decision gate before proposing a pin bump:
   parity passes, no CPU-suite regression, a **net step-wall speedup** (not merely
   no-regression — the pinned stack was already `0.939×`) vs the pinned baseline on the
   same GPU, sm120 + fp8 intact.
4. **Phase 3 — document + registry.** Update the `compile-blocks` parity-tier doc and
   the plan/README status with the supported stack envelope; keep the registry opt
   gated and labelled with its validated stack.

## Phase 0 results (2026-10-10, pinned stack, GPU 0, bitwise eager control)

Probe: `benchmarks/compile_parity_probe.py` →
`results/p2_parity_followup/phase0/probe.json`; e2e verification via
`benchmarks/profile_denoise_step.py --profile compile-resident`.

**Divergence localized to Inductor codegen, not tracing or decompositions.**
On real step-1 golden-trajectory activations, `torch.compile` with
`backend=eager` and `backend=aot_eager` is **bitwise** on sampled blocks
(0/24/49); only `backend=inductor` diverges (rms_rel 4.9e-3 / 2.6e-3 /
4.1e-2). Piece bisect on a bitwise-verified block reconstruction: the
attention GEMMs (`attn_qkv`, `attn_out`), SDPA, and `HostResidentAdaLN`
each compile **bitwise** under Inductor (extern kernels). The error enters
in the fused pointwise/reduction codegen: dominant **q/k RMSNorm + rotary
region inside attention** (`attn_qknorm_rope` = 3.98e-2 of block 49's
4.06e-2), secondary RMSNorm+AdaLN-modulate chains (norm1mod up to 2.1e-2),
gated residual adds (≤5.6e-3), and the SwiGLU FeedForward piece (≤7.3e-4;
its linears were not isolated from the fused activation). Per-block error
grows with depth; chained within-forward divergence peaks at 1.32e-1 at
block 44 → 6.5e-2 at block 49, matching the e2e 7.26e-2 scale across 7
forwards.

**In-stack mitigation verdict: NEGATIVE — no tested config closes the
golden gate.** End-to-end video-latent `rms_rel`, all four measured on
GPU 0 with a fresh per-experiment `TORCHINDUCTOR_CACHE_DIR`
(`results/p2_parity_followup/phase0/e2e-*/compile-resident.json`; allclose
2e-2 all fail): default `0.0411`; `TORCHINDUCTOR_FORCE_SAME_PRECISION=1`
`0.0664`; `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` `0.0900`;
emulate+force_same `0.0900` (bit-identical to emulate alone). Prior
measurements — the P2 GPU-3 artifact (default `0.0726`,
`results/p2_phase1/parity/compile-resident.json`) and the P2 GPU-2
FSP probe note (`~0.0389`, no artifact) — differ in magnitude from the
GPU-0 runs: while the eager path is bitwise-reproducible on these cards,
the compiled path's parity error is **not stable across cards or runs**
(0.04–0.09 band), so no flag ordering ("FSP helps") survives
re-measurement. Block-level:
`pattern_matcher=False` 4.06e-2 and `max_fusion_size=1 + epilogue_fusion
=False` 3.69e-2 do not help; `emulate_precision_casts` cuts block 49 to
1.32e-3 (30×) yet is **worse** end to end — per-forward rounding deltas
amplify chaotically over the 50-block × 7-forward trajectory, so only
near-bitwise per-forward codegen can pass, and no config achieves it.
Excluding the culprit norm/rotary regions from compilation would forfeit
the fusion that is compile-blocks' entire perf rationale (wall already
0.939×). → Phase 0 does not close parity in-stack; proceed to Phase 1.

## Tasks

- [x] Phase 0 per-op/per-block divergence localization + in-stack mitigation probe, note appended here
- [ ] Isolated upgrade-matrix harness + `rms_rel`/allclose/sm120-fp8 results table (abort gate)
- [ ] (conditional) Full validation on a passing stack: CPU suite + GPU parity/timing + golden re-baseline
- [ ] (conditional) Pin-bump proposal behind a decision gate (separate reviewed change)
- [ ] `compile-blocks` parity-tier doc + supported-stack envelope updated

## Verification

Compile-parity harness `rms_rel` + allclose tiers on GPU 0 (provenance match); CPU
suite `pytest tests/ -m "not gpu and not weights"` on the candidate stack (no
regression vs the 430 passed / 13 deselected baseline); timing ablation showing a
**net step-wall speedup** (not merely non-regression) on GPU 4; sm120 + opt-in fp8
smoke on any candidate stack; golden re-baseline only behind its own review.

## Risks

- Newer torch may drop or alter sm120 / fp8 kernel support → upgrade non-starter; smoke
  this first in Phase 1 before investing.
- diffusers API drift (0.40 → newer) may change `MiniMaxH3Transformer3DModel`
  `_repeated_blocks` or the denoiser's packed-layout signature that the C5/graph
  wrappers depend on.
- cu13x wheel availability for candidate torch versions.
- Re-baselining goldens is provenance-sensitive; a wrong re-record silently weakens
  every parity gate → quarantine behind explicit review.
- A closed compile parity may still not yield an end-to-end win (host-AdaLN H2D
  dominated wall on the pinned stack); Phase 2 must show a net speedup, not just parity.
