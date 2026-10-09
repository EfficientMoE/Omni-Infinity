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

## Tasks

- [ ] Phase 0 per-op/per-block divergence localization + in-stack mitigation probe, note appended here
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
