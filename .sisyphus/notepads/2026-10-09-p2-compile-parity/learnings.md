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
