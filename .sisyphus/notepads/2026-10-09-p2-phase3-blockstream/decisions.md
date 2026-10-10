# Decisions — P2 Phase 3: CUDA graphs under block-streaming

## [2026-10-09] Branch + base
- Worktree: /mnt/raid0nvme0/leyang/omni-wt/p2-phase3-blockstream-graphs
- Branch: plan/p2-phase3-blockstream-graphs, based off origin/plan/p2-cudagraph-compile
  @ 6ef518a (P2 tip — has cuda_graph.py, runner graph wiring, block-stream support).
- PR target: plan/p2-cudagraph-compile until P2 PR #44 merges, then retarget main.
  (Plan B depends on unmerged P2 code; it is NOT on main yet.)

## [2026-10-09] MANDATORY per-commit review policy
Every commit MUST pass review BEFORE push:
  git show HEAD | opencode run -m anthropic/claude-fable-5 "SINGLE-PASS review, no
  subagents. Strict reviewer for Omni-Infinity SM120 roadmap. Check: plan conformance,
  golden/parity-gate impact, CUDA-graph capture correctness (pointer stability,
  stream/event sync, NO CPU sync inside capture), 22 GiB envelope, type-suppression,
  deleted tests, accidental main-checkout edits. FIRST line APPROVE or REQUEST-CHANGES,
  then findings."
- REQUEST-CHANGES blocks: fix with follow-up commit, re-review. Push only after APPROVE.
- Record one-line verdict per commit in this file (.sisyphus is gitignored -> git add -f).
- Do NOT pre-record a commit's own verdict before its review.
- elm/gpt-5.6-sol is BUDGET-EXCEEDED (see issues.md) — do NOT use it. Review model is
  anthropic/claude-fable-5 (verified working 2026-10-09).

## [2026-10-09] Execution model
- Execute the plan doc 2026-10-09-p2-phase3-cudagraph-block-streaming.md phase by phase.
- Phase 0 is make-or-break: gate everything on its feasibility verdict.
- Timing (Phase 4) deferred until an idle GPU exists (see issues.md). Do not report
  contaminated timing as a result.

## Review verdicts
(record here, newest last)
- cd5c6b5 probe(p2-phase3): Phase 0 feasibility probe + decision note — APPROVE (claude-fable-5; minor non-blocking: soften 'exact config' phrasing re offload_to_disk_path; cross-step hazard note for Phase 1/3; toy≠H3 caveat)
- 6f937c0 feat(p2-phase3): arena block streamer (Phase 1) — APPROVE (claude-fable-5; non-blocking: multi-ModuleList execution-order assumption worth an assertion/doc before Phase 3; document "pinned host buffers are sole source of truth, never read weights outside the hook schedule"; single copy stream vs plan's 3-stream pool is a recorded deviation — D2H unnecessary since weights are read-only)
- 3f9c8eb feat(p2-phase3): runner wiring + scoped guard lift (Phase 2) — APPROVE (claude-fable-5; recorded deviation: guard lift lands in Phase 2 not Phase 3, general for h3-dense; Phase 4 parity/22GiB/net-win MUST gate before merge; non-blocking: CM interplay rests on the .to() guard — add a data_ptr spot-check to the Phase 4 harness; toy≠H3, real-model smoke is the parity test)
- f824041 feat(p2-phase3): capture glue + real-model smoke (Phase 3) — APPROVE (claude-fable-5; Namespace defect fixed by 299dc8c; carry-forwards: pipefail-safe gate runs; Phase 4 docs must say the supported envelope is "whole-forward capture or eager fallback — no sub-graph tier"; raw fallback counters recorded below)
- 299dc8c fix(p2-phase3): namespace-safe cuda-graph access — APPROVE (claude-fable-5; clean minimal fix)

## [2026-10-09] Phase 3 real-model smoke (GPU 0, shared, correctness only)
Raw telemetry: {"captures": 1, "replays": 3, "capture_failures": 0,
"graph_pool_bytes": 34542592, "capture_time_ms": 9366.87,
"warmup_calls": 3, "capture_warmup_calls": 2, "graphs": 1,
"generation": 0, "arena_bytes": 1541450752,
"pinned_host_bytes": 38536268800, "fallback_reasons":
{"warmup_not_done": 2, "shape_bucket_miss": 1}}
7 transformer forwards (8 steps): warmup, kwarg-reset warmup, warmup,
capture, 3 replays. Parity: bitwise=True rms_rel=0.0000 allclose=True.
Denoise-window peak 13.40 GiB (22 GiB budget). Wall 92.5 s on a ~70%
utilized shared GPU — NOT a timing result; Phase 4 ablation pending an
idle GPU. Whole-forward capture succeeded; sub-graph fallback NOT built
(not needed). Gate-run lesson: never pipe pytest through tail without
checking the exit code (pipefail).

## [2026-10-09] Phase 4 timing ablation (GPU 4, idle: 2 MiB/0% before+during+after)
block-stream median step wall 2657.1 ms vs graph-block-stream 1186.2 ms
=> NET WIN 2.24x. Replay median 1185.7 ms (3 replays); captures=1, fails=0;
fallbacks warmup_not_done:2 shape_bucket_miss:1 only. Both bitwise. Copy busy
2037.8->1127.8 ms; launch gap 106.2->0.8 ms; cold_start_forward_wall_ms 25.01s->0.77s (first probed step wall 1205.6 ms). Evidence
results/p2_phase3/{block-stream,graph-block-stream}.json (gitignored, medians
mirrored in the plan doc Phase 4 note). Bench harness peak (71.9/75.6 GiB)
includes the resident text encoder - NOT the envelope gate; envelope evidence
stays the Phase-3 smoke 13.40 GiB denoise-window peak.
- c399446 bench(p2-phase3): timing ablation 2.24x — APPROVE (claude-fable-5; all numbers independently recomputed; non-blocking F1 cold-start metric attribution + F2 7-vs-8 step note fixed in follow-up; F3 DEFAULT_RESULTS nit + F4 profile-gate test gap recorded, not blocking)
- 0ba572e docs(p2-phase3): cold-start attribution + step accounting — APPROVE (claude-fable-5; all numbers re-verified vs evidence; stylistic nit only)

## [2026-10-10] Main squash-integration (PR #75 retarget to main)
- f6083c4 [P2 Phase 3] squash-integration onto main (post-P5/P6/P7 drift) — APPROVE
  (claude-fable-5; scope verified: 17 Phase-3-owned files only, no P2 leftovers
  resurrected, NVFP4/C-caches/multi-GPU intact — 585 CPU tests pass; capture
  correctness re-verified: pointer-stable arena rebinding, event-only sync, no CPU
  sync in capture; scoped guard lift intact — to-disk + step_overlap stay rejected;
  22 GiB claims scoped honestly to the 13.40 GiB denoise-window peak; no
  type-suppression; one test replaced by the broader validation matrix, none
  deleted. Non-blocking: F1 arch-scoping lives in registry only, F2 getattr slop
  in smoke CLI, F3 38.5 GB pinned-host cost worth a README note eventually, F4
  untracked generated_latents.pt hygiene.)
- Integration-gate evidence: ruff clean; 585 passed CPU suite; GPU 0 real-model
  smoke bitwise=True rms_rel=0.0000 captures=1 capture_failures=0 (requires
  --first-frame tests/fixtures/ref.png — image-less FL2VA routes to the t2va
  graph and fails in prepare_condition_latents).
