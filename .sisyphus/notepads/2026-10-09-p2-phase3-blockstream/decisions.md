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
