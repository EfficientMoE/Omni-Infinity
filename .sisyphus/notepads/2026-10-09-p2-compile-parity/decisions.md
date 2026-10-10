# Decisions — P2 follow-up: compile-blocks parity on a newer stack

## [2026-10-09] Branch + base
- Worktree: /mnt/raid0nvme0/leyang/omni-wt/p2-compile-parity
- Branch: plan/p2-compile-parity-newer-stack, based off origin/plan/p2-cudagraph-compile
  @ 6ef518a (P2 tip — has the `compile-blocks` registry opt + the Phase-1 compile parity
  harness; NOT on main).
- PR target: plan/p2-cudagraph-compile until P2 PR #44 merges, then retarget main.

## [2026-10-09] MANDATORY per-commit review policy
Every commit MUST pass review BEFORE push:
  git show HEAD | opencode run -m anthropic/claude-fable-5 "SINGLE-PASS review, no
  subagents. Strict reviewer for Omni-Infinity SM120 roadmap. Check: plan conformance,
  golden/parity-gate impact, numerics correctness, environment isolation (no edits to the
  shared pinned env), type-suppression, deleted tests, accidental main-checkout edits.
  FIRST line APPROVE or REQUEST-CHANGES, then findings."
- REQUEST-CHANGES blocks: fix with follow-up commit, re-review. Push only after APPROVE.
- Record one-line verdict per commit here (.sisyphus is gitignored -> git add -f).
- Do NOT pre-record a commit's own verdict. elm/gpt-5.6-sol is BUDGET-EXCEEDED — do NOT
  use it; review model is anthropic/claude-fable-5 (verified working 2026-10-09).

## [2026-10-09] Execution model (per plan doc)
- Phase 0 FIRST (in-stack, no upgrade): localize the divergence (per-op/per-block parity
  probes), try in-stack mitigations. If an in-stack fix closes parity -> record + skip upgrade.
- Phase 1: isolated upgrade-matrix spike in SEPARATE venvs (see issues.md isolation rule),
  parity harness on GPU 0. ABORT GATE: no stack closes parity -> STOP, documented negative.
- Phase 2 (conditional): full validation on a passing stack incl. provenance-clean golden
  re-baseline behind its OWN review gate; decision gate requires a NET speedup, not just
  parity (pinned stack was already 0.939x).
- Phase 3: docs + registry supported-stack envelope.

## Review verdicts
(record here, newest last)
