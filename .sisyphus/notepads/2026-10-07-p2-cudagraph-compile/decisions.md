
## [2026-10-08] MANDATORY CODE REVIEW POLICY (user directive)
Every commit on plan/p2-cudagraph-compile MUST pass review BEFORE push:
1. Commit locally in worktree
2. Oracle subagent reviews exact diff (git show HEAD) — review model openai/gpt-5.2 where exposed
3. Blocking findings -> follow-up commits, re-reviewed
4. Push only after clean pass
Checks: plan conformance, golden/parity-gate impact, kernel numerics (scales/dtypes/SMEM), no main-checkout edits, no type-suppression, no deleted tests.
Record one-line verdict per commit below.

## Review verdicts
- 385e9b2 bench(p2): phase-0 profile + decision note — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 936274c fix(p2): clarify phase-0 methodology — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)

## [2026-10-08] REVIEW MODEL UPDATE (user directive)
openai/gpt-5.2 FAILS on this installation — do not use. Review model is now elm/gpt-5.6-sol.
Per-commit review command (run from worktree):
  git show HEAD | opencode run -m elm/gpt-5.6-sol "You are a strict code reviewer for the Omni-Infinity SM120 roadmap. Review this commit diff for: plan conformance, golden/parity-gate impact, kernel numerics (scales, dtypes, SMEM budgets), accidental main-checkout edits, type-suppression, deleted tests. Verdict line first (APPROVE or REQUEST-CHANGES), then numbered findings."
REQUEST-CHANGES is blocking: fix, re-review, then push. Everything else in the policy stands.
- 02390bc docs(p2): checkbox + verdicts — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- d206c5a feat(p2): add regional compile runner support — REQUEST-CHANGES (registry reachability and parity evidence; resolved by ebf11a9, c029edc, and 784acdf)
- ebf11a9 feat(registry): register compile blocks optimization — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 570acfa bench(p2): measure resident regional compile — REQUEST-CHANGES (hard gate and provenance; resolved by c029edc and 784acdf)
- c029edc fix(p2): enforce compile parity provenance — REQUEST-CHANGES (bitwise exact-profile gate and workload provenance; resolved by 784acdf)
- 784acdf fix(p2): tighten compile parity gate — APPROVE (elm/gpt-5.6-sol, single-pass re-review, 2026-10-08)
- d1aa441 docs(p2): document compile blocks outcome — REQUEST-CHANGES (AdaLN scope clarification resolved by 7139da5; checkbox intentionally remains orchestrator-owned)
- 7139da5 docs(p2): clarify compiled AdaLN scope — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 41bd549 docs(p2): record phase 1 compile learnings — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 7b3a320 docs(p2): record phase 1 review verdicts — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
