
## [2026-10-08] MANDATORY CODE REVIEW POLICY (user directive)
Every commit on plan/p2-cudagraph-compile MUST pass review BEFORE push:
1. Commit locally in worktree
2. Oracle subagent reviews exact diff (git show HEAD) — review model openai/gpt-5.2 where exposed
3. Blocking findings -> follow-up commits, re-reviewed
4. Push only after clean pass
Checks: plan conformance, golden/parity-gate impact, kernel numerics (scales/dtypes/SMEM), no main-checkout edits, no type-suppression, no deleted tests.
Record one-line verdict per commit below.

## Review verdicts
- e26cb7e feat(p2): add resident CUDA graph manager — REQUEST-CHANGES (per-bucket pool safety, disabled CUDA initialization, and side-stream warmup; resolved by follow-up commit)
- 41c0bc5 fix(p2): harden graph capture lifecycle — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- fd5d8f4 feat(p2): pin AdaLN graph copy sources — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 4dc30a8 feat(registry): register CUDA graph optimization — REQUEST-CHANGES (runner reachability; resolved by follow-up runner integration commit)
- 1a38447 feat(p2): integrate resident CUDA graph execution — REQUEST-CHANGES (component-offload pointer lifetime; parity evidence lands with benchmark artifacts; serve telemetry remains Task 4)
- 5c6009f fix(p2): invalidate graphs after component offload — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 48fcf50 bench(p2): add resident graph measurement harness — REQUEST-CHANGES (parity/timing could pass on eager fallback; resolved by strict graph execution gate)
- a67f2aa fix(p2): require graph execution in benchmark gates — REQUEST-CHANGES (partial replay failure still passed; resolved by live-graph and fallback gate)
- fb77301 fix(p2): reject degraded graph benchmark runs — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 4a1877e bench(p2): record resident CUDA graph evidence — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 5666ab7 docs(p2): document resident graph outcome — REQUEST-CHANGES (clarify combined graph+pinned-AdaLN speedup; plan checkbox remains orchestrator-owned by explicit user directive)
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

## [2026-10-08] Phase 2 graph decisions

- Scope CUDA graphs to `h3-dense` resident transformer forwards. Reject
  block-stream composition because streamed weights are not pointer-stable;
  keep compile+graph allowed but quarantine and report capture failures.
- Key buckets by effective `(height, width, frames)` video shape, use one graph
  pool per live bucket, and cap live graphs at two with LRU eviction to bound
  VRAM. Separate pools permit arbitrary LRU replay order; a shared allocator
  pool would require replaying graphs in capture order.
- Pin AdaLN host weight/bias/scale tensors only when `cuda_graph=True`, require
  20% host-memory headroom before pinning, and preserve graph operation when no
  host cache is installed.
- Keep C5 outside the graph wrapper. A cache-hit step bypasses replay; generation
  completion forwards the C5 skipped count to the manager's cache-skip reason.
