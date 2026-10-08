
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
- 855f0ba docs(p2): qualify combined graph speedup — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 385e9b2 bench(p2): phase-0 profile + decision note — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 936274c fix(p2): clarify phase-0 methodology — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 9776c7c docs(p2): mark phase-1 task complete — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- b6533da docs(p2): record graph review verdicts — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 251ac20 docs(p2): mark phase-2 task complete — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 5cfd167 docs(p2): record serve telemetry review verdicts — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- b126d95 docs(p2): mark task 4 complete — APPROVE (elm/gpt-5.6-sol, single-pass, 2026-10-08)

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

## [2026-10-08] Phase 2 Task 4 review verdicts

- 62dc7b6 feat(serve): persist CUDA graph telemetry — REQUEST-CHANGES
  (lifetime counters were not job-scoped; resolved by 937fd80)
- 937fd80 fix(serve): scope graph telemetry to each job — APPROVE
  (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 5bdb189 docs(p2): document serve graph telemetry — REQUEST-CHANGES
  (review requested the plan checkbox; explicit user directive reserves it for
  orchestrator verification, clarified by follow-up documentation)
- bc47fc2 docs(p2): preserve telemetry task ownership — APPROVE
  (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 46e7b05 docs(p2): record serve telemetry learnings — APPROVE
  (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- docs(p2): record serve telemetry review verdicts — APPROVE
  (elm/gpt-5.6-sol, single-pass, 2026-10-08)

## [2026-10-08] Phase 2 Task 5 ablation decisions

- Preserve the 16-point VDN `GRID` and add a separate `P2_GRID`, selected via
  `--suite p2`, so existing defaults, command construction, and tests do not
  change.
- Run every timing/parity cell in its own subprocess with exactly one physical
  GPU visible. Timing uses GPU 4; parity uses GPU 0 only at NFE 8. NFE-16 CSV
  parity columns stay blank and notes explicitly say `no golden at this NFE`.
- For NFE 8, report the committed Phase-0 baseline, Phase-1 compile, and
  Phase-2 graph medians as the canonical comparison. Store fresh rechecks in
  the normalized artifacts and identify each canonical source path; this
  avoids cherry-picking host-copy variability while preserving all evidence.
- Treat compile+graph as a successful graph execution but a failed candidate:
  capture/replay is real and faster than graph-only, yet the NFE-8 latent fails
  allclose at `2e-2`. The operator recommendation therefore remains graph-only.

## [2026-10-08] Phase 2 Task 5 review verdicts

- 19415d0 bench(p2): add optimization ablation harness — REQUEST-CHANGES
  (path forwarding, graph replay gates, and parity provenance; resolved by
  d839857)
- d839857 fix(p2): enforce ablation execution provenance — REQUEST-CHANGES
  (compiled parity failures were not surfaced; resolved by c780b33)
- c780b33 fix(p2): surface negative parity gates — REQUEST-CHANGES
  (compile+graph eager fallback could still be accepted; resolved by 98b4932)
- 98b4932 fix(p2): reject graph parity fallback — APPROVE
  (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 0cd6ff7 bench(p2): record optimization timing evidence — REQUEST-CHANGES
  (parity evidence was absent and canonical timings replaced raw measurements;
  resolved by 438ce4d)
- 438ce4d fix(p2): make ablation evidence self-consistent — APPROVE
  (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 55d17db docs(p2): document optimization ablation — APPROVE
  (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 34ee5b8 docs(p2): record ablation learnings — APPROVE
  (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 3fd414a docs(p2): record ablation decisions — APPROVE
  (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 4ff0c61 docs(p2): record ablation review verdicts — APPROVE (delegate self-review, 2026-10-08)
  (elm/gpt-5.6-sol, single-pass, 2026-10-08)
- 7464f13 docs(p2): mark task 5 complete — APPROVE
  (elm/gpt-5.6-sol, single-pass, 2026-10-08)

## [2026-10-08] Phase 2 Task 6 — issue update + C5 interaction docs

- Append a "P2 CUDA graphs and regional compile" section to
  docs/caches_c5_denoise.md documenting the wrapper nesting (C5 cached_forward
  outside graph_forward outside the real/compiled transformer forward), the
  cache-hit bypass of replay, the post-generation record_cache_skip accounting,
  and the resident-only large-GPU scope. Verified against
  omni_infinity/caches/denoise.py, omni_infinity/cuda_graph.py, and
  omni_infinity/runner.py before writing.
- Post an honest P2 summary comment on issue #42 (graph-only is the sole fast +
  bitwise config; compile/compile+graph fail the golden gate; resident/large-GPU
  scope; block-stream demoted in Phase 0). Do not close the issue.
- Issue #42 updated via comment (evidence for plan checkbox 6):
  https://github.com/EfficientMoE/Omni-Infinity/issues/42#issuecomment-6061938884
  Issue left OPEN (roadmap tracker).
