# P2 — CUDA-graph the denoise loop + regional torch.compile

Tracking: [#42](https://github.com/EfficientMoE/Omni-Infinity/issues/42) (P2).
Refs: [sm120 gap analysis](../../sm120_gap_analysis.md) §gaps,
[t2v survey](../../t2v_optimization_survey.md) §3 compile/graphs.

## Objective

Cut per-step CPU launch overhead in the MiniMax-H3 denoise loop via CUDA
graph capture and/or regional `torch.compile`, while preserving (a) the
block-streaming offload contract, (b) AdaLN host-cache materialization,
(c) golden parity gates. Survey evidence: regional compile ≈1.5× runtime on
comparable DiTs; FastUSP found kernel-launch overhead dominant on Blackwell.

## Constraints that shape the design

- **Block-streaming moves weights between/within steps** → naive whole-step
  graph capture freezes stale weight pointers. Graphs must either (i) capture
  only per-block compute with weights in stable arena buffers that streaming
  writes into (pointer-stable H2D), or (ii) apply only to the fully-resident
  profile (no `block-stream`).
- **C5 denoise cache skips evals data-dependently** → breaks fullgraph
  compile. Decision: compile/graph scope excludes the skip decision; the
  cached-step fast path bypasses the graph entirely.
- Reuse: MoE-Infinity `moe_infinity/serving/cuda_graph.py` (capture/replay
  coexisting with offload, 45+ fallback reasons — adopt its fallback-reason
  pattern) and BatchGen `batchgen/cuda_graph/graph_manager.py` (bucketing,
  warmup); MoE-Infinity `core/memory/stream_pool.h` for stream discipline.

## Approach (phased)

1. **Phase 0 — measure.** Nsight/CUDA-event breakdown of one denoise step:
   launch gap vs kernel time, per profile (resident vs block-stream).
   Abort criterion: if launch overhead <5 % in block-stream profile, demote
   P2 priority for that profile and target the resident profile only.
2. **Phase 1 — regional torch.compile.** Compile the repeated transformer
   block (`fullgraph=True, dynamic=True` scoped to the block module), not
   the pipeline. Keep AdaLN materialize + cache hooks outside. Gate behind
   registry optimization `compile-blocks`. Validate goldens (expect
   allclose-not-bitwise; document the parity tier it achieves).
3. **Phase 2 — CUDA graph the resident-profile step.** Static shapes per
   (resolution, frames) bucket; warmup→capture→replay manager modeled on the
   two donor implementations, with explicit fallback reasons (shape miss,
   cache-skip step, first N warmup steps).
4. **Phase 3 (stretch) — graphs under block-streaming.** Pointer-stable
   weight arenas sized per block group; prefetch writes into arenas on the
   copy stream; graph replays compute stream only. Only attempt if Phase 0
   shows launch overhead matters in this profile.

## Tasks

- [ ] Phase 0 profile + decision note appended to this plan
- [ ] `compile-blocks` registry opt + parity-tier doc + bench row
- [ ] Graph manager (vendored pattern) + resident-profile capture
- [ ] Fallback-reason telemetry in job logs
- [ ] Ablation rows: baseline / compile / graph / compile+graph × NFE
- [ ] Update issue #42; note interaction rules with C5 in caches docs

## Verification

Golden gates per profile (bitwise where the path claims bitwise; otherwise
`rtol=2e-2` tier documented); wall-clock s/NFE deltas in the ablation
harness; OOM check at 22 GiB envelope (graph pools count against budget).

## Risks

- Graph memory pools inflate peak VRAM → cap bucket count, reuse pools.
- diffusers hooks (group offload) inserting CPU sync points inside capture →
  Phase 3 may be infeasible; Phase 1+2 still deliver value.
- compile recompile storms on frame-count changes → `mark_dynamic` on the
  temporal dim, document supported shape envelope.
