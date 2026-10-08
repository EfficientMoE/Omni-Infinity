# P2 — CUDA-graph the denoise loop + regional torch.compile

> **Tracking PR** — implementation for this plan lands on this branch. Roadmap: #42.

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

- [x] Phase 0 profile + decision note appended to this plan
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

## Phase 0 decision note (2026-10-08)

Measured on physical GPU 4 (RTX PRO 6000 Blackwell Max-Q, 96 GB) with
PyTorch 2.12.0+cu130 and diffusers 0.40.0. Both runs used the store-backed
transformer and AdaLN host cache at 256p, 120 requested frames, seed 0, and
8 requested scheduler steps. This scheduler executes 7 transformer forwards;
the first forward was discarded and the table reports medians over the
remaining 6. Raw records are in `results/p2_phase0/{resident,block-stream}.json`.

| profile | median wall (ms) | median compute (ms) | median copy (ms) | median gap (ms) | median per-step gap | median kernels/step |
|---|---:|---:|---:|---:|---:|---:|
| resident | 1483.38 | 507.70 | 955.75 | 18.66 | 1.31% | 2487 |
| block-stream | 2951.59 | 508.56 | 2360.30 | 72.91 | 2.41% | 2487 |

Method: `torch.profiler` CPU+CUDA tracing with a `record_function` range
entered/exited by transformer forward hooks. Per-stream CUDA annotation
windows bound each step. Compute busy is the union of kernel intervals; copy
busy is the separately reported union of memcpy intervals (block-stream uses
copy stream 13 in addition to compute/default stream 7). `launch gap` is the
step envelope minus the union of compute and copy intervals, so serialized or
overlapped weight H2D is not mislabeled as CPU launch overhead. It is a
conservative upper bound because any remaining dependency, synchronization,
allocator, or profiler idle time is included. The percentage is the median of
per-step percentages, not a ratio of independently computed medians. Per-step
CUDA events were also recorded as a wall-time fallback and agree with the
profiler median within 0.22 ms. Peak allocated memory was 71.89 GiB resident
and 13.37 GiB block-stream (including the discarded warmup forward).

**Verdict.** Block-stream launch overhead is **2.41%**, below the 5% decision
threshold. Demote P2 for block-stream: Phase 1 regional compile and Phase 2
graph experiments are resident-profile-only, and Phase 3 pointer-stable graph
capture under block streaming is dropped. The resident copy-adjusted launch
gap is also only 1.31%, which bounds the likely pure launch-overhead win; keep
resident as the sole target because its transformer pointers are stable, but
require the later ablation to justify continuation rather than assuming a
graph speedup. P2 therefore becomes a large-GPU-only optimization: the
resident profile peaks at 71.89 GiB and does not satisfy the separate 22 GiB
memory-constrained serving envelope.

Measurement gotchas: FL2VA requires `tests/fixtures/ref.png`; an image-less
call fails before denoising. The merged local H3 snapshot keeps the transformer
config under `FL2VA/transformer`, so the benchmark applies a process-local
config lookup adapter while leaving the checkpoint and runtime sources
unchanged. The video VAE rounds 120 requested frames to 124 internally.
