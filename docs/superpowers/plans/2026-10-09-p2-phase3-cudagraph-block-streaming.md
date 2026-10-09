# P2 Phase 3 — CUDA graphs under block-streaming

> **Tracking PR** — implementation for this plan lands on its own branch. Roadmap: #42.

Tracking: [#42](https://github.com/EfficientMoE/Omni-Infinity/issues/42) (P2 Phase 3).
Resurrects the Phase 3 stretch goal demoted in [P2](2026-10-07-p2-cudagraph-compile.md)
under a full-implementation mandate. Refs: [sm120 gap analysis](../../sm120_gap_analysis.md)
§gaps, [t2v survey](../../t2v_optimization_survey.md) §3; MoE-Infinity
`core/memory/stream_pool.h` for stream discipline.

## Objective

Enable CUDA-graph capture/replay for the MiniMax-H3 denoise step under the
**block-streaming** offload profile (not just the resident profile), within the 22 GiB
memory-constrained serving envelope, with **bitwise** golden parity.

Success = `block-stream + cuda-graph` runs end-to-end, replays **bitwise** against the
golden (GPU 0), peaks **≤ 22 GiB** (arenas + graph pools counted), and shows a **net**
step-wall win vs the block-stream baseline.

## Background / honest value proposition

- P2 resident graphs: `1.535×` step wall, bitwise — but `71.89 GiB` peak, large-GPU only.
- Phase 0 measured block-stream launch gap at only **2.41%** (< 5% threshold), so
  launch-overhead reduction **alone will not justify this**. The real P2 graph win came
  from pinning the `24.23 GiB` AdaLN host-cache H2D sources (copy `−52.5%`), not the
  launch gap. **This plan's value must therefore come from pinned/overlapped H2D in the
  streamed envelope, and Phase 4 must PROVE a net wall-time win rather than assume one.**
- Current guard: `runner._validate_cuda_graph` hard-rejects `cuda_graph + block_stream`
  (`block_stream_blocks_per_group > 0`). `omni_infinity/cuda_graph.py` captures the whole
  transformer forward against static buffers; block streaming moves weights between/within
  steps, so those captured pointers go stale.

## Constraints that shape the design

- Graph capture needs **pointer-stable** storage for every tensor the captured region
  reads. Block streaming's group-offload writes weights to fresh device allocations per
  prefetch → pointers not stable across steps.
- diffusers `group_offload` hooks may insert **CPU sync points**; a hard sync inside the
  capture region makes whole-forward capture infeasible (this is the make-or-break risk).
- 22 GiB envelope: arena buffers + graph pools + working set must all fit; double-buffered
  arenas add VRAM that counts against the budget.
- C5 stays outside the graph wrapper (a cache hit bypasses replay, counted as `cache_skip`).
- Parity on GPU 0 (golden provenance); timing on GPU 4; one GPU per run; GPU etiquette
  (never touch physical GPUs 1,2,3 or the MPS server).

## Design — pointer-stable per-group weight arenas

```
           copy stream (H2D)                 compute stream (graph replay)
           ─────────────────                 ──────────────────────────────
  prefetch group N+1 ─┐                       replay captured graph for group N
                      ▼                         reads weights from arena[N % 2]
    arena[(N+1) % 2]  (fixed pinned buffer, pointer never changes)
                      │  event: H2D(N+1) done ──► compute waits before it uses N+1

  double-buffered arenas arena[0], arena[1], each sized to the MAX block-group footprint
  the graph captures compute against arena pointers that are STABLE across steps;
  only the arena CONTENTS change, written by streaming on the copy stream
```

## Approach (phased)

1. **Phase 0 — feasibility checkpoint (make-or-break).** Prove diffusers `group_offload`
   H2D can be redirected into pre-allocated arenas **without a CPU sync inside the capture
   region**. Micro-probe with `torch.cuda.graph` over a 2-block toy using a stubbed
   offload-into-arena. If an unavoidable sync is found → pivot to Phase 3 sub-graphs, or
   document infeasibility and stop. (Full-build mandate still honours a hard technical
   blocker — this is a pivot/stop condition, not a "marginal gain" abort.)
2. **Phase 1 — arena allocator.** Pointer-stable per-group arenas sized to the max group
   footprint; double-buffered; VRAM-accounted; integrated with the `CudaGraphManager`
   pool discipline and a 3-stream pool (H2D / D2H / compute).
3. **Phase 2 — streaming integration.** Intercept the group-offload prefetch path to write
   into arenas on the copy stream; event-sync the compute stream before replay; keep the
   AdaLN host-cache pinning path working in this profile.
4. **Phase 3 — capture granularity.** Attempt whole-forward capture; if syncs/pointer
   issues block it, fall back to **per-block-group sub-graphs** (each group its own small
   graph). The sub-graph fallback carries the **same bitwise-parity bar** as
   whole-forward capture. Lift the `_validate_cuda_graph` block-stream rejection
   **only** for the validated config + shapes; everything else still falls back to eager.
5. **Phase 4 — validate.** Bitwise parity (GPU 0); 22 GiB OOM check with arenas + pools
   counted; timing ablation block-stream OFF vs block-stream+graph (must be a **net** win);
   serve `graph_telemetry` fallback reasons; CPU suite non-regression.

## Tasks

- [ ] Phase 0 feasibility probe: arena-redirected group-offload under `torch.cuda.graph`, note appended here
- [ ] Pointer-stable double-buffered per-group arena allocator + VRAM accounting
- [ ] Streaming→arena H2D on copy stream with event-synced compute-stream replay
- [ ] Capture path (whole-forward, sub-graph fallback) + scoped lift of the block-stream guard
- [ ] Validation: bitwise parity, 22 GiB OOM check, block-stream timing ablation (net win), telemetry
- [ ] Docs: supported block-stream+graph envelope + updated P2 status

## Verification

Bitwise golden parity under `block-stream + cuda-graph` on GPU 0; peak allocated
**≤ 22 GiB** (arenas + graph pools included); timing ablation showing a net step-wall
win vs the **block-stream** baseline (resident numbers are not the comparison); serve
`graph_telemetry` fallback reasons; `pytest tests/ -m "not gpu and not weights"` no
regression.

## Risks

- **CORE UNKNOWN:** diffusers `group_offload` may inject CPU syncs inside capture → may
  force sub-graphs or a documented infeasibility stop even under the full-build mandate.
- With only a 2.41% launch gap, the win depends on H2D overlap/pinning in the streamed
  profile; if copy cannot be overlapped under the 22 GiB budget there may be **no net
  win** → Phase 4 is the honesty gate.
- Arena double-buffering increases peak VRAM against the 22 GiB envelope; sizing to the
  max group footprint may be tight.
- Pointer stability through diffusers hooks is fragile across diffusers versions.
- Per-group sub-graphs add replay/launch overhead that can erode the already-small gap.
