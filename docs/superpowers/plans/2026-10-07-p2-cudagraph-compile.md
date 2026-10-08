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
- [x] `compile-blocks` registry opt + parity-tier doc + bench row
- [x] Graph manager (vendored pattern) + resident-profile capture
- [x] Fallback-reason telemetry in job logs
- [x] Ablation rows: baseline / compile / graph / compile+graph × NFE
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

## Phase 1 compile-blocks results (2026-10-08)

`compile-blocks` uses diffusers regional compilation on the repeated H3
transformer and token-refiner blocks with `fullgraph=True, dynamic=True`. It is
resident-profile-only: combining it with block streaming raises before model
loading because group-offload hooks can expose stale weight pointers to a
compiled region. The C5 decision still wraps the whole transformer forward, so
a cache hit bypasses the compiled blocks.

This differs from the initial Phase 1 assumption: in diffusers 0.40.0,
`HostResidentAdaLN.forward` is called inside each repeated transformer's block
forward, so per-block AdaLN materialization is also inside the compiled region.
Only host-cache installation and C5's outer skip decision remain outside. The
extra H2D work inside that scope is a likely contributor to the wall-time
regression below.

The canonical compile-ON run used the same physical GPU 4 and workload as the
Phase 0 resident artifact: store-backed transformer + AdaLN host cache, 256p,
120 requested (124 effective) frames, seed 0, and 8 requested scheduler steps
(7 transformer forwards). Step 1 was excluded from the steady-state median.
The OFF row is the required reuse of `results/p2_phase0/resident.json`; the ON
record is `results/p2_phase1/compile-resident.json` and used a fresh isolated
Inductor cache. That same-GPU timing record is not used for parity because the
golden was recorded on the Server Edition GPU model. Provenance-valid parity
records are in `results/p2_phase1/parity/{resident,compile-resident}.json`.

| resident profile | median step wall (ms) | median compute (ms) | kernels/step | relative throughput |
|---|---:|---:|---:|---:|
| compile OFF | 1483.38 | 507.70 | 2487 | 1.000x |
| compile ON | 1579.38 | 471.05 | 983 | 0.939x |

Regional compilation fused kernels as intended: compute time improved 7.2%
(`507.70 -> 471.05 ms`) and launches fell 60.5% (`2487 -> 983`). That did not
translate to an end-to-end win for the host-AdaLN resident profile: variable
per-block host-to-device materialization dominated the step and wall time
regressed 6.5%. The first compiled forward took 26.75 s including cold
compilation and execution. Dynamo created 3 unique graphs during that cold
forward and 0 new graphs over the remaining six forwards, so changing
timesteps did not cause a recompile storm. `TORCH_LOGS=recompiles` showed one
cold-start token-refiner guard transition for an optional `None` attention
mask, not a steady-state recompile.

Parity did not reach the expected allclose tier. On the Server Edition GPU
matching the golden metadata, eager remained bitwise while compiled video
latents had `rms_rel=0.0726177`, were not bitwise, and failed elementwise allclose at
`rtol=atol` values `1e-5`, `1e-4`, `1e-3`, and `2e-2`. An eager control on the
golden's GPU model remained bitwise. The benchmark writes the artifact and
returns nonzero for this failed gate. Isolated precision probes showed that
`TORCHINDUCTOR_FORCE_SAME_PRECISION=1` reduced but did not close the error, and
`TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` did not help. Therefore this task
records `compile-blocks` as an opt-in measured negative result on torch
2.12.0+cu130/diffusers 0.40.0; it must not be represented as passing the golden
gate or as a speedup. Phase 2 should not assume regional compile is beneficial.

## Phase 2 graph manager results (2026-10-08)

`cuda-graph` captures the whole dense H3 transformer forward only for the
fully resident profile. Graphs are keyed by effective `(height, width, frames)`
video shape, use one CUDA graph pool per live bucket, and are capped at two
live LRU buckets. Separate pools allow buckets to replay in arbitrary order;
sharing one allocator pool would require preserving capture order. The manager
warms eagerly, then warms again on its capture side stream, copies every tensor
input into static buffers, requires exact non-tensor kwargs, clones the static
output tree for the scheduler, quarantines failed buckets, and exposes
plain-dict statistics. Its fallback reasons cover new/tensor-shape bucket
misses, warmup, capture failure and quarantine, C5 cache skips, kwarg mismatch,
generation invalidation, replay failure, and disabled graphs. C5 remains
outside the graph wrapper, so a cache hit bypasses replay and increments the
cache-skip fallback hook.

The AdaLN host cache is the important part of this result. Enabling the graph
pins its timestep-invariant weight, bias, and optional scale tensors only after
checking that `MemAvailable` covers the full footprint plus 20% headroom. The
canonical run pinned 24.23 GiB. This makes the captured H2D copies legal and,
independently of launch capture, avoids pageable-memory staging. CUDA graph use
without an AdaLN host cache is still supported, but then only the previously
measured 1.31% launch-gap ceiling is available. Block streaming remains a hard
error because its weight pointers are not stable. `compile-blocks` may be
combined with graphs; a capture failure is quarantined and reported rather
than silently claimed as a graph replay.

Timing used physical GPU 4 (RTX PRO 6000 Blackwell Max-Q), driver 590.48.01,
PyTorch 2.12.0+cu130, and diffusers 0.40.0 with the same store-backed
AdaLN-host-cache workload as Phase 0: 256p, 120 requested/124 effective frames,
seed 0, and 8 requested steps (7 forwards). The OFF row reuses
`results/p2_phase0/resident.json`; graph-ON replay medians use the three
steady-state replay forwards in `results/p2_phase2/graph-resident.json`.

| resident profile | median step wall (ms) | compute (ms) | copy (ms) | launch gap (ms) | kernels/step | throughput |
|---|---:|---:|---:|---:|---:|---:|
| graph OFF | 1483.38 | 507.70 | 955.75 | 18.66 | 2487 | 1.000x |
| graph ON replay | 966.42 | 510.95 | 453.71 | 1.95 | 2487 | 1.535x |

The graph manager recorded 1 capture, 3 steady replays, 0 capture failures,
three eager warmup calls, two capture-side warmup calls, and a 32.94 MiB
graph-pool delta. Two warmup fallbacks and one tensor-shape bucket transition
occurred before capture. Manager-side side-stream warmup, capture, and the
mandatory first replay took 2561.91 ms to enqueue; the full capture-step CUDA
window was 2968.89 ms. That one-time cost is separate from the replay median.

Parity was tested independently on physical GPU 0, the RTX PRO 6000 Blackwell
Server Edition matching the golden provenance. Both the eager control and
graph-ON candidate were bitwise (`rms_rel=0`) and passed the exact golden gate;
records are in `results/p2_phase2/parity/{resident,graph-resident}.json`.

**Verdict.** Phase 2 is a measured resident-profile win: steady replay is
34.9% faster in step wall time (1.535x throughput) with bitwise parity. This is
not evidence that the original 1.31% launch gap was underestimated: compute is
flat and the graph-only gap reduction contributes about 16.7 ms, while most of
the 517.0 ms improvement comes from explicitly pinning the 24.23 GiB AdaLN H2D
sources (copy time falls 52.5%). T5 should expect graph-only to be the useful
row. Compile+graph may recover some H2D time but still inherits Phase 1's
failed compile parity and should not be expected to beat or validate against
graph-only without new evidence.

## Phase 2 Task 4 — serve telemetry (2026-10-08)

The task-list checkbox remains intentionally unchanged for orchestrator
verification.

Successful serve records now expose optional `JobRecord.graph_telemetry` data
in both persisted `job.json` and `GET /v1/jobs/{id}` responses. `JobService`
brackets generation with CUDA-graph manager snapshots: the terminal
`succeeded` transition stores per-job deltas for capture, replay, failure,
warmup, capture time, and fallback-reason counters, plus the final graph-pool,
live-graph, and generation gauges. Profiles without a graph manager and legacy
records remain valid with `graph_telemetry: null`; a telemetry snapshot failure
is logged without changing successful job completion.

Example `job.json` fragment:

```json
{
  "graph_telemetry": {
    "capture_failures": 0,
    "capture_time_ms": 2561.91,
    "capture_warmup_calls": 2,
    "captures": 1,
    "fallback_reasons": {
      "shape_bucket_miss": 1,
      "warmup_not_done": 2
    },
    "generation": 0,
    "graph_pool_bytes": 34539520,
    "graphs": 1,
    "replays": 3,
    "warmup_calls": 3
  }
}
```
