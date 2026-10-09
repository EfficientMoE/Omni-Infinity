# Learnings — P2 cudagraph + compile

## [2026-10-08] Orchestrator: environment + codebase map (from exploration wave)

### Environment (VERIFIED)
- python = `/home/leyang/anaconda3/bin/python` (3.13), torch **2.12.0+cu130**, diffusers **0.40.0** (site-packages, NOT third_party vendored copy)
- GPUs: 6× RTX PRO 6000 Blackwell (sm120, ~96 GB each), all idle. Pick one via `CUDA_VISIBLE_DEVICES`.
- Checkpoint: `/mnt/raid0nvme0/leyang/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe`
- moe-store: `/mnt/raid0nvme0/leyang/h3-store-v2` (196G), components `transformer,vae,audio_vae`
- Goldens: `tests/fixtures/goldens/fl2va_goldens.pt` (+ ref2va)
- Use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- NOTE: `ls /mnt/raid0nvme0/leyang/hf-home` does NOT exist; use `~/.cache/huggingface` path above. README env examples use stale paths.

### Regional compile — READY OUT OF THE BOX
- `diffusers.ModelMixin.compile_repeated_blocks(*args, **kwargs)` exists in 0.40.0; compiles every module whose class name is in `_repeated_blocks`.
- `MiniMaxH3Transformer3DModel._repeated_blocks = ['MiniMaxH3TransformerBlock', 'MiniMaxH3TokenRefinerBlock']` (installed diffusers).

### Key integration points (file:line, verified by explorer)
- Registry: `omni_infinity/registry.py` — `OptimizationSpec` dataclass L40-47, `_kw()` L59, `OPTIMIZATIONS` dict; `runner_kwargs_for()` L162-185 validates arch support. Pattern for new opt: add spec with `runner_kwargs_by_arch=MappingProxyType({...: _kw(compile_blocks=True)})`.
- Runner: `omni_infinity/runner.py` — `ReferenceRunner.generate()` L384-442; `_denoising_progress()` L44-66 registers `register_forward_hook` on transformer (per-step counter); `_transformer_component()` L30-41; `_enable_block_streaming()` L69-91 (`enable_group_offload(block_level, use_stream=True)` called at L343-349 in `from_pretrained`); AdaLN cache load L315-324; step overlap L350-355.
- Denoise loop per-step forward: `third_party` modular pipeline — but RUNTIME uses installed diffusers `diffusers/modular_pipelines/minimax_h3/denoise.py` `MiniMaxH3LoopDenoiser.__call__()` (transformer called with `hidden_states=latents[None]`, `audio_hidden_states`, `encoder_hidden_states`, `timestep`, `timestep_indices`, ...).
- Transformer: `MiniMaxH3Transformer3DModel.forward()` block loop at L655-661 (`for block in self.transformer_blocks: hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)`); 50 layers.
- C5 cache: `omni_infinity/caches/denoise.py` `denoise_step_cache()` L158-227 — skip decision in `cached_forward()` L173-216, wraps transformer forward entirely → graph/compile scope must sit INSIDE (cached step bypasses forward → bypasses graph naturally if graph is at forward level).
- Serve telemetry: `omni_infinity/serve/models.py` `JobRecord` L75 (has `error` L84); `serve/store.py` `JobStore.transition()` L104, `_write()` L208 (job.json); `serve/service.py` `_execute()` L114, `_generate()` L143, step_callback L149-151.
- Parity tiers: bitwise `torch.equal` (test_reference_parity.py L91); FP8 `allclose(rtol=2e-2)` (test_fp8_parity.py L10); C5 `rms_rel <= rms_rel_max` (test_cache_bench_contract.py L73).
- Parity test env vars: `OMNI_H3_CHECKPOINT`, `OMNI_H3_OFFLOAD`, `OMNI_H3_STORE`, `OMNI_H3_STORE_COMPONENTS`, `OMNI_H3_ADALN_CACHE`, `OMNI_H3_BLOCK_STREAM`, `OMNI_H3_STREAM_TEXT_ENCODER`.
- Ablation harness: `benchmarks/ablation_vdn.py` — `Run` dataclass L41, `GRID` L51-189, `CSV_FIELDS` L29 (name, stack, s_per_nfe, peak_gib, rms_rel, cosine, notes, config_json), `build_command()` L196-282, metrics via `_read_omni_metrics()` L364. Results table: `docs/ablation_vdn.md`.
- CUDA-event timing pattern: `benchmarks/bench_fp8_gemm.py` `_time()` L35-49 (median of 50).

### Donor pattern summary (for graph manager task)
- MoE-Infinity `moe_infinity/serving/cuda_graph.py`: `CudaGraphRunner` — `GraphKey`/`GraphDecision(eligible, reason, key)`/`GraphExecutionStats(captures, replays, capture_failures, graph_pool_bytes, fallback_reasons: Counter)`; `check_eligibility()` → `try_execute()` returns None on fallback (caller runs eager); warmup on separate stream then `torch.cuda.graph(graph)`; static output buffer + `copy_()` inside capture; generation-based `invalidate()`; quarantine dict for failed keys; `FALLBACK_REASONS` tuple of reason strings. ADOPT: decision/reason pattern, stats Counter, quarantine, generation invalidation.
- BatchGen `batchgen/cuda_graph/graph_manager.py`: `BatchSizeBucketing` O(1) lookup table; `TensorSpec(shape with 'batch_size' placeholder, dtype, fill)`; `CapturableSegment` Protocol (get_static_input_specs/get_static_output_specs/forward + optional setup hooks); shared `torch.cuda.graph_pool_handle()` across all graphs; WARMUP_ITERATIONS=2 then capture with `pool=self._pool`; replay copies inputs into static buffers (non_blocking) + returns sliced outputs; `drop_bucket()` eviction. ADOPT: shared pool, TensorSpec/protocol, bucketing.
- Stream discipline (MoE-Inf `core/memory/stream_pool.h`): 3 streams/device — H2D(0), D2H(1), compute(2).
- For Omni denoise: bucket key = (resolution, frames) i.e. latent shape — NOT batch size (always 1).

### Survey evidence (plan refs)
- Regional compile `fullgraph=True, dynamic=True` ≈1.5× runtime, 7× faster cold start vs full-model compile (t2v survey §3).
- C5 data-dependent skip breaks fullgraph → compile/graph scope excludes skip decision (already satisfied: C5 wraps forward from outside).
- sm120: no WGMMA/tcgen05, 99KB SMEM.

## [2026-10-08] Phase 0 denoise-step profiling

- GPU 4, RTX PRO 6000 Blackwell Max-Q, torch 2.12.0+cu130, diffusers 0.40.0;
  same GPU used sequentially for both profiles. Inputs: 8 requested steps,
  256p, 120 requested frames, seed 0, `tests/fixtures/ref.png`; scheduler ran
  7 transformer forwards and step 1 was discarded (6 measured).
- Resident median: wall 1483.38 ms, compute 507.70 ms, copy 955.75 ms,
  copy-adjusted launch gap 18.66 ms / 1.31%, 2487 kernels/step, peak allocated
  71.89 GiB.
- Block-stream median: wall 2951.59 ms, compute 508.56 ms, copy 2360.30 ms
  (H2D 2024.61 ms), copy-adjusted launch gap 72.91 ms / 2.41%, 2487
  kernels/step, peak allocated 13.37 GiB.
- Decision: 2.41% < 5%, so demote block-stream P2, scope Phase 1/2 to resident,
  and drop Phase 3 block-stream graph capture. Resident remains the only
  stable-pointer target, although its 1.31% pure launch-gap ceiling means the
  later ablation must justify continuing.
- Method: torch.profiler CUPTI ranges from forward-hook `record_function`;
  kernel and memcpy unions computed separately by stream. Launch gap subtracts
  the union of both, preventing H2D stalls from masquerading as launch gaps.
  CUDA-event median wall agreed within 0.22 ms.
- Gotchas: merged snapshot has no top-level `transformer/config.json`; use
  `FL2VA/transformer` for the store model config without mutating the snapshot.
  FL2VA requires the keyframe fixture. Requested 120 frames round to 124.
  `num_inference_steps=8` yields 7 transformer forwards in this scheduler.

### Phase 0 review clarifications

- Launch-gap percentages are medians of per-step percentages, not ratios of
  the independently computed median gap and median wall columns.
- `wall - union(kernel, memcpy)` is a copy-adjusted device-idle upper bound on
  launch overhead; dependencies, synchronization, allocator effects, and
  profiler idle time can only make it more conservative. Since this upper
  bound is below 5% for block-stream, the demotion decision still follows.
- Effective video length is 124 frames after the H3 VAE aligns the requested
  120 frames to `17*n+5`; raw JSON records both values.
- Resident-only P2 is a large-GPU scope: its 71.89 GiB peak does not satisfy
  the separate 22 GiB memory-constrained serving envelope.

## [2026-10-08] Phase 1 regional compile

- `compile_repeated_blocks(fullgraph=True, dynamic=True)` compiled the 50 H3
  blocks and 2 token-refiner blocks into 3 unique Dynamo graphs. No new graph
  appeared after the cold forward, so timestep changes did not cause a storm.
- Same-GPU-4 canonical result: OFF 1483.38 ms vs ON 1579.38 ms median step wall
  (0.939x); compute improved 507.70 -> 471.05 ms and kernels fell 2487 -> 983,
  but variable host-AdaLN H2D dominated and erased the fusion win.
- Cold first forward was 26.75 s. Always use a fresh
  `TORCHINDUCTOR_CACHE_DIR` for compiler-option experiments: the global cache
  reused a `force_same_precision` kernel after the environment flag was
  removed, contaminating a nominal default rerun.
- Provenance-valid Server Edition compile parity failed: video-latent
  `rms_rel=0.0726177`, not bitwise, and not elementwise allclose at
  `rtol=atol=2e-2`; the paired eager control stayed bitwise. The benchmark now
  writes evidence before returning nonzero on mismatch/failure.
  `TORCHINDUCTOR_FORCE_SAME_PRECISION=1` improved a GPU-2
  probe from ~7.26% to ~3.89% RMS error but still failed; eager precision-cast
  emulation did not improve it.
- AdaLN materialization is inside each repeated H3 block's forward in diffusers
  0.40.0; only cache installation and C5's skip decision are outside regional
  compilation. C5 still safely bypasses compiled blocks on a cache hit.

## [2026-10-08] Phase 2 resident CUDA graph

- Pinning the 50 host AdaLN entries covered 24.23 GiB (weight, bias, optional
  scale) after a `MemAvailable >= 1.2 * bytes` guard. This is required for H2D
  capture and was the dominant speedup: replay copy time was 453.71 ms versus
  the pageable Phase-0 baseline's 955.75 ms.
- GPU-4 steady replay was 966.42 ms versus 1483.38 ms OFF (1.535x throughput,
  34.9% less wall time). Compute stayed flat at 510.95 versus 507.70 ms; launch
  gap fell 18.66 -> 1.95 ms, consistent with the 1.31% pure-launch ceiling.
- The canonical 8-step workload produced two ordinary warmups, one tensor-shape
  bucket transition, two capture-side warmups, one capture, and three replays.
  Graph-pool growth was 32.94 MiB. Capture-side warmup, capture, and the required
  first replay enqueued in 2561.91 ms; the capture step's full CUDA window was
  2968.89 ms.
- CUDA stream capture does not yield a scheduler-ready output by itself. The
  graph must replay once immediately after capture; returning the captured
  output buffer before that replay caused a reproducible `rms_rel=1` at the
  capture forward and 0.3728 final latent error. After the first-replay fix,
  capture and every replay matched eager outputs and final parity was bitwise.
- Replacing an instance `forward` with `*args, **kwargs` breaks diffusers H3:
  its denoiser filters packed-layout fields through `inspect.signature`. Wrap
  with `functools.wraps(type(transformer).forward)` so the bound instance keeps
  the original signature.
- Provenance-valid GPU-0 eager and graph runs both reproduced the golden
  bitwise (`rms_rel=0`). Timing and parity artifacts are under
  `results/p2_phase2/`.
- Keep a distinct allocator pool per live graph bucket. PyTorch shared-pool
  memory reuse assumes replay in capture order, which conflicts with arbitrary
  shape-bucket access and LRU eviction. Capture-side warmups must also run on
  the same side stream used for graph capture.

## [2026-10-08] Phase 2 Task 4 serve telemetry

- `GraphTelemetry` mirrors the plain-dict `CudaGraphManager.stats_snapshot()`
  schema and is optional on both `JobRecord` and `JobResponse`, preserving
  compatibility with legacy `job.json` records.
- `JobService` brackets generation with snapshots so reused-runner counters and
  fallback reasons are stored as per-job deltas. Pool bytes, live graphs, and
  manager generation remain final gauges. Runners without a
  `cuda_graph_manager` persist `null` without special casing their type.
- Snapshot or validation failures are logged and do not convert an otherwise
  successful generation into a failed job. CPU-only tests cover persistence,
  API conversion, fallback-reason counts, manager absence, failure isolation,
  and legacy records.

## [2026-10-08] Phase 2 Task 5 optimization ablation

- The 4×2 timing grid and four NFE-8 parity cells are normalized under
  `results/p2_phase2_ablation/`; the original 16-row VDN `GRID` remains
  unchanged and P2 is selected with `--suite p2`.
- GPU-4 graph replay was stable across NFE: 966.42 ms canonical at NFE 8 and
  965.41 ms at NFE 16. Compile+graph capture succeeded (it did not quarantine),
  replaying at 924.82/924.90 ms with ~2.67 s one-time capture and 24.23 GiB
  pinned AdaLN host memory.
- Compile+graph's faster replay is not usable under the accuracy contract:
  GPU-0 parity was `rms_rel=0.04114`, cosine 0.999162, and failed elementwise
  allclose at `rtol=atol=2e-2`. Compile-only also failed (`rms_rel=0.06644`,
  cosine 0.997807). Baseline and graph remained bitwise.
- Pageable AdaLN H2D is host-state-sensitive. An initial baseline recheck
  produced a 3082.59 ms median because copy time inflated to 2552.94 ms while
  compute stayed at 508.29 ms; a clean retry returned 1492.84 ms. The fresh
  compile recheck was unusually favorable at 1238.67 ms. Keep the committed
  Phase-0/1/2 canonical NFE-8 timings in the headline table and retain fresh
  values in JSON as rechecks rather than silently replacing canonical data.
- The shared latent cosine helper can return 1.000004 for bitwise-identical
  bf16 tensors because it computes cosine in fp32; use `rms_rel=0` plus the
  explicit `bitwise` tier as the exactness criterion, not cosine clipping.
- `/proc/driver/nvidia/version` contains the driver after `Kernel Module for
  x86_64`; selecting the token after the word `Module` incorrectly records
  `for`. The harness now parses the numeric version explicitly.
