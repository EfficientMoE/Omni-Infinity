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
