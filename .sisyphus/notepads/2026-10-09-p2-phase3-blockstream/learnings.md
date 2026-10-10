# Learnings — P2 Phase 3 (seed from P2)

## P2 integration points (verified on the base branch, file:fn)
- `omni_infinity/runner.py`:
  - `_wrap_transformer_with_cuda_graph` (~172-190): replaces transformer.forward with
    `graph_forward` calling `manager.try_execute(current_bucket, original_forward, ...)`;
    eager fallback on None. Installed at model load.
  - `_validate_cuda_graph` (~109-117): HARD-REJECTS cuda_graph + block_stream
    (block_stream_blocks_per_group > 0). Phase 3 must lift this ONLY for the validated
    config/shapes.
  - block-streaming via diffusers `apply_group_offloading` / `enable_group_offload`
    (`_APPLY_GROUP_OFFLOADING`, `_enable_block_streaming`); step-overlap in step_overlap.py.
  - AdaLN host-cache pinning `_pin_adaln_for_cuda_graph` (~134); C5 skip forwarding
    `record_cache_skip(denoise_stats.skipped)` (~609-610).
- `omni_infinity/cuda_graph.py`: `CudaGraphManager.try_execute` (~324-442) warmup ->
  capture (`captured_forward` + first replay) -> replay; `FALLBACK_REASONS` includes
  `cache_skip`; one CUDA pool per live bucket, max 2 LRU buckets; `record_cache_skip`.
- `omni_infinity/caches/denoise.py`: C5 `denoise_step_cache` wraps transformer.forward
  OUTSIDE the graph wrapper (cache hit bypasses replay).
- Donor patterns: MoE-Infinity `core/memory/stream_pool.h` (3 streams: H2D/D2H/compute);
  BatchGen arena/bucketing.

## Key P2 result that shapes Phase 3 value
- The P2 graph win (1.535x) was DOMINATED by pinning the 24.23 GiB AdaLN host-cache H2D
  sources (copy -52.5%), NOT launch-gap elimination. Block-stream Phase-0 launch gap was
  only 2.41% (< 5%). So Phase 3's value MUST come from pinned/overlapped H2D into the
  streamed (22 GiB) envelope. Phase 4 must PROVE a net step-wall win vs the block-stream
  baseline, not assume one.

## Env (verified)
- python=/home/leyang/anaconda3/bin/python (3.13), torch 2.12.0+cu130, diffusers 0.40.0.
- HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; PYTHONPATH=<worktree>.
- Checkpoint snapshot 42ed227ee7df40d41602854ae760620d6eb651fe
  (~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/...).
- moe-store /mnt/raid0nvme0/leyang/h3-store-v2 (components transformer,vae,audio_vae).
- Goldens tests/fixtures/goldens/fl2va_goldens.pt (+ ref2va); FL2VA needs
  tests/fixtures/ref.png keyframe. 256p, 120 req (124 effective) frames, seed 0, 8 steps
  (7 transformer forwards; step 1 discarded in steady-state medians).
- Parity tiers: bitwise torch.equal; allclose rtol=atol=2e-2; C5 rms_rel.
- CPU test baseline: 430 passed, 13 deselected.
