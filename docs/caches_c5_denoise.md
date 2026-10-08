# C5 denoise-step cache

The C5 cache is explicitly opt-in and requires coefficients calibrated for
the target model. No published TeaCache or cache-dit table covers MiniMax-H3.
This plan does not add a CLIP, SSIM, or PSNR gate.

The cache is local to one generation and wraps only the transformer's forward
method. Never publish H3 text K/V from a denoise step.
mode="output" is the H3 default. Residual mode is available only when output
and named-input tensor shapes match.

## Interaction with P2 CUDA graphs and regional compile

The P2 optimizations (`cuda-graph` and `compile-blocks`, roadmap #42) compose
with the C5 cache through nested wrappers on the transformer's `forward`
method. From outermost to innermost the call order is:

1. **C5** `cached_forward` — installed per generation by `denoise_step_cache`
   (`omni_infinity/caches/denoise.py`); it captures the then-current `forward`
   as its `original_forward`, so it always sits outside the P2 wrappers.
2. **CUDA-graph** `graph_forward` — installed at model load by
   `_wrap_transformer_with_cuda_graph` (`omni_infinity/runner.py`). It calls
   `CudaGraphManager.try_execute(...)` and runs eager `original_forward` on any
   fallback.
3. The real `MiniMaxH3Transformer3DModel.forward`, whose repeated blocks are the
   units replaced by `compile_repeated_blocks` when `compile-blocks` is on.

Because C5 is the outermost wrapper, a **cache-hit step returns before calling
`original_forward`**, so it bypasses graph replay (and the compiled blocks)
entirely; only computed steps descend into the graph/compiled path. This keeps
the data-dependent skip decision outside every static region, which is also why
`compile-blocks` can use `fullgraph=True` — the skip never enters a compiled
graph. The same nesting holds for `compile+graph`: C5 wraps outside both.

**Fallback accounting.** A skipped step never reaches `try_execute`, so it is
not counted there. Instead, at generation end the runner forwards the C5 skip
count to the manager via `CudaGraphManager.record_cache_skip(
denoise_stats.skipped)`, which increments the `cache_skip` entry of
`fallback_reasons` (`omni_infinity/cuda_graph.py`). Serve telemetry
(`JobRecord.graph_telemetry`) therefore reports C5 skips as `cache_skip`
alongside the graph's own fallbacks (shape-bucket miss, warmup, and so on).

**Scope.** Graphs capture the whole dense H3 transformer forward, keyed by
`(height, width, effective frames)`, and are resident-profile only: block
streaming is rejected because its weight pointers are not pointer-stable.
`compile-blocks` is likewise resident only. Both are large-GPU optimizations —
the resident profile peaks at ~71.9 GiB and does not fit the 22 GiB
memory-constrained serving envelope. See the P2 plan
(`docs/superpowers/plans/2026-10-07-p2-cudagraph-compile.md`) for measured
results: graph-only is the only configuration that is both faster (1.535×
throughput, i.e. 34.9% less step wall time) and bitwise-correct. `compile-blocks`
is slower (0.939×) and still fails the `2e-2` parity gate; `compile+graph`,
though faster (1.604×), also fails that gate. Neither compile variant may be
presented as passing the bitwise golden gate.
