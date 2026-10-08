# Omni-Infinity documentation

Guides, design contracts, and measurement reports for Omni-Infinity. Start
with the [project README](../README.md) for the overview, installation, and
the operator quickstart; this hub indexes the longer-form documents grouped
by topic.

## Getting started

- [Project README](../README.md) — overview, key features, installation,
  job-serving API, streaming playback, and the multi-prompt demo.
- [Architecture](../ARCHITECTURE.md) — contributor map of the codebase: the
  core inference path, the serving and streaming paths, and the request
  lifecycle.
- [Contributing](../CONTRIBUTING.md) — development setup, the Ruff and pytest
  gates, coding standards, and commit/PR conventions.
- [Security](../SECURITY.md) — supported versions, how to report a
  vulnerability, and the server's operational-security note.
- [VDN-Minimax-H3 reproduction (sm120)](repro_vdn.md) — end-to-end dense and
  hybrid reproduction, environment pins, and the two cross-repo gotchas.
- [Docker image](../docker/README.md) — reproducible CUDA 12.9 / Python 3.12
  image for the VDN-Minimax-H3 results, with the parity gate and native ladder.

## Memory & offloading

- Component offload, the host-resident AdaLN branch cache, bf16 block-streaming,
  text-encoder streaming, and the opt-in FP8 weight path are described under
  [Key Features](../README.md#key-features) and
  [Roadmap and status](../README.md#roadmap-and-status).
- [Ref2VA store path & cross-step prefetch](ref2va_step_overlap.md) — the
  Task-3 store path, denoising-step prefetch overlap, golden provenance, and
  the QA commands and measurements.

## Caches (opt-in, issue #24)

Nothing is on by default. Each level has an opt-in interface and a design note.

- [Cache benchmarks](cache_benchmarks.md) — datasets, suites, the frozen
  metric contract, and how each level's contribution is quantified.
- [C1 — exact condition cache](caches_c1_condition.md) — `condition-cache`.
- [C2 — encoder prefix cache (deferred)](caches_c2_prefix.md).
- [C3 — vision-embedding cache](caches_c3_vision.md) — `vision-cache`.
- [C4 — VDN shape-cache limits](caches_c4_shape.md).
- [C5 — denoise-step cache](caches_c5_denoise.md) — calibrated opt-in only.
- [Cache contract (design)](superpowers/plans/2026-09-30-caches-contract.md) —
  the shared interfaces tracked for issue #24.

## Serving & streaming

- [Job-serving API](../README.md#job-serving-api) — job lifecycle, request
  schema, on-disk persistence, and artifact access.
- [Streaming playback](../README.md#streaming-playback) — operator quickstart
  for the opt-in WebSocket player.
- [Streaming chunk playback (contract)](streaming_chunk_playback.md) — the
  behavior contract for the model-agnostic streaming layer.
- [Streaming workload benchmark](streaming_bench.md) — the metric contract and
  the recorded SKIP snapshot.

## VDN-Minimax-H3

- [Reproduction (sm120)](repro_vdn.md) — native and diffusers-path ladders,
  dense→hybrid+fp8 **2.15×** per-NFE, and the streamed ~20 GiB peak.
- [Ablation study](ablation_vdn.md) — a 16-run grid across the arch, NFE,
  kernels, backend, precision, and memory axes.
- [Speedup attribution](attribution_vdn.md) — a CUDA-event decomposition of
  the published VDN-H3 speedup (hypotheses H1–H6).
- [Attribution runlog](attribution_vdn_runlog.md) — the raw replay commands
  and QA output backing the attribution study.
- [Baseline tracking](baselines_vdn.md) — VDN-H3 as the strongest tracked
  baseline, the MiniMax-H3 reference numbers, and the 2026-10 survey of
  models claiming stronger video-generation results than H3.

## Research notes

- [SM120 optimization gap analysis](sm120_gap_analysis.md) — what vLLM and
  SGLang ship for omni models, the SM120 kernel ecosystem, and the ranked
  gaps in Omni-Infinity on consumer/workstation Blackwell.
- [Text-to-video optimization survey](t2v_optimization_survey.md) — the
  T2V serving framework landscape (FastVideo, SGLang diffusion, vLLM-omni,
  TensorRT-LLM VisualGen, xDiT), attention/caching/quantization techniques,
  and a ranked porting list for this stack.
- [H3 sm120 measured roofline](h3_sm120_measured_roofline.md) — measured PRO 6000
  roofline (BW 1485 GB/s, BF16 411 / FP8 721 TFLOPS, ridge 277 FLOP/byte), the
  compute-bound binding of dense attention, the gap-vs-N curve (46→89×), and the
  measured dense R≪1 verdict with the derived PRO 5000 upper bound.
