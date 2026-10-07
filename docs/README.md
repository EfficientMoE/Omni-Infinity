# Omni-Infinity documentation

Guides, design contracts, and measurement reports for Omni-Infinity. Start
with the [project README](../README.md) for the overview, installation, and
the operator quickstart; this hub indexes the longer-form documents grouped
by topic.

## Getting started

- [Project README](../README.md) — overview, key features, installation,
  job-serving API, streaming playback, and the multi-prompt demo.
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
