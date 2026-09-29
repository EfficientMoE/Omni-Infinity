# Streaming Workload Benchmark Contract and Report

Metric contract: [`benchmarks/streaming/contract.py`](../benchmarks/streaming/contract.py).
The original columns remain frozen; support/reason and client-observed metric
columns are appended for compatibility. The typed shared client is
[`benchmarks/streaming/client.py`](../benchmarks/streaming/client.py).
Tracks live in worktrees under `.worktrees/bench-stream-{baseline,micro,ablation}`
(branches `bench/stream-baseline`, `bench/stream-micro`, `bench/stream-ablation`).
CSVs under `results/streaming/` are gitignored; tables below are the recorded
snapshot from this collection pass.

## Protocol

| Knob | Value |
| --- | --- |
| Dense default | FL2VA, 256p, 120 frames (24 fps), seed 0 |
| Architecture comparison | Dense and VDN both at 768p |
| Prompt | `a red ball bouncing` |
| Baseline opts | `adaln-host-cache,block-stream,text-encoder-stream` |
| Chunk / transport | 24 frames / WebSocket |
| Warmup | 1 discarded |
| Measured reps | baseline/micro 3; ablation 1 (retry to 3 if e2e variance > 3%) |
| Hardware target | RTX PRO 6000 (sm120), one GPU at a time |

Frame substitution rule: if the runner rejects 120, use the next legal
`17*n+5` length (≥ 120 → 124) and set `notes=frame_substitution=124`.

The WebSocket clock supports accept-to-first-decodable-fragment TTFF and
inter-arrival gaps. An arrival gap is not chunk production latency and must
never populate `chunk_produce_ms` or production-derived chunk RTF columns.
Production latency is `unsupported` until the server emits production
timestamps. HLS is a post-generation retrieval fallback: its playlist becomes
ready only after the WebSocket stream has completed, so HLS TTFF is also
`unsupported`.

## Targets

### Omni ClipChunker (`omni-clip`)

- `ttff_ms <= t_job_ready_ms`
- `quality_vs_job == bitwise`
- `cue_alignment == 1`, `av_offset_ms <= 40`, `stall_count == 0`
- `peak_gib <= 24`
- Arrival gaps are report-only; `chunk_rtf` is unsupported without production
  telemetry (ClipChunker generates the whole clip first).
- Micro overhead: `e2e_ms - t_job_ready_ms <= max(2000, 0.05 * t_job_ready_ms)`
- Detailed denoise/VAE/audio/fMP4 spans and the micro envelope are unsupported
  unless explicit server telemetry is present.

### Omni NativeChunker (`omni-native`)

- The checked-in implementation is a synthetic 16x16 stub and must be
  detected as `unsupported`, never measured as native model performance.
- A future registered H3-World runner may enable native TTFF/production gates,
  but only with explicit implementation identity and production telemetry.
- `quality_vs_job=skipped` (different sample from one-shot job)

### External live controls

- Helios-Distilled: sustained fps p50 ≥ 16, stall 0, chunk RTF p50 ≤ 1.0
- LingBot / SANA-WM: chunk RTF p50 ≤ 1.0, stall 0; cookbook controls must be accepted

### Ablation axis choices

- Chunk size: smallest `{24,48,72}` with stall 0 + bitwise whose `ttff_ms` is
  strictly less than `chunk-120`; else stay on 24 and `REPORT`
- Transport: WebSocket is the only live transport. HLS reports
  post-generation retrieval behavior and is not a TTFF competitor.
- Arch: compare `h3-dense` and `vdn-hybrid` on the common 768p canvas;
  choose VDN when both peaks ≤ 24 GiB and VDN `e2e_ms` ≤ dense.
- Opts: text-encoder stream Δs/eval < 1% at peak ≤ 24; fp8 e2e ≤ bf16;
  no-block-stream OOM is recorded, not retried

## Baseline verdicts

Collection date: 2026-09-29. All rows `SKIP`:

- Local Omni stacks: the 2026-09-29 rows are SKIP because
  `bench/stream-baseline`, `bench/stream-micro`, and `bench/stream-ablation`
  do not register the stream routes. `bench/stream-harness` does.
  `notes=issue-14-absent`.
- External Helios / LingBot / SANA-WM stay `weights-absent` when
  `OMNI_HELIOS_WEIGHTS`, `OMNI_LINGBOT_WEIGHTS`, and `OMNI_SANA_WM_WEIGHTS`
  are unset.

| track | stack | arch | chunk_frames | transport | rep | ttff_ms | t_job_ready_ms | chunk_rtf_p50 | stall_count | quality_vs_job | peak_gib | sustained_fps | verdict | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | omni-clip | h3-dense | 24 | ws | 0 |  |  |  | 0 | skipped |  |  | SKIP | issue-14-absent |
| baseline | omni-clip | h3-dense | 24 | ws | 1 |  |  |  | 0 | skipped |  |  | SKIP | issue-14-absent |
| baseline | omni-clip | h3-dense | 24 | ws | 2 |  |  |  | 0 | skipped |  |  | SKIP | issue-14-absent |
| baseline | omni-clip | vdn-hybrid | 24 | ws | 0 |  |  |  | 0 | skipped |  |  | SKIP | issue-14-absent |
| baseline | omni-clip | vdn-hybrid | 24 | ws | 1 |  |  |  | 0 | skipped |  |  | SKIP | issue-14-absent |
| baseline | omni-clip | vdn-hybrid | 24 | ws | 2 |  |  |  | 0 | skipped |  |  | SKIP | issue-14-absent |
| baseline | omni-native | h3-dense | 24 | ws | 0 |  |  |  | 0 | skipped |  |  | SKIP | issue-14-absent |
| baseline | omni-native | h3-dense | 24 | ws | 1 |  |  |  | 0 | skipped |  |  | SKIP | issue-14-absent |
| baseline | omni-native | h3-dense | 24 | ws | 2 |  |  |  | 0 | skipped |  |  | SKIP | issue-14-absent |
| baseline | vllm-helios | helios-distilled | 24 | ws | 0 |  |  |  | 0 | skipped |  |  | SKIP | weights-absent |
| baseline | vllm-helios | helios-distilled | 24 | ws | 1 |  |  |  | 0 | skipped |  |  | SKIP | weights-absent |
| baseline | vllm-helios | helios-distilled | 24 | ws | 2 |  |  |  | 0 | skipped |  |  | SKIP | weights-absent |
| baseline | sglang-lingbot | lingbot-world | 24 | ws | 0 |  |  |  | 0 | skipped |  |  | SKIP | weights-absent |
| baseline | sglang-lingbot | lingbot-world | 24 | ws | 1 |  |  |  | 0 | skipped |  |  | SKIP | weights-absent |
| baseline | sglang-lingbot | lingbot-world | 24 | ws | 2 |  |  |  | 0 | skipped |  |  | SKIP | weights-absent |
| baseline | sglang-sana-wm | sana-wm | 24 | ws | 0 |  |  |  | 0 | skipped |  |  | SKIP | weights-absent |
| baseline | sglang-sana-wm | sana-wm | 24 | ws | 1 |  |  |  | 0 | skipped |  |  | SKIP | weights-absent |
| baseline | sglang-sana-wm | sana-wm | 24 | ws | 2 |  |  |  | 0 | skipped |  |  | SKIP | weights-absent |

## Micro envelope

No live session wall was collected. H3 span columns are `SKIP` for every row
(same blockers as baseline). The current wire protocol has no detailed
denoise, VAE, audio, fMP4, or WebSocket-send telemetry, so these spans remain
explicitly `unsupported`; client arrival timestamps cannot substitute for
them.

| track | stack | arch | rep | denoise_ms | vae_decode_ms | fmp4_frag_ms | ws_send_ms | envelope_ok | verdict | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| micro | omni-clip | h3-dense | 0 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | issue-14-absent |
| micro | omni-clip | h3-dense | 1 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | issue-14-absent |
| micro | omni-clip | h3-dense | 2 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | issue-14-absent |
| micro | omni-clip | vdn-hybrid | 0 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | issue-14-absent |
| micro | omni-clip | vdn-hybrid | 1 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | issue-14-absent |
| micro | omni-clip | vdn-hybrid | 2 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | issue-14-absent |
| micro | omni-native | h3-dense | 0 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | issue-14-absent |
| micro | omni-native | h3-dense | 1 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | issue-14-absent |
| micro | omni-native | h3-dense | 2 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | issue-14-absent |
| micro | vllm-helios | helios-distilled | 0 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | weights-absent |
| micro | vllm-helios | helios-distilled | 1 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | weights-absent |
| micro | vllm-helios | helios-distilled | 2 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | weights-absent |
| micro | sglang-lingbot | lingbot-world | 0 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | weights-absent |
| micro | sglang-lingbot | lingbot-world | 1 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | weights-absent |
| micro | sglang-lingbot | lingbot-world | 2 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | weights-absent |
| micro | sglang-sana-wm | sana-wm | 0 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | weights-absent |
| micro | sglang-sana-wm | sana-wm | 1 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | weights-absent |
| micro | sglang-sana-wm | sana-wm | 2 | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | weights-absent |

## Ablation choice per axis

Unmeasured grid → stay on the baseline knobs and mark each axis `REPORT`:

| Axis | Choice | Verdict |
| --- | --- | --- |
| chunk_frames | 24 | REPORT |
| transport | ws | REPORT |
| arch | h3-dense | REPORT |
| text-encoder-stream | (baseline keep) | REPORT |
| fp8 | (baseline keep) | REPORT |
| block-stream | (baseline keep) | REPORT |

| name | stack | arch | chunk_frames | transport | opts | chunker | verdict | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chunk-24 | omni-clip | h3-dense | 24 | ws | adaln-host-cache,block-stream,text-encoder-stream | clip | SKIP | issue-14-absent |
| chunk-48 | omni-clip | h3-dense | 48 | ws | adaln-host-cache,block-stream,text-encoder-stream | clip | SKIP | issue-14-absent |
| chunk-72 | omni-clip | h3-dense | 72 | ws | adaln-host-cache,block-stream,text-encoder-stream | clip | SKIP | issue-14-absent |
| chunk-120 | omni-clip | h3-dense | 120 | ws | adaln-host-cache,block-stream,text-encoder-stream | clip | SKIP | issue-14-absent |
| transport-ws | omni-clip | h3-dense | 24 | ws | adaln-host-cache,block-stream,text-encoder-stream | clip | SKIP | issue-14-absent |
| transport-hls | omni-clip | h3-dense | 24 | hls | adaln-host-cache,block-stream,text-encoder-stream | clip | SKIP | issue-14-absent |
| arch-dense | omni-clip | h3-dense | 24 | ws | adaln-host-cache,block-stream,text-encoder-stream | clip | SKIP | issue-14-absent |
| arch-vdn | omni-clip | vdn-hybrid | 24 | ws | adaln-host-cache,block-stream,text-encoder-stream | clip | SKIP | issue-14-absent |
| opt-no-block-stream | omni-clip | h3-dense | 24 | ws | adaln-host-cache,text-encoder-stream | clip | SKIP | issue-14-absent |
| opt-fp8 | omni-clip | h3-dense | 24 | ws | adaln-host-cache,block-stream,text-encoder-stream,fp8 | clip | SKIP | issue-14-absent |
| opt-no-text-stream | omni-clip | h3-dense | 24 | ws | adaln-host-cache,block-stream | clip | SKIP | issue-14-absent |
| chunker-clip | omni-clip | h3-dense | 24 | ws | adaln-host-cache,block-stream,text-encoder-stream | clip | SKIP | issue-14-absent |
| chunker-native | omni-native | h3-dense | 24 | ws | adaln-host-cache,block-stream,text-encoder-stream | native | SKIP | issue-14-absent; native-runner-absent |

## Threats to validity

- sm120 only; absolute milliseconds are not compared to B200/B300 publications.
- Single fixed prompt and seed.
- ClipChunker is not live causal streaming; fragment-0 produce time holds the
  full denoise and is excluded from chunk RTF summaries.
- External live fps targets are model-native (Helios ~16, LingBot chunk
  geometry), not H3's 24 fps.
- Issue #14's `NativeChunker` is interface-only until H3-World lands; native
  rows are `SKIP`, never a fake pass.
- This collection pass did not measure GPU sessions. The stream routes were
  absent on `bench/stream-baseline`, `bench/stream-micro`, and
  `bench/stream-ablation`, and the external weight env vars were unset.

## How to re-run with #14

1. Start the server from a checkout that contains the stream routes.
   `bench/stream-harness` does. `bench/stream-baseline`,
   `bench/stream-micro`, and `bench/stream-ablation` do not.
2. A dense measurement needs the README streamed profile, plus streaming
   flags: `OMNI_CHECKPOINT`, `OMNI_STORE_DIR`, `OMNI_STORE_COMPONENTS`,
   `OMNI_MODEL_ARCH`, `OMNI_OPTIMIZATIONS`, `OMNI_MAX_VRAM`, and
   `OMNI_STREAM_ENABLED=1`. VDN measurements use `vdn-hybrid` at 768p and
   do not use the dense AdaLN cache.
3. From `.worktrees/bench-stream-baseline`, run
   `python benchmarks/streaming/baseline.py --base-url http://127.0.0.1:8000 --out results/streaming/baseline.csv`.
   From `.worktrees/bench-stream-micro`, run
   `python benchmarks/streaming/micro.py --base-url http://127.0.0.1:8000 --out results/streaming/micro.csv`.
   Those scripts are not on the harness branch.
4. ablation.py writes a skip manifest. It prints one `restart_command`
   per row. That command sets `OMNI_MODEL_ARCH`, `OMNI_OPTIMIZATIONS`,
   `OMNI_STREAM_ENABLED`, `OMNI_STREAM_CHUNK_FRAMES`, and
   `OMNI_STREAM_FALLBACK_HLS`. The operator still supplies
   `OMNI_CHECKPOINT`, `OMNI_STORE_DIR`, and `OMNI_MAX_VRAM`, starts a fresh
   process per row, and does not retry OOM. Run it from
   `.worktrees/bench-stream-ablation`.
5. Client arrival gaps stay in `arrival_gap_ms`. They do not fill
   `chunk_produce_ms`. Measure one GPU at a time.

Optional external controls: set `OMNI_HELIOS_WEIGHTS`, `OMNI_LINGBOT_WEIGHTS`,
and `OMNI_SANA_WM_WEIGHTS` to local checkpoint directories before measuring
those stacks.
