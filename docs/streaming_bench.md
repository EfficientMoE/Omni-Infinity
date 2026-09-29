# Streaming Workload Benchmark Report

Frozen metric contract: [`benchmarks/streaming/contract.py`](../benchmarks/streaming/contract.py).
Tracks live in worktrees under `.worktrees/bench-stream-{baseline,micro,ablation}`
(branches `bench/stream-baseline`, `bench/stream-micro`, `bench/stream-ablation`).
CSVs under `results/streaming/` are gitignored; tables below are the recorded
snapshot from this collection pass.

## Protocol

| Knob | Value |
| --- | --- |
| Shape | FL2VA, 256p short edge, 120 frames (24 fps), seed 0 |
| Prompt | `a red ball bouncing` |
| Baseline opts | `adaln-host-cache,block-stream,text-encoder-stream` |
| Chunk / transport | 24 frames / WebSocket |
| Warmup | 1 discarded |
| Measured reps | baseline/micro 3; ablation 1 (retry to 3 if e2e variance > 3%) |
| Hardware target | RTX PRO 6000 (sm120), one GPU at a time |

Frame substitution rule: if the runner rejects 120, use the next legal
`17*n+5` length (≥ 120 → 124) and set `notes=frame_substitution=124`.

## Targets

### Omni ClipChunker (`omni-clip`)

- `ttff_ms <= t_job_ready_ms`
- `quality_vs_job == bitwise`
- `prompt_index_match == 1`, `av_offset_ms <= 40`, `stall_count == 0`
- `peak_gib <= 24`
- `chunk_rtf` is not a fail gate (ClipChunker generates the whole clip first)
- Micro overhead: `e2e_ms - t_job_ready_ms <= max(2000, 0.05 * t_job_ready_ms)`
- Micro envelope: named spans + `other_ms` within 5% of session wall
- Micro `fmp4_frag_ms` p50 ≤ 50 ms per 24-frame 256p fragment

### Omni NativeChunker (`omni-native`)

- Requires a registered H3-World runner; otherwise `SKIP`
- `ttff_ms <= 1.5 * chunk_duration_ms`
- `chunk_rtf_p50 <= 1.0`, `chunk_rtf_p95 <= 1.2`, `stall_count == 0`
- `quality_vs_job=skipped` (different sample from one-shot job)

### External live controls

- Helios-Distilled: sustained fps p50 ≥ 16, stall 0, chunk RTF p50 ≤ 1.0
- LingBot / SANA-WM: chunk RTF p50 ≤ 1.0, stall 0; cookbook controls must be accepted

### Ablation axis choices

- Chunk size: smallest `{24,48,72}` with stall 0 + bitwise whose `ttff_ms` is
  strictly less than `chunk-120`; else stay on 24 and `REPORT`
- Transport: WS when `ttff_ms(ws) <= ttff_ms(hls)`
- Arch: `vdn-hybrid` when both peaks ≤ 24 GiB and VDN `e2e_ms` ≤ dense
- Opts: text-encoder stream Δs/eval < 1% at peak ≤ 24; fp8 e2e ≤ bf16;
  no-block-stream OOM is recorded, not retried

## Baseline verdicts

Collection date: 2026-09-29. All rows `SKIP`:

- Local Omni stacks: `POST /v1/streams` / `WS /v1/streams/{id}/ws` are not
  present on `bench/stream-*` (issue
  [#14](https://github.com/EfficientMoE/Omni-Infinity/issues/14) not merged
  into these measurement branches) → `notes=issue-14-absent`.
- External Helios / LingBot / SANA-WM: checkpoint env paths unset →
  `notes=weights-absent`.

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
(same blockers as baseline). The 5% envelope gate and `fmp4_frag_ms` p50 ≤ 50
are unit-tested with synthetic spans; they were not exercised against a GPU
server in this pass.

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
- This collection pass did not measure GPU sessions: stream API absent on the
  bench branches, and external weight env vars were unset.

## How to re-run after #14 lands

```bash
# contract + report branch
git checkout bench/stream-harness

# one GPU at a time
cd .worktrees/bench-stream-baseline
OMNI_STREAM_ENABLED=1 python -m omni_infinity.serve &
python benchmarks/streaming/baseline.py --out results/streaming/baseline.csv

cd ../bench-stream-micro
python benchmarks/streaming/micro.py --out results/streaming/micro.csv

cd ../bench-stream-ablation
python benchmarks/streaming/ablation.py --out results/streaming/ablation.csv
```

Optional external controls: set `OMNI_HELIOS_WEIGHTS`, `OMNI_LINGBOT_WEIGHTS`,
and `OMNI_SANA_WM_WEIGHTS` to local checkpoint directories before measuring
those stacks.
