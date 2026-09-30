# Cache benchmarks (issue #24, PR #25)

This suite quantifies the contribution of each opt-in cache described in
[the cache design](caches.md) and tracked by
[issue #24](https://github.com/EfficientMoE/Omni-Infinity/issues/24). It
covers the exact C1 condition cache, exact C3 vision cache, and approximate
C5 denoise-step cache without enabling any cache in the parity gates.

## Datasets

| Role | Dataset | Access |
|---|---|---|
| Serving trace / C1 hit rate | `WenhaoWang/VidProM` (1.67M real-user T2V prompts with natural duplicates) | Sampled only when refreshing with `datasets.load_dataset("WenhaoWang/VidProM", split="train", streaming=True)` |
| C5 quality-gate prompts | VBench `all_dimension.txt` from Vchitect/VBench | Sampled only when refreshing |
| Offline default | `benchmarks/caches/fixtures/prompts.json` | Committed seeded sample; the only runtime and test source |

The default commands are hermetic and never access the network. Install the
`bench` extra with `pip install -e '.[bench]'` only to regenerate fixture data.
`python -m benchmarks.caches.workload --refresh --source vidprom` refreshes
VidProM; `--source vbench` refreshes VBench. A refresh is an explicit
maintenance operation, not part of a benchmark run or CI.

## Suites

### Microbenchmarks

`benchmarks.caches.micro` times C1 condition-key hashing, C1 memory-tier
put/get and cold disk get, C3 call-key hashing, and C5 decision/replay
overhead on synthetic tensors. It fills the common `suite` and
`cache_config` context while reporting benchmark-specific latency statistics
in `results/cache-bench/micro.json`.

```bash
python -m benchmarks.caches.micro --results-dir results/cache-bench
```

### Contribution ablation

`benchmarks.caches.ablation` runs the one-factor grid `baseline`, `c1`, `c3`,
`c5`, and `all-exact`. Every cell runs in a separate subprocess, performs a
cold and warm generation, records cache counters, latency, peak VRAM, and
relative latent error, then writes per-cell JSON and
`results/cache-bench/results.csv`. It fills all applicable contract fields,
including `phase`, `e2e_ms`, `vram_peak_gib`, C1/C3 hit/miss counters,
C5 counters, `rms_rel`, `rms_rel_max`, `speedup_vs_baseline`, `verdict`, and
`notes`.

```bash
python -m benchmarks.caches.ablation --results-dir results/cache-bench
```

Without `OMNI_CHECKPOINT`, cells report `weights-absent`. C5 additionally
reports `c5-uncalibrated` until coefficients and a threshold are supplied.

### Serving trace

`benchmarks.caches.serve_trace` replays a deterministic VidProM-derived
request trace against the job API. It writes
`results/cache-bench/serve_trace.csv` and fills request-level `trace_len`,
`repeat_ratio`, `rep`, `phase`, `e2e_ms`, and `expected_hits` fields. Start a
server with this exact immutable profile:

```bash
CUDA_VISIBLE_DEVICES=0 \
OMNI_CHECKPOINT=<h3-snapshot-dir> \
OMNI_STORE_DIR=<moe-store> \
OMNI_STORE_COMPONENTS=transformer,vae,audio_vae \
OMNI_MODEL_ARCH=h3-dense \
OMNI_OPTIMIZATIONS=adaln-host-cache,block-stream,text-encoder-stream,condition-cache \
OMNI_CONDITION_CACHE_DIR=./cache-c1 \
OMNI_JOBS_DIR=./jobs \
python -m omni_infinity.serve
```

Then run:

```bash
python -m benchmarks.caches.serve_trace \
  --repeat-ratio 0.5 --results-dir results/cache-bench
```

Restart the server while retaining `OMNI_CONDITION_CACHE_DIR` and replay the
same trace to measure disk-tier persistence.

### C5 calibration probe

`benchmarks.caches.denoise_probe` records per-step relative-L1 distances for
transformer inputs and outputs. It writes
`results/cache-bench/denoise_probe.json`. Fit a polynomial to those signals,
then pass its coefficients through `--c5-coefficients`, together with
`--c5-threshold`, to the ablation. The probe is calibration data rather than
a scored contract row.

```bash
OMNI_CHECKPOINT=... python -m benchmarks.caches.denoise_probe \
  --results-dir results/cache-bench
```

## Metric contract

`benchmarks/caches/contract.py` freezes field order under the same rule as
the streaming benchmarks: integrations may append fields but must not rename
or reorder existing fields. The tuple is:

```python
FIELDS = (
    "suite",
    "arch",
    "cache_config",
    "dataset",
    "trace_len",
    "repeat_ratio",
    "rep",
    "phase",
    "e2e_ms",
    "vram_peak_gib",
    "c1_hits",
    "c1_misses",
    "c3_hits",
    "c3_misses",
    "c5_computed",
    "c5_skipped",
    "expected_hits",
    "rms_rel",
    "rms_rel_max",
    "speedup_vs_baseline",
    "verdict",
    "notes",
)
```

Exact C1, C3, and all-exact warm rows pass only when `rms_rel == 0` and the
required cache reports at least one hit. C5 warm rows pass only when
`c5_skipped >= 1`, `speedup_vs_baseline > 1`, and
`rms_rel <= rms_rel_max`. Rows marked `weights-absent`, `c5-uncalibrated`, or
`server-down` are `SKIP`, not failures. Baseline, microbenchmark, and serving
trace rows are reporting rows.

## Invariants

The benchmark harness never enables caches in `test_reference_parity.py`,
`test_ref2va_parity.py`, or `test_vdn_parity.py`. Those remain unchanged
bitwise gates. C5 has no default calibration: the only supported route to a
C5 row that is not `c5-uncalibrated` is to run `denoise_probe.py`, fit the
signals, and explicitly pass `--c5-coefficients` and `--c5-threshold`.

## Run everything

```bash
# CPU-only (CI-safe)
python -m pytest tests/test_cache_bench_*.py -q -m "not gpu and not weights"
python -m benchmarks.caches.micro --results-dir results/cache-bench
python -m benchmarks.caches.ablation --dry-run

# GPU (H3 weights)
OMNI_CHECKPOINT=... OMNI_STORE_DIR=... python -m benchmarks.caches.ablation
OMNI_CHECKPOINT=... python -m benchmarks.caches.denoise_probe
# serving trace: start the server above with condition-cache, then
python -m benchmarks.caches.serve_trace --repeat-ratio 0.5
```
