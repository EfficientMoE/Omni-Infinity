# Cache benchmarks (issue #24)

This suite tests and quantifies the contribution of each cache level listed
under the README's [Opt-in caches](../README.md#opt-in-caches) section. The
cache semantics are defined by the [C1 condition-cache](caches_c1_condition.md)
and [C5 denoise-cache](caches_c5_denoise.md) designs and tracked in
[issue #24](https://github.com/EfficientMoE/Omni-Infinity/issues/24).

The benchmarks never enable caches in reference-parity gates. They are
separate opt-in measurements.

## Datasets

| Role | Dataset | Access |
|---|---|---|
| Serving trace and C1 hit rate | `WenhaoWang/VidProM`, a real-user text-to-video prompt collection with natural duplicates | Sampled only when refreshing the fixture with `datasets.load_dataset(..., streaming=True)` |
| C5 quality prompts | VBench `all_dimension.txt` from Vchitect/VBench | Sampled only when refreshing the fixture |
| Offline default | `benchmarks/caches/fixtures/prompts.json` | Committed seeded sample; the only runtime and test source |

CI and ordinary benchmark runs never access the network. To regenerate a pool,
install the refresh-only dependency with `pip install -e '.[bench]'`, then run:

```bash
python -m benchmarks.caches.workload --refresh --source vidprom
python -m benchmarks.caches.workload --refresh --source vbench
```

## Suites

### Micro

`benchmarks.caches.micro` times C1 condition-key hashing, memory put/get and
cold disk get, C3 vision-call key hashing, and C5 decision/replay overhead on
synthetic tensors. It reports suite, cache identity, repetitions, device,
latency percentiles, and cache-specific counters in
`results/cache-bench/micro.json`.

```bash
python -m benchmarks.caches.micro --results-dir results/cache-bench
```

### Ablation

`benchmarks.caches.ablation` runs `{baseline,c1,c3,c5,all-exact}` as isolated
subprocess cells. Each measured cell performs a cold and warm generation,
records `e2e_ms`, `vram_peak_gib`, C1/C3 hit and miss counters, C5 step
counters, `rms_rel`, its quality bound, and warm speedup. Per-cell JSON and
the frozen rows in `results/cache-bench/results.csv` are written under the
results directory.

```bash
python -m benchmarks.caches.ablation --dry-run
OMNI_CHECKPOINT=... OMNI_STORE_DIR=... \
  python -m benchmarks.caches.ablation
```

Cells without weights are `SKIP` with `weights-absent`. C5 has no invented
default calibration, so it is `SKIP` with `c5-uncalibrated` unless calibrated
coefficients and a threshold are supplied.

### Serve trace

`benchmarks.caches.serve_trace` replays a seeded VidProM trace through the job
API. It fills request index, expected-hit status, cold/warm phase, trace length,
repeat ratio, and end-to-end latency in
`results/cache-bench/serve_trace.csv`. The server profile and the request's
ordered optimization list must match exactly:

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

python -m benchmarks.caches.serve_trace --repeat-ratio 0.5
```

Restart the server against the same `OMNI_CONDITION_CACHE_DIR` and replay the
same trace to measure persistent disk-tier recovery.

### Denoise probe

`benchmarks.caches.denoise_probe` records one input and output relative-L1
distance per denoising transformer call in
`results/cache-bench/denoise_probe.json`:

```bash
OMNI_CHECKPOINT=... python -m benchmarks.caches.denoise_probe
```

Fit a polynomial to these records, then pass the coefficients, threshold, and
quality bound to `benchmarks.caches.ablation --c5-coefficients ...`. This
probe-to-fit workflow is the only path to a measured C5 row that is not
`c5-uncalibrated`.

## Metric contract

`benchmarks/caches/contract.py` freezes field order under the same append-only
rule as the streaming benchmark. The tuple is:

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
`rms_rel <= rms_rel_max`. `weights-absent`, `c5-uncalibrated`, and
`server-down` are `SKIP`, not failures. Baselines and micro/serve-trace rows
are reports rather than accuracy claims.

## Run all checks

```bash
# CPU-only (CI-safe)
python -m pytest tests/test_cache_bench_*.py -q -m "not gpu and not weights"
python -m benchmarks.caches.micro --results-dir results/cache-bench
python -m benchmarks.caches.ablation --dry-run

# GPU (H3 weights)
OMNI_CHECKPOINT=... OMNI_STORE_DIR=... \
  python -m benchmarks.caches.ablation
OMNI_CHECKPOINT=... python -m benchmarks.caches.denoise_probe
# Start the condition-cache server above, then:
python -m benchmarks.caches.serve_trace --repeat-ratio 0.5
```
