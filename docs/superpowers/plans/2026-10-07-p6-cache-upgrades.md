# P6 — C5 denoise-cache upgrade + C2 encoder-prefix revisit

Tracking: [#42](https://github.com/EfficientMoE/Omni-Infinity/issues/42) (P6).
Refs: [caches_c5](../../caches_c5_denoise.md), [caches_c2](../../caches_c2_prefix.md),
[cache_benchmarks](../../cache_benchmarks.md), [caches contract](2026-09-30-caches-contract.md),
[t2v survey](../../t2v_optimization_survey.md) §3A.

## Objective

(a) Upgrade C5's skip indicator from the current calibrated-coefficient
signal to a TeaCache-style **timestep-embedding-modulated input delta with
polynomial rescaling** (survey: 1.8–2× at matched quality on comparable
DiTs), optionally with a TaylorSeer-style residual extrapolator instead of
plain reuse. (b) Un-defer C2 with a minimal exact encoder-prefix cache.
All work stays inside the issue-#24 contract: opt-in, off by default,
quantified in cache_benchmarks.

## C5 upgrade approach

1. **Indicator v2**: compute rel-L1 of timestep-modulated noisy input
   (modulated by H3's AdaLN timestep embedding — we already materialize it
   in `adaln.py`) instead of raw signal; fit the polynomial rescaler on the
   existing C5 calibration probe set; accumulate-until-threshold trigger.
   Donor logic: TeaCache (ali-vilab) coefficient fitting scripts.
2. **Approximator option**: `--denoise-cache-mode {reuse,taylor1}` —
   taylor1 extrapolates the cached residual (cache-dit TaylorSeer order 1).
   Keeps eval-skip decision identical; only the substitute output differs.
3. **FBCache variant (cheap A/B)**: first-transformer-block residual gate
   as an alternative indicator — trivially computable in our runner since
   blocks stream one at a time; compare in the same calibration probe.
4. **Compile interaction rule** (coordinate with P2): skip decision stays
   outside any compiled/graphed region; cached steps bypass graphs.
5. Re-run the frozen metric suite: microbench, one-factor ablation,
   VidProM serving trace, calibration probe → new rows in cache_benchmarks.

## C2 revisit (minimal viable)

- Scope: exact-prefix reuse of Qwen3-VL encoder hidden states keyed on
  (tokenized prompt prefix, image refs hash) — SHA256 keying like C1,
  thread-safe LRU like C3 (`caches/vision.py` is the template).
- Admission/eviction policy pattern: MoE-Infinity
  `core/prefetch/expert_residency.h` / `expert_prefetcher.py` (phase-aware
  admission, transient entries) — port the *policy shape*, not the code.
- Value test first: measure encoder time share under text-encoder-stream
  (64-layer stream ≈ large share of short jobs; C2 skips it entirely on
  repeat prompts — the multi-prompt demo is the natural beneficiary).
- Exact-match only in v1 (no partial-prefix splicing); bitwise by
  construction → can pass the strict gate.

## Tasks

- [ ] C5 indicator v2 + polynomial fit on existing probe set
- [ ] taylor1 mode + FBCache-style alternative; A/B on probe
- [ ] cache_benchmarks rows (speedup vs quality curves, thresholds)
- [ ] C2 exact cache + `encoder-cache` registry opt + LRU bounds
- [ ] C2 contribution row (VidProM trace + multi-prompt demo timing)
- [ ] Contract doc updates (caches_c5/caches_c2) + #42 checkboxes

## Verification

C5: quality-vs-NFE-skipped curves on the frozen suite; refuse-to-run
without calibration preserved. C2: bitwise parity on hit path, LRU memory
bound respected, measured hit-rate on the demo workload.

## Risks

- Indicator v2 coefficients are model-specific → keep refusal-without-
  calibration semantics; ship fitting script, not numbers.
- Audio-video joint DiT may need per-modality thresholds → expose per-
  modality threshold override, default shared.
- C2 host memory growth → entry cap + byte budget like C3.
