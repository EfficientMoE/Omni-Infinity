# P6 Cache Upgrades — Inherited Wisdom

## [2026-10-08] Research findings (Atlas orchestrator pre-work)

### Plan source
Actual plan file: `docs/superpowers/plans/2026-10-07-p6-cache-upgrades.md` (NOT `.sisyphus/plans/`).
Tracking: GitHub issue #42 (P6). Part of a P1-P7 SM120 roadmap (see commit 83d1950).

### Existing cache architecture (C1/C3/C5 reference patterns)

- **C1 condition cache**: `omni_infinity/caches/condition.py`. SHA256 keying via
  `condition_key(namespace, prompt, media, height, width, num_frames)` ->
  hexdigest. Schema version constant `CACHE_SCHEMA_VERSION = 1`. Text-only
  FL2VA handling via `_is_text_only()` checking `image/last_image/references`
  all None, with `_TEXT_ONLY_SKIP_BLOCKS` skip-block list.
- **C3 vision cache**: `omni_infinity/caches/vision.py`. Thread-safe
  entry-bounded LRU: `threading.Lock()` + `collections.OrderedDict` +
  `move_to_end()` + `popitem(last=False)` eviction. `max_entries=4` default,
  NO byte-budget enforcement (only tracks `bytes` stat, doesn't cap it).
  Key: SHA256 of `("omni-vision-v1", namespace, args, sorted kwargs)`.
  Wraps vision tower's `.forward`, stores on CPU, materializes to device on
  hit via `_map_output_tensors`.
- **C5 denoise cache**: `omni_infinity/caches/denoise.py`. Current skip
  decision: relative-L1 distance between consecutive `hidden_states` ->
  Horner-method polynomial `_polynomial(coefficients, distance)` -> skip iff
  `P(distance) <= threshold` (per-call decision, NOT accumulated). Refuses to
  run without calibration (`ValueError` mentioning "MiniMax-H3" if
  `coefficients` empty). `mode="output"` (default) replays last prediction;
  `mode="residual"` replays `input + cached_residual`. Wraps
  `transformer.forward` via a `register_forward_pre_hook` guardian pattern
  (see the unmerged fix commit `5252cda` on local branch
  `plan/cache-c5-denoise` — NOT yet in main — "survive forward rebinding";
  worth cherry-picking logic since it fixes a real re-binding bug: uses
  `forward_delegate` list + `@wraps` + `register_forward_pre_hook` instead of
  bare `MethodType` assignment).
- **Registry wiring**: `omni_infinity/registry.py` ->
  `load_cache_optimizations()` dynamically imports
  `omni_infinity.caches.condition` and `omni_infinity.caches.vision` (NOT
  denoise — C5 is a `generate(denoise_cache=...)` kwarg, not a registry
  optimization). Each cache module exports module-level `OPTIMIZATION =
  OptimizationSpec(name=..., description=..., supported_archs=(...),
  runner_kwargs_by_arch=MappingProxyType({...}))`.
- **Attach pattern**: `omni_infinity/caches/attach.py` — `attach_caches()`
  and `bind_generation()` are the ONLY places runners call into a cache.
  Missing module + set flag => loud `RuntimeError`, never silent no-op.
- **CLI flags**: `examples/fl2va_smoke.py` lines ~60-64:
  `--condition-cache`, `--condition-cache-dir`, `--vision-cache`,
  `--denoise-cache-coefficients`, `--denoise-cache-threshold`.
- **Serve env vars**: `omni_infinity/serve/models.py` + `serve/app.py` also
  wire cache flags (`OMNI_CONDITION_CACHE_DIR` etc.) — second wiring point,
  don't forget it when adding `--encoder-cache`/`OMNI_ENCODER_CACHE`.
- **AdaLN timestep embedding**: `omni_infinity/adaln.py` ->
  `HostResidentAdaLN.forward(temb)` — `activated = F.silu(temb)` then
  `F.linear(...)`. This materialized `temb`/`activated` is the modulation
  signal C5-v2 needs (TeaCache modulates the INPUT by AdaLN's shift/scale,
  not just the raw hidden_states).
- **Benchmark harness**: `benchmarks/caches/{contract.py,ablation.py,
  micro.py,denoise_probe.py,serve_trace.py,workload.py}`. Frozen field order
  in `contract.py` `FIELDS` tuple is APPEND-ONLY (never reorder/remove).
  `CACHE_CONFIGS` tuple + `verdict_for(row)` dispatch by `cache_config` name.
  C5 exact rows need `c5_computed`/`c5_skipped` fields; exact caches need
  `rms_rel == 0` and `hits >= 1` on warm phase.

### C2 deferral — MUST FIX before implementing

`tests/test_cache_c2_deferred.py` currently asserts:
1. `omni_infinity/caches/prefix.py` does NOT exist
2. No `.py` file under `omni_infinity/` contains `use_cache=True` / `use_cache = True`
3. `docs/caches_c2_prefix.md` contains specific "deferred" sentences
   including a `prefix_blocks(token_ids, block_size) -> list[bytes]` stub
   signature and "This plan does not implement prefix_blocks."

**This test WILL FAIL once C2 lands and MUST be deleted/rewritten** as part
of the C2 implementation task, in the SAME commit/task (not a follow-up).
Do not leave it red.

Also note: the OLD C2 design (pre-deferral, in
`docs/caches_c2_prefix.md`/`.sisyphus/plans/pr32-cache-c2-prefix.md`) sketched
a **block-based prefix cache** (vLLM-style `prefix_blocks()` hashing complete
leading token blocks). The NEW P6 plan explicitly wants a SIMPLER
**exact-match-only v1** (whole tokenized-prompt-prefix + image-refs-hash,
SHA256 keyed like C1, no partial-prefix splicing). Do not resurrect the
block-based design — follow the P6 plan's simpler scope.

### TeaCache / TaylorSeer / FBCache algorithms (exact formulas)

**TeaCache** (`github.com/ali-vilab/TeaCache`):
```
modulated_inp = t2i_modulate(norm(hidden_states), shift_msa, scale_msa)  # AdaLN-modulated
rel_l1 = mean(|modulated_inp_t - modulated_inp_{t-1}|) / mean(|modulated_inp_{t-1}|)
rescaled = poly4(rel_l1)  # degree-4 polynomial, model-specific fitted coefficients
accumulated_rel_l1_distance += rescaled
if accumulated_rel_l1_distance < threshold:
    skip (reuse cached output)
else:
    compute; accumulated_rel_l1_distance = 0   # reset on compute
# Always compute at first and last timestep regardless of accumulator.
```
Key difference from current C5: CURRENT C5 does NOT accumulate — it computes
a fresh per-call polynomial(distance) and skips if under threshold, no
running accumulator, no reset-on-compute. P6's "accumulate-until-threshold
trigger" is a genuine semantic upgrade, not just a signal swap.

**TaylorSeer order-1** (`github.com/vipshop/cache-dit`,
`caching/cache_contexts/calibrators/taylorseer.py`):
```
# finite-difference derivative estimate from cached history
dY/dt ≈ (Y_current - Y_prev) / (step_current - step_prev)
# extrapolate at elapsed steps since last full compute:
Y_approx(t) = Y_0 + dY/dt * elapsed          # order-1
# order-2 adds: + (1/2) * d2Y/dt2 * elapsed**2
```
This REPLACES plain substitution (`output = cached_output`) with an
extrapolated prediction — `--denoise-cache-mode taylor1` should extrapolate
the cached RESIDUAL (not the raw output) per the plan text ("taylor1
extrapolates the cached residual").

**FBCache** (`github.com/chengzeyi/ParaAttention`,
`first_block_cache/utils.py`):
```
first_block_residual_t = hidden_states_after_block0 - input_to_block0
diff = mean(|first_block_residual_t - first_block_residual_{t-1}|) / mean(|first_block_residual_t|)
skip iff diff < residual_diff_threshold   # typical 0.06-0.12, per-step decision (no accumulator)
```
H3's transformer streams blocks one at a time (block-level group offload),
so block-0's residual is "trivially computable" per the plan — compute it
inline in the same forward wrapper without extra overhead.

### MoE-Infinity admission/eviction policy shape (for C2 LRU)

Port the POLICY SHAPE only (not C++ code) from
`EfficientMoE/MoE-Infinity` `core/prefetch/expert_residency.cpp` +
`moe_infinity/memory/expert_prefetcher.py`:
- **Phase-aware admission**: analogous mapping for C2 — treat a cold/first
  admission for a given session as lenient (admit even under pressure,
  mark "transient" = don't actually cache, just let it pass through) vs a
  repeat/follow-up admission (strict — requires an eviction victim, reject
  if none available). Maps MoE-Infinity's PREFILL=TRANSIENT_ON_PRESSURE vs
  DECODE=CACHE(strict) distinction onto "first use of this cache" vs
  "cache under memory pressure."
- **LRU eviction ranking**: not-currently-leased (no refcount), oldest
  `last_access_time` first — same shape as C3's `OrderedDict.move_to_end()`
  + `popitem(last=False)`, just confirm no additional "lease protection"
  needed for C2 since encoder outputs aren't held across concurrent
  requests the way experts are (single serialized worker, see README job
  API notes — `OMNI_WORKERS` must stay `1`). So full MoE-Infinity lease
  machinery is likely OVERKILL for C2 — port only entry-cap + BYTE BUDGET
  (C3 lacks byte-budget enforcement; C2's risk note explicitly wants both
  entry cap AND byte budget, which is a genuine enhancement beyond C3).

### Stray/unmerged work found (not yet in main)
- Local branch `plan/cache-c5-denoise` has ONE unmerged commit `5252cda`
  "fix(caches): survive forward rebinding in the denoise-step cache" — fixes
  a real bug in the forward-wrapping guardian pattern. Cherry-pick or
  reimplement this fix as part of the C5-v2 work since v2 touches the same
  wrapping code.
- Remote branch `origin/plan/cache-benchmarks` has one unmerged commit
  `74882b9` "docs: plan the cache benchmark & test suite" — doc-only, check
  if superseded by current `docs/cache_benchmarks.md` before reusing.

### Environment note
This repo has MULTIPLE concurrent Atlas/Sisyphus sessions running on sibling
P-series plans (P1-P4 fp8, P2 cudagraph, P3 attention, P5 nvfp4, P7
multigpu) that share `.sisyphus/boulder.json`. Completion-gate reminders
referencing `2026-10-07-p2-cudagraph-compile.md` or other sibling plans are
CROSS-TALK from those sessions — ignore them. This session's ground truth
is `docs/superpowers/plans/2026-10-07-p6-cache-upgrades.md`.

## [2026-10-08] C5 Indicator v2 implementation findings

- Confirmed MiniMax-H3's first-block modulation in installed diffusers
  `models/transformers/transformer_minimax_h3.py:355-361`: obtain
  `shift_msa, scale_msa, ... = adaln_proj(temb)`, then compute
  `norm1(hidden_states) * (1 + scale_msa[index]) + shift_msa[index]`.
- Final `DenoiseCacheConfig` v2 fields are `indicator` (`"raw"` default,
  `"teacache"` opt-in) and `accumulate` (`False` default). This keeps the
  legacy raw/per-call path unchanged and makes signal selection independent
  from accumulation semantics for later indicator variants.
- The H3 TeaCache signal uses the first block's real `proj_in`, `norm1`, and
  `adaln_proj` modules plus `timestep_indices`, `token_tags`, and
  `video_indices`; no polynomial coefficients were added. The installed
  modular denoiser calls the transformer with these values as keyword args.
- `torch.compile` does not occur under `omni_infinity/`. The cache decision is
  a plain Python function guarded with the public `torch.compiler.disable`
  decorator when available, while cached calls bypass transformer forward.
- Ported commit `5252cda`'s `forward_delegate` + `functools.wraps` + pre-hook
  guardian pattern because an offload/rebinding wrapper can replace
  `forward` during a computed call.

## [2026-10-08] C2 exact encoder-prefix cache implementation findings

- Encoder wrap site: diffusers 0.40
  `diffusers/modular_pipelines/minimax_h3/encoders.py:91-98` calls
  `text_encoder.model(input_ids=..., attention_mask=...,
  mm_token_type_ids=..., use_cache=False, output_hidden_states=True,
  **vision_kwargs)`. C2 therefore wraps `text_encoder.model.forward`, not the
  top-level language-model head.
- Key design: SHA256 `omni-encoder-v1` over the encoder-model namespace, full
  token-ID tensor, and ordered vision tensors normalized through C1's
  `stable_image_bytes()`. This is whole-presentation exact matching only; no
  block hashes or partial-prefix splicing.
- Admission: `max_entries=4` and an 8 GiB default `max_bytes` bound share one
  locked OrderedDict LRU. Hits promote entries; pressure evicts oldest first.
  An output larger than the entire byte budget has no viable victim set and is
  transient (returned but not stored), matching MoE-Infinity's
  TRANSIENT_ON_PRESSURE policy shape without leases/refcounts.

## [2026-10-08] Per-commit review verdicts (elm/gpt-5.6-sol)

- 094ff9f (C5 indicator v2): REQUEST-CHANGES — P0 device mismatch under block
  streaming (teacache signal called offloaded submodules directly); P1 v2 not
  reachable from CLI/attach; P1 probe still raw-signal (deferred to Track C by
  design); P1 "main checkout dirty" = cross-session state, out of scope.
- b0589e6 (fix: device alignment + CLI wiring): REQUEST-CHANGES — adaln_proj
  (stock param-bearing module) still unaligned; wiring approved.
- d4a799c (fix: adaln_proj bridge + dtype folding): APPROVE.
- 02738c9 (C2 cache): review #1 timed out pre-verdict; material finding: key
  hashed only input_ids + vision values.
- 9a030b1 (fix: hash all non-vision inputs): folded into combined review.
- combined C2 review: REQUEST-CHANGES — vision slot-name collision
  (pixel_values=t vs image_grid_thw=t); "mixed C5 changes" = diff-range
  artifact, those were separately reviewed in b0589e6.
- abee426 (fix: hash vision slot names): APPROVE.

## Model-failure fallback log

- deep/ultrabrain categories (elm/gpt-5.6-sol via bot server): ProviderModelNotFoundError
  (stale catalog in long-running server) → switched to `opencode run -m elm/gpt-5.6-sol`
  fresh CLI processes for implementation workers; works.
- unspecified-high (anthropic/claude-opus-4-8): quota exhausted mid-run → same fallback.
- openai/gpt-5.2 via kimaki send: Incorrect API key → killed workers, relaunched on elm.
- opencode run --agent build workers delegating to subagents hit broken providers →
  relaunched with explicit "no delegation, work directly" instruction; works.

## [2026-10-08] C5 Taylor1 + FBCache implementation findings

- Installed diffusers 0.40.0 exposes
  `MiniMaxH3TransformerBlock.forward(hidden_states, temb, adaln_indices,
  rotary_emb, attention_mask=None)`. Every required value can be constructed
  from the whole-transformer call boundary: the three modality inputs and
  indices build the projected packed sequence, `timestep` builds `temb`,
  `timestep_indices` plus `token_tags` build the AdaLN rows, and
  `position_ids` builds RoPE. Therefore the implementation uses the faithful
  FBCache signal `block0(packed, ...) - packed`, not a wrapper-level proxy.
- The FBCache signal repeats the transformer's projection/token-refiner prelude
  and first block solely to make the gate decision. Each parameter-bearing
  stage uses the same `_module_target` device/dtype bridge as TeaCache so it is
  compatible with host-resident/block-streamed weights. The config docstring
  records that FBCache's usual non-accumulating threshold behavior is
  `accumulate=False` with identity polynomial coefficients `(1, 0)`; other
  combinations remain intentionally allowed.
- Taylor1 tracks the substituted value, finite-difference derivative, compute
  call index, and elapsed calls independently in every `calls_per_step` slot.
  Output mode extrapolates the cached output; residual mode extrapolates the
  cached residual and adds it to the current input. A first skip and any
  computed-value tree shape change fall back to reuse.
