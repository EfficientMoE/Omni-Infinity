# Opt-in caches (C1–C5): design and implementation plan

> Refs [#24](https://github.com/EfficientMoE/Omni-Infinity/issues/24).
> This is the implementation plan for the cache tracker. It changes no code.
> Execution lands in later PRs, one phase at a time.

**Goal:** add the five opt-in caches from #24 — exact condition cache (C1),
encoder prefix cache (C2), vision-embedding cache (C3), documented shape
caches (C4), and an approximate denoise-step feature cache (C5) — without
touching the bitwise golden path.

**Architecture:** one new `omni_infinity/caches/` package holding every cache
implementation behind small classes; the registry
(`omni_infinity/registry.py`) exposes each cache as a named optimization the
smoke CLIs, ablation harness, and job server already know how to resolve;
runners consume caches through explicit keyword arguments, mirroring how
`adaln_host_cache` and `stream_text_encoder` are wired today.

**Ground rules (from #24, non-negotiable):**

1. Every cache is **off by default**. Enabling one is a registry
   optimization (`condition-cache`, …) or an explicit runner kwarg.
2. The parity gates — `tests/test_reference_parity.py`,
   `tests/test_ref2va_parity.py`, `tests/test_vdn_parity.py` — never run
   with any of these caches enabled. Identity mode stays bitwise.
3. C1–C3 are **exactness-preserving** (they replay stored tensors, byte for
   byte); they get their own bitwise hit-parity tests, *separate* from the
   golden gates. C5 is **approximate**; it gets a quality gate and never
   claims `rms_rel=0`.
4. No patches to `third_party/vdn-minimax-h3` or to
   `MiniMaxH3ModularPipeline`. C1–C3 wrap inputs the runners already pass;
   C5 is a forward hook, outside `third_party/`.
5. No shot reordering (that is #23 and it is rejected), no NIRVANA-style
   similarity retrieval, no publishing of denoise-step text K/V as a
   cross-request prefix.

---

## What exists today (survey recap)

Condensed from #24; the full table lives in the issue.

| Cache | Where | Cross-request? |
|---|---|---|
| Prompt embeds + packed layout | `encoder_hidden_states` re-fed every denoise step; layout set once per generation | No — next job re-encodes |
| VDN prompt `.pt` | `third_party/vdn-minimax-h3/src/inference/utils/prompt_cache.py` — `prompt_embeds (L, 5120)` bf16, `text_token_tags (L,)`, optional `keyframe_anchors` + `condition_latents` | Only if the caller keeps the file; the job server does not |
| AdaLN host cache | `omni_infinity/adaln.py` (`AdaLNEntry`, `HostResidentAdaLN`), `h3-dense` only | Yes, for weights — not a prompt cache |
| VDN shape caches (BlockMask, gather-index, FLASH compile) | VDN inference stack | Yes for the shape; caption is not part of the key |
| Qwen3-VL KV / vision embeds / text linear-attention state / denoise features | none | No |

The three shipped T2VA examples share only the `[Shot 1]` opener; a prefix
cache cannot see a matching later shot. That is why **C1 keys the entire
condition** and never a prefix.

## The five caches and their contracts

| ID | Name | Exact? | Keyed by | Phase |
|---|---|---|---|---|
| C1 | Exact condition cache | bitwise | sha256(model arch, checkpoint, workflow, full prompt string, ordered image bytes) | 1 |
| C3 | Vision-embedding cache | bitwise | sha256(image bytes) + encoder identity | 2 |
| C2 | Encoder prefix cache | bitwise | leading token blocks, vLLM-style; vision tokens included | 3 (conditional) |
| C4 | VDN shape caches | n/a (already exist) | packed length, frame count, window radius, device | 0 (documented below) |
| C5 | Denoise-step feature cache | **approximate** | adjacent-step modulated residual distance | 4 (gated on calibration) |

Ordering rationale: C1 is the only cross-request reuse that needs neither a
resident encoder nor a KV contract, so it lands first. C3 helps whenever the
same image recurs inside *different* conditions (recurring Ref2VA character
refs) and needs no resident tower either. C2 requires the Qwen3-VL tower to
stay resident, which conflicts with the shipped ~10 GiB
`text-encoder-stream` story, so it is last among the exact caches and gated.
C5 is a different animal entirely (quality-for-speed) and is gated on an
H3-specific calibration measurement.

## New module layout

```text
omni_infinity/caches/
  __init__.py        # facade: ConditionCache, condition_key, VisionEmbedCache
  condition.py       # C1 (Phase 1)
  vision.py          # C3 (Phase 2)
  encoder_prefix.py  # C2 (Phase 3, only if the decision gate passes)
  denoise.py         # C5 (Phase 4, only if calibration passes)
tests/
  test_condition_cache.py   # CPU
  test_vision_cache.py      # CPU
  test_cache_hit_parity.py  # GPU+weights, opt-in, separate from golden gates
docs/opt_in_caches.md       # this file (Phase 0)
```

`omni_infinity/caches` must never be imported by `omni_infinity/runner.py`
at module level — only lazily inside the code paths that the opt-in kwargs
enable, the same pattern `from_pretrained` uses for
`omni_infinity.store` and `omni_infinity.step_overlap` today
(`omni_infinity/runner.py:298-346`).

---

## Phase 0 — this document (no code)

- [x] Survey + design recorded here and in #24.
- [x] **C4 is documented, not expanded** (below) — that closes C4's scope.

### C4: the VDN shape caches, documented

`vdn-hybrid` already caches, per process: the Flex BlockMask, the
gather-index tables, the delta-rule backend selection, and one static FLASH
compile per `seq_len`. Reuse condition: same packed length, frame count,
window radius, device. Consequences worth knowing when operating the job
server:

- A **new caption length** misses the FLASH compile (one `seq_len` per
  process) and the BlockMask entry; the first request at that length pays
  the compile again. This matches SGLang's fixed text-row CUDA-graph bucket
  (the validated Ref2VA profile uses one bucket) — do not capture a
  graph/compile per job.
- The caption *content* is not part of the shape key; only its length is.

No code change is planned for C4. Expanding these caches is explicitly out
of scope of #24.

---

## Phase 1 — C1: exact condition cache (cross-request)

The VDN prompt `.pt` (`prompt_cache.py`) lifted into the runner as a
content-addressed store. Hit condition: the **entire condition** matches —
full prompt string plus every image, in order. A shared `[Shot 1]` opener is
not a hit, by design.

### Task 1.0: hookpoint probe (go/no-go, GPU)

Before any cache code: verify the modular pipeline accepts precomputed
prompt embeddings and reproduces the goldens bitwise. The job API already
reads intermediates out of the modular state
(`_state_value` / `get_intermediate`, `omni_infinity/runner.py:413-424`);
this probe confirms the write direction.

**Files:**
- Create: `tests/test_cache_hit_parity.py` (first test only)

- [ ] **Step 1: write the probe test** (marked `gpu` + `weights`, same
  skip conditions as `tests/test_reference_parity.py`):

```python
@pytest.mark.gpu
@pytest.mark.weights
def test_precomputed_prompt_embeds_reproduce_goldens_bitwise(runner, goldens):
    # Run 1: normal text path, capture the encoded intermediates.
    first = runner.generate("a red ball bouncing", seed=0,
                            num_inference_steps=8, resolution="256p",
                            num_frames=120)
    embeds = runner.last_prompt_embeds  # captured via forward hook in the test
    # Run 2: feed the captured embeddings, skip the text encoder.
    second = runner.generate_from_condition(embeds, seed=0,
                                            num_inference_steps=8,
                                            resolution="256p", num_frames=120)
    assert torch.equal(first.latents, second.latents)
    assert torch.equal(first.audio_latents, second.audio_latents)
```

- [ ] **Step 2: implement the minimal `generate_from_condition` spike** on
  a branch. Two candidate mechanisms, in preference order:
  1. Pass the encoded intermediates into the modular pipeline call
     (modular diffusers state accepts pre-set intermediates).
  2. If (1) is not supported by `MiniMaxH3ModularPipeline`, memoize at the
     text-encoder component boundary: a wrapper module that returns stored
     outputs without running the tower (device-only moves, so bitwise
     holds), installed via `pipeline.update_components`.
- [ ] **Step 3: run on the dev GPU.** Expected: bitwise equal. If **neither**
  mechanism reproduces the goldens bitwise, stop — report on #24 and
  re-scope C1 to the `vdn-hybrid` runner only (VDN's own render path
  already consumes prompt `.pt` files natively).
- [ ] **Step 4: record the winning mechanism in this doc and commit.**

### Task 1.1: key derivation + entry schema

**Files:**
- Create: `omni_infinity/caches/__init__.py`, `omni_infinity/caches/condition.py`
- Test: `tests/test_condition_cache.py` (CPU)

- [ ] **Step 1: write the failing CPU tests:**

```python
from omni_infinity.caches import condition_key

ARGS = dict(model_arch="h3-dense", checkpoint="MiniMaxAI/MiniMax-H3",
            workflow="fl2va")

def test_condition_key_is_deterministic():
    assert condition_key("p", (b"img",), **ARGS) == \
        condition_key("p", (b"img",), **ARGS)

def test_condition_key_orders_images():
    assert condition_key("p", (b"a", b"b"), **ARGS) != \
        condition_key("p", (b"b", b"a"), **ARGS)

def test_shared_shot1_prefix_is_not_a_hit():
    a = condition_key("[Shot 1] opener\n[Shot 2] beach", (), **ARGS)
    b = condition_key("[Shot 1] opener\n[Shot 2] city", (), **ARGS)
    assert a != b  # whole-condition key: shared leading span is invisible

def test_key_covers_model_identity():
    assert condition_key("p", (), model_arch="h3-dense",
                         checkpoint="MiniMaxAI/MiniMax-H3",
                         workflow="fl2va") != \
        condition_key("p", (), model_arch="vdn-hybrid",
                      checkpoint="OpenVDN/vdn-minimax-h3", workflow="t2va")
```

- [ ] **Step 2: run** `pytest tests/test_condition_cache.py -q` — FAIL
  (module missing).
- [ ] **Step 3: implement:**

```python
# omni_infinity/caches/condition.py
CACHE_SCHEMA_VERSION = 1

def condition_key(prompt: str, image_bytes: tuple[bytes, ...] = (), *,
                  model_arch: str, checkpoint: str, workflow: str) -> str:
    h = hashlib.sha256()
    for part in (str(CACHE_SCHEMA_VERSION), model_arch, checkpoint,
                 workflow, prompt):
        h.update(part.encode("utf-8")); h.update(b"\x00")
    for blob in image_bytes:  # prompt first, then images in order (#24 C1)
        h.update(hashlib.sha256(blob).digest())
    return h.hexdigest()
```

- [ ] **Step 4: run tests** — PASS. **Step 5: commit**
  (`feat(caches): condition cache key derivation`).

### Task 1.2: on-disk store with atomic writes + LRU cap

Entry payload mirrors the VDN prompt cache so the two stay
interconvertible: `prompt_embeds (L, 5120)` bf16, `text_token_tags (L,)`
int64, optional `keyframe_anchors: list[str]` +
`condition_latents: list[fp32 tensor]`, plus `schema`, `created`, and the
key inputs echoed for audit.

**Files:**
- Modify: `omni_infinity/caches/condition.py`
- Test: `tests/test_condition_cache.py` (CPU)

- [ ] **Step 1: failing tests** — `test_round_trip_is_byte_exact`
  (put → get returns `torch.equal` tensors, dtype preserved),
  `test_put_is_atomic` (a crashed write leaves no `<digest>.pt`, only a
  temp file that `get` ignores), `test_lru_evicts_oldest_under_max_bytes`
  (3 entries, cap sized for 2, oldest-`mtime` entry gone),
  `test_get_miss_returns_none`.
- [ ] **Step 2: implement `ConditionCache`:**

```python
class ConditionCache:
    """Content-addressed store: <dir>/<digest[:2]>/<digest>.pt"""
    def __init__(self, cache_dir: str, max_bytes: int = 64 << 30): ...
    def get(self, key: str) -> dict | None       # torch.load(weights_only=True)
    def put(self, key: str, entry: dict) -> None # tmp file + os.replace
    def _evict_lru(self) -> None                 # by mtime until under cap
```

- [ ] **Step 3: tests pass; commit**
  (`feat(caches): on-disk condition store, atomic + LRU`).

### Task 1.3: runner integration (both archs)

**Files:**
- Modify: `omni_infinity/runner.py` (`from_pretrained`: add
  `condition_cache_dir: str | None = None`; `generate`: key lookup before
  encode, `put` after a miss, using the Task 1.0 mechanism on hit)
- Modify: `omni_infinity/arch/vdn.py` (same kwarg; VDN consumes the entry
  via its native prompt-cache format — `load_prompt` contract from
  `third_party/vdn-minimax-h3/src/inference/utils/prompt_cache.py`,
  imported nowhere: the *format* is mirrored, the module is not patched)
- Test: extend `tests/test_cache_hit_parity.py` (GPU) and
  `tests/test_runner_api.py` (CPU: kwarg accepted, default `None`, cache
  module not imported when off)

- [ ] **Step 1: CPU guard test** — with `condition_cache_dir=None`,
  `sys.modules` gains no `omni_infinity.caches` entry after a (mocked)
  generate; parity gates therefore cannot silently pick the cache up.
- [ ] **Step 2: GPU hit-parity test** —
  `test_condition_cache_hit_reproduces_goldens_bitwise`: cold run (miss →
  entry written) then warm run (hit → encoder never loaded, assert via the
  text-encoder load counter), both bitwise equal to
  `tests/fixtures/goldens`.
- [ ] **Step 3: implement, run both, commit**
  (`feat(runner): opt-in cross-request condition cache (C1)`).

### Task 1.4: registry + serve wiring

**Files:**
- Modify: `omni_infinity/registry.py` — new entry, same shape as
  `adaln-host-cache` (`omni_infinity/registry.py:82-89`):

```python
"condition-cache": OptimizationSpec(
    name="condition-cache",
    description=("Cross-request exact condition cache: layer-50 prompt "
                 "embeds + keyframe/reference latents keyed by the full "
                 "prompt string and ordered image bytes."),
    supported_archs=("h3-dense", "vdn-hybrid"),
    runner_kwargs_by_arch=MappingProxyType({
        "h3-dense": _kw(condition_cache=True),
        "vdn-hybrid": _kw(condition_cache=True),
    }),
),
```

- Modify: `omni_infinity/serve/app.py` settings — `OMNI_CONDITION_CACHE_DIR`
  (default unset = off) and `OMNI_CONDITION_CACHE_MAX_BYTES` (default
  `64GiB`), following the existing `OMNI_*` env pattern.
- Test: `tests/test_registry.py` (resolve profile with `condition-cache`),
  `tests/test_job_api.py` (server passes the dir through; off by default).

- [ ] Steps: failing tests → implement → pass →
  commit (`feat(serve): condition cache env wiring`).
- [ ] README: one paragraph under *Job-serving API* stating the cache is
  opt-in, exact-hit-only, and excluded from the golden gates.

**Phase 1 acceptance:** repeated identical job on the serve profile skips
text-encoder load entirely (log line + load-counter assertion), latents
bitwise-equal to the cold run; all three golden gates untouched and green.

---

## Phase 2 — C3: vision-embedding cache

Content hash of each image → vision-tower output rows (vLLM-Omni's
"multimodal encoder cache", mechanism 2 in #24). Looked up whenever the
tower must run because the *condition* (C1) missed but an individual image
recurs — the recurring-character Ref2VA case. Independent of any block
boundary.

**Files:** create `omni_infinity/caches/vision.py`; test
`tests/test_vision_cache.py` (CPU: key = sha256(image bytes) + encoder
identity, round-trip byte-exact, LRU — same skeleton as Task 1.2); extend
`tests/test_cache_hit_parity.py` (GPU: Ref2VA run with two refs, second run
replaces one ref → cache hits the unchanged ref, output bitwise-equal to an
uncached run of the same condition); registry entry `vision-embed-cache`
(both archs) + `OMNI_VISION_CACHE_DIR`.

- [ ] Same TDD step sequence as Tasks 1.1–1.4.
- [ ] Hook point: the same encoder boundary Task 1.0 selected — per-image
  memoization inside the visual patch-embedder wrapper
  (`_stream_text_encoder` already locates `pipeline.text_encoder.visual`,
  `omni_infinity/runner.py:131-136`). Device-only moves ⇒ bitwise.
- [ ] Commit (`feat(caches): vision-embedding cache (C3)`).

**Phase 2 acceptance:** the GPU hit-parity test above, plus CPU suite green.

---

## Phase 3 — C2: encoder prefix cache (conditional — decision gate first)

vLLM-style hashing of leading token blocks, complete blocks only. Vision
tokens sit in front of the text, so `<Picture 1>` matches only when the
image bytes match *and* it is first. **No reordering of `[Shot N]` blocks
to manufacture a prefix** (#23); no ContextPilot `.pth` or chat proxy.

**Decision gate (must pass before any C2 code):**

- [ ] C2 requires the Qwen3-VL tower resident — it *conflicts* with
  `text-encoder-stream` and the ~10 GiB memory story. Measure on the dev
  GPU: resident-encoder VRAM cost vs. the measured per-request encode time
  C1 already eliminates for exact repeats. Publish the numbers on #24.
- [ ] Proceed only if a serving profile exists where the tower is resident
  anyway *and* real workloads show partial-prefix repeats that C1 misses.
  Otherwise close C2 on #24 as "won't do — C1 covers the observed reuse."

**If the gate passes:**

- [ ] Registry: `encoder-prefix-cache` entry whose resolution **fails
  loudly** when combined with `text-encoder-stream` — add an optional
  `conflicts_with: tuple[str, ...] = ()` field to `OptimizationSpec` and
  enforce it in `runner_kwargs_for` (`omni_infinity/registry.py:139-158`),
  with `tests/test_registry.py::test_conflicting_optimizations_raise`.
- [ ] `omni_infinity/caches/encoder_prefix.py`: block-hash table over
  leading token blocks (block size fixed per process), complete blocks
  only, per-block KV rows pinned on host; miss runs the tower from the
  first uncached block.
- [ ] GPU test: prefix hit reproduces the full-encode `prompt_embeds`
  bitwise for a two-request pair sharing a true leading block; a pair
  sharing only a *later* shot must miss (asserted).

---

## Phase 4 — C5: denoise-step feature cache (approximate, calibration-gated)

TeaCache / Cache-DiT-class reuse of block residuals across adjacent steps
of **one** sample. Never cross-request; never publishes text K/V (in this
DiT the packed text rows live in the residual stream and change with the
latents — #24, "How vLLM-Omni does it", mechanism 3 caveat).

**Calibration gate (must pass before any skip logic):**

- [ ] Create `benchmarks/denoise_cache_probe.py`: run the standard 8-step
  256p/120f smoke, record per-step timestep-modulated input L1 distance and
  true output change per transformer block (forward hooks, the pattern of
  `_denoising_progress`, `omni_infinity/runner.py:43-64`). Both shipped
  coefficient tables (vLLM-Omni, SGLang) exclude MiniMax-H3; SGLang's
  uncalibrated Wan2.2 flag no-ops — shipping that is the failure mode this
  gate exists to prevent. Publish the correlation plot on #24.
- [ ] Proceed only if the probe shows a usable predictor at ≥ 8 steps
  (SCM-class methods want ≥ 8; the 8-step smoke is the minimum, not a free
  win).

**If the gate passes:**

- [ ] Prefer integrating the `cache-dit` library or TeaCache's hook over a
  third implementation (both serving stacks call, not re-implement).
  `omni_infinity/caches/denoise.py` holds only the hook install/uninstall
  and the H3 coefficient table.
- [ ] First and last steps always compute. Opt-in via registry
  `denoise-feature-cache`; `conflicts_with` nothing, but the runner refuses
  it when goldens capture is requested.
- [ ] **Quality gate, not parity gate:** new
  `tests/test_denoise_cache_quality.py` (GPU) asserting `rms_rel` vs the
  goldens under a threshold *chosen from the probe data and recorded here*,
  plus wall-clock speedup > 1. It must never assert bitwise equality, and
  the three golden gates must never enable it.

---

## Rejected alternatives (from #24, recorded so they stay rejected)

- RadixAttention / vLLM prefix caching dropped in front of the job API —
  the job API is not chat completions; the denoiser owns no such KV.
- Publishing denoise-step text KV à la HunyuanImage3 — H3 text rows change
  with the latent; rejected until a measurement shows otherwise.
- NIRVANA latent similarity retrieval — approximate, cross-prompt; a
  time-marked H3 prompt is a timeline. Out of scope until C1–C3 exist.
- Shot reordering to manufacture a prefix — #23, rejected; the writing
  guide requires chronological `[Shot N]` order.

## Reference index

Implementation references, by cache: **C1** — VDN prompt `.pt`
(`prompt_cache.py`); **C2** — PagedAttention (SOSP '23,
arXiv:2309.06180), vLLM-Omni AR stage-output prefix cache; **C3** —
vLLM-Omni multimodal encoder cache; **C5** — TeaCache (arXiv:2411.19108),
cache-dit (DBCache/TaylorSeer/SCM), MagCache (arXiv:2506.09045), Δ-DiT
(arXiv:2406.01125). Full annotated list: #24.
