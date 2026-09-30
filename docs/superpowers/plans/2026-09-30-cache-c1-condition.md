# C1 Exact Condition Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse a whole H3 condition — layer-50 prompt embeds, token tags, and keyframe/reference VAE latents — across requests when the prompt, image bytes, and canvas all match.

**Architecture:** `prepare` hashes the condition, replays a host-resident entry into an encoder-less pipeline view on a hit, and records the modular state on a miss. A shared `[Shot 1]` is not a hit. Any surprise fails open to the normal encode path.

**Tech Stack:** Python 3.10+, PyTorch, existing `hashlib` / `torch.save`. No new dependencies.

**Spec:** Issue [#24](https://github.com/EfficientMoE/Omni-Infinity/issues/24) item C1, and the Interfaces in `docs/superpowers/plans/2026-09-30-caches-contract.md`.

## Global Constraints

- Opt-in. This module's `OPTIMIZATION` is what turns `condition_cache=True` on. Do not change default server optimizations.
- A hit requires the entire condition. Do not implement prefix matching or `[Shot N]` reordering.
- Do not modify `third_party/`, `runner.py`, `arch/vdn.py`, `registry.py`, `serve/`, `examples/fl2va_smoke.py`, or any parity test.
- Do not edit `attach.py`. It already imports `ConditionCache` and `prepare`. If those names are absent from `attach.py`, stop; the contract is not on this branch.
- CPU tests with fake pipelines. The diffusers structural test uses `pytest.importorskip` and must not download weights.
- New Python files start with `# Copyright (c) EfficientMoE.` and `# SPDX-License-Identifier: Apache-2.0`. Ruff line length stays 80.
- Fresh context: read this plan and the contract Interfaces only. Do not open the other cache plans or draft PR #25.

## Review Focus

- `[Shot 1]` shared and `[Shot 2]` different must hash differently. Task 1.
- Same prompt and images at `height=256` and `height=512` must hash differently. Task 1.
- `ReferenceRunner:/ckpt-a` and `ReferenceRunner:/ckpt-b` must hash differently. Task 1.
- A hit must not call the full pipeline, and the request `generator` object in `call_kwargs` must be the same object the caller passed. Task 3.
- An unhashable reference and a pipeline that cannot be reduced must return `hit is False` and the original kwargs. Task 3.

## File Structure

Create:

- `omni_infinity/caches/condition.py`
- `tests/test_condition_cache.py`
- `docs/caches_c1_condition.md`

No other files.

---

### Task 1: Condition key

**Files:**
- Create: `omni_infinity/caches/condition.py`
- Test: `tests/test_condition_cache.py`

**Interfaces:**
- Consumes: `update_hash_for_value` from `omni_infinity.caches._tensor_tree`.
- Produces: `condition_key(namespace: str, prompt: str, media: tuple = (), *, height: int, width: int, num_frames: int) -> str`. `CACHE_SCHEMA_VERSION = 1`.

- [ ] **Step 1: Write the failing test**

```python
def test_identical_condition_gives_identical_key():
    kwargs = dict(
        namespace="ReferenceRunner:/ckpt",
        prompt="[Shot 1] a",
        media=(b"png", None),
        height=368,
        width=640,
        num_frames=120,
    )
    assert condition_key(**kwargs) == condition_key(**kwargs)


def test_shared_shot1_opener_is_not_a_hit():
    common = dict(
        namespace="ReferenceRunner:/ckpt",
        media=(),
        height=368,
        width=640,
        num_frames=120,
    )
    left = condition_key(prompt="[Shot 1] same\n[Shot 2] left", **common)
    right = condition_key(prompt="[Shot 1] same\n[Shot 2] right", **common)
    assert left != right


def test_canvas_and_checkpoint_and_none_slots_are_in_the_key():
    base = dict(
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(b"a", None),
        height=368,
        width=640,
        num_frames=120,
    )
    assert condition_key(**base) != condition_key(**{**base, "height": 512})
    assert condition_key(**base) != condition_key(
        **{**base, "namespace": "ReferenceRunner:/other"}
    )
    assert condition_key(**base) != condition_key(
        **{**base, "media": (None, b"a")}
    )
```

Also assert the digest is 64 hex characters, and that `media=(object(),)` raises `TypeError` from `condition_key` itself.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_condition_cache.py -q`
Expected: FAIL, `condition_key` missing.

- [ ] **Step 3: Implement `condition_key`**

Hash, in order, the tag `f"omni-condition-v{CACHE_SCHEMA_VERSION}"`, then `namespace`, `prompt`, `tuple(media)`, then `height`, `width`, `num_frames`, all through `update_hash_for_value`. Return `hexdigest()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_condition_cache.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/caches/condition.py tests/test_condition_cache.py
git commit -m "feat(caches): hash the whole condition for C1"
```

---

### Task 2: Host LRU and disk tier

**Files:**
- Modify: `omni_infinity/caches/condition.py`
- Test: `tests/test_condition_cache.py`

**Interfaces:**
- Consumes: `tree_map`, `tree_nbytes` from `_tensor_tree`.
- Produces:
  - `ConditionEntry.to(device) -> dict`
  - `ConditionCache(max_entries: int = 8, max_bytes: int | None = None, cache_dir=None, capture: tuple[str, ...] = DEFAULT_CAPTURE, required: tuple[str, ...] = ("prompt_embeds",))`
  - `DEFAULT_CAPTURE = ("prompt_embeds", "text_token_tags", "condition_latents", "audio_condition_latents")`
  - `get(key) -> ConditionEntry | None`, `put(key, values) -> ConditionEntry | None`, `stats() -> dict`

- [ ] **Step 1: Write the failing test**

Assert `put` without `prompt_embeds` returns `None` and `get` misses. A stored tensor's device is `cpu` even if `put` was given a CUDA-less tensor created on CPU with a non-cpu... use `torch.zeros(2).to("meta")` only if `.to("cpu")` works; otherwise store `torch.zeros(2) + 1` and assert `.device.type == "cpu"`. `to("cpu")` returns equal values. `stats()["hits"]` increments on `get` of a present key and `stats()["misses"]` increments on a missing key.

`max_entries=1` keeps the newest key. `max_bytes` smaller than the second entry drops the oldest. A disk round trip through `tmp_path` returns equal tensors after a new `ConditionCache(cache_dir=tmp_path)`. A truncated `<key>.pt` is a miss and does not raise. `weights_only=True` is required: the test writes a non-tensor object with `torch.save` and asserts `get` returns `None`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_condition_cache.py -q`
Expected: FAIL, cache class missing.

- [ ] **Step 3: Implement the cache**

`max_entries < 1` raises `ValueError`. `required` names missing from `capture` raise `ValueError`. Store CPU clones under a `threading.Lock`. Disk path is `cache_dir / f"{key}.pt"`. Write to a temp file in that directory and `os.replace` it. Load with `torch.load(..., weights_only=True)`. Any load exception is a miss. `put` drops `None` values. If `prompt_embeds` is absent after that drop, store nothing.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_condition_cache.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/caches/condition.py tests/test_condition_cache.py
git commit -m "feat(caches): store condition entries on CPU and disk"
```

---

### Task 3: Hit, miss, and fail-open replay

**Files:**
- Modify: `omni_infinity/caches/condition.py`
- Test: `tests/test_condition_cache.py`
- Create: `docs/caches_c1_condition.md`

**Interfaces:**
- Consumes: `ConditionCache`, `condition_key`.
- Produces:
  - `ENCODE_BLOCKS = ("text_encoder", "vae_encoder")`
  - `build_conditioned_pipeline(pipeline, skip_blocks=ENCODE_BLOCKS)`
  - `prepare(pipeline, cache, *, namespace, prompt, media, height, width, num_frames, call_kwargs) -> ConditionReplay`
  - `ConditionReplay.hit: bool`, `.pipeline` (`None` means the caller pipeline), `.call_kwargs() -> dict`, `.observe(state) -> None`
  - `OPTIMIZATION`: name `condition-cache`, archs `h3-dense` and `vdn-hybrid`, both kwargs `{"condition_cache": True}`

- [ ] **Step 1: Write the failing test**

Fake pipeline: `blocks.sub_blocks` maps `text_encoder` and `vae_encoder` to objects with `inputs`, plus a `denoise` block whose `inputs` include a spec named `prompt_embeds`. `components` is `{"transformer": object()}`. Calling the fake records `kwargs`.

```python
def test_miss_captures_and_hit_skips_encoders():
    cache = ConditionCache()
    first = prepare(pipeline, cache, namespace="ReferenceRunner:/ckpt",
                     prompt="p", media=(b"img",), height=368, width=640,
                     num_frames=120, call_kwargs={"prompt": "p", "generator": gen})
    assert first.hit is False
    assert first.pipeline is None
    assert first.call_kwargs()["generator"] is gen
    first.observe(_state(prompt_embeds=torch.ones(2)))
    second = prepare(pipeline, cache, namespace="ReferenceRunner:/ckpt",
                      prompt="p", media=(b"img",), height=368, width=640,
                      num_frames=120, call_kwargs={"prompt": "p", "generator": gen})
    assert second.hit is True
    assert second.pipeline is not pipeline
    assert torch.equal(second.call_kwargs()["prompt_embeds"], torch.ones(2))
    assert "prompt" not in second.call_kwargs()
    assert second.call_kwargs()["generator"] is gen
```

`_state` is `types.SimpleNamespace`. `observe` must also accept a mapping with the same names. A different prompt is a miss. `media=(object(),)` returns `hit is False` and does not raise. A pipeline with no `blocks` returns `hit is False` on the second call even after a previous `observe` stored an entry: the stored entry exists, but `build_conditioned_pipeline` returns `None`, and `prepare` then fails open (does not inject).

`test_loader_publishes_condition_cache` calls `registry.load_cache_optimizations()` and expects `runner_kwargs_for("h3-dense", ["condition-cache"]) == {"condition_cache": True}` and the same for `vdn-hybrid`.

`test_h3_blocks_can_drop_encoders` does `pytest.importorskip("diffusers")`, builds `MiniMaxH3Blocks()`, and asserts `text_encoder` and `vae_encoder` are sub-block names and that the reduced pipeline does not contain those names while sharing any component object that both pipelines already have. If the installed diffusers uses different block names, stop and report the names. Do not change `ENCODE_BLOCKS` just to skip.

Doc test: `docs/caches_c1_condition.md` contains `A shared [Shot 1] opener is not a hit.`, `use_cache=False`, `keyframe_encode_seed`, and `the request generator is not consumed`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_condition_cache.py -q`
Expected: FAIL, `prepare` missing.

- [ ] **Step 3: Implement replay**

`build_conditioned_pipeline` uses `SequentialPipelineBlocks.from_blocks_dict`, drops `skip_blocks`, calls `init_pipeline()`, and `update_components` with the shared non-`None` components. Any exception or missing blocks returns `None`.

`prepare` computes the key. `TypeError` from `condition_key` returns a miss replay that still calls `observe` as a no-op. On a cache hit, if `build_conditioned_pipeline` returns `None`, return a miss replay and do not inject. On a usable hit, `call_kwargs()` is the cached tensors filtered to names the reduced pipeline declares, plus the caller's non-condition keys except `prompt`, `image`, `last_image`, and `references`. Keep `generator`. `observe` on a hit is a no-op. On a miss, `observe` reads the capture names off the state and `put`s them.

`OPTIMIZATION` uses `registry.OptimizationSpec` and `MappingProxyType`. Import `registry` inside the assignment only if a module-level import cycles; a function-level import at the bottom of the module is fine.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_condition_cache.py tests/test_cache_contract.py tests/test_registry.py -q`
Expected: PASS. `test_static_optimization_set_excludes_caches` still passes.

Then ruff check and format check on `omni_infinity/caches/condition.py` and `tests/test_condition_cache.py`.

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/caches/condition.py tests/test_condition_cache.py docs/caches_c1_condition.md
git commit -m "feat(caches): replay an exact condition without re-encoding"
```
