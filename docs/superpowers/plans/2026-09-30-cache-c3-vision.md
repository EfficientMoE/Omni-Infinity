# C3 Vision-Embedding Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skip the Qwen3-VL vision tower when a later request repeats the same tower inputs, even if the prompt text changed.

**Architecture:** Replace the tower's instance `forward` with a content-hash lookup. The key is one whole call, not one image inside the call. Values live on CPU and move back to the caller's device on hit.

**Tech Stack:** Python 3.10+, PyTorch. No new dependencies.

**Spec:** Issue [#24](https://github.com/EfficientMoE/Omni-Infinity/issues/24) item C3, and `enable_vision_cache` in `docs/superpowers/plans/2026-09-30-caches-contract.md`.

## Global Constraints

- Opt-in through `OPTIMIZATION` name `vision-cache`. Do not change default optimizations.
- Do not split a multi-image call into per-image entries. That split belongs to C2.
- Do not modify `third_party/`, runners, `registry.py`, `serve/`, `examples/fl2va_smoke.py`, `attach.py`, or any parity test.
- If `attach.py` does not import `enable_vision_cache` from `omni_infinity.caches.vision`, stop.
- CPU tests. Fake modules only. No checkpoint.
- New Python files start with `# Copyright (c) EfficientMoE.` and `# SPDX-License-Identifier: Apache-2.0`. Ruff line length stays 80.
- Fresh context: this plan and the contract Interfaces. Do not open the other cache plans or draft PR #25.

## Review Focus

- Two calls with equal tensor inputs call the original `forward` once. Task 2.
- Changing the second tensor of a two-tensor call is a miss. Task 2.
- `close()` removes an instance `forward` that this wrapper installed, and restores one that was already there. Task 2.
- A tuple `(embeds, deepstack)` round-trips with equal tensors on CPU storage and is returned on the input tensor's device. Task 2.
- A tower with no `visual` and no `model.visual` raises `AttributeError`. Task 1.

## File Structure

Create:

- `omni_infinity/caches/vision.py`
- `tests/test_vision_cache.py`
- `docs/caches_c3_vision.md`

No other files.

---

### Task 1: Cache and tower lookup

**Files:**
- Create: `omni_infinity/caches/vision.py`
- Test: `tests/test_vision_cache.py`

**Interfaces:**
- Consumes: `tree_map`, `tree_nbytes`, `update_hash_for_value`.
- Produces:
  - `VisionEmbedCache(max_entries: int = 4)`
  - `get(key) -> Any | None`, `put(key, value) -> None`, `stats() -> dict`
  - `_visual_module(text_encoder)`

- [ ] **Step 1: Write the failing test**

`max_entries=1` evicts the older key. `stats` reports `hits`, `misses`, `entries`, and `bytes`. `max_entries < 1` raises `ValueError`.

```python
def test_visual_attribute_is_found_on_the_encoder_or_its_model():
    tower = torch.nn.Linear(1, 1)
    assert _visual_module(SimpleNamespace(visual=tower)) is tower
    assert _visual_module(SimpleNamespace(model=SimpleNamespace(visual=tower))) is tower
    with pytest.raises(AttributeError, match="visual"):
        _visual_module(SimpleNamespace())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vision_cache.py -q`
Expected: FAIL

- [ ] **Step 3: Implement the cache and `_visual_module`**

LRU via `OrderedDict` and a `threading.Lock`. `put` stores the value as given; the wrapper in Task 2 moves tensors to CPU before `put`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vision_cache.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/caches/vision.py tests/test_vision_cache.py
git commit -m "feat(caches): add the vision-embedding LRU"
```

---

### Task 2: Forward wrapper

**Files:**
- Modify: `omni_infinity/caches/vision.py`
- Test: `tests/test_vision_cache.py`
- Create: `docs/caches_c3_vision.md`

**Interfaces:**
- Consumes: `VisionEmbedCache`, `_visual_module`.
- Produces:
  - `VisionCacheController.close() -> None`
  - `enable_vision_cache(text_encoder, cache: VisionEmbedCache | None = None, *, max_entries: int = 4) -> VisionCacheController`
  - `OPTIMIZATION` name `vision-cache`, archs `h3-dense` and `vdn-hybrid`, kwargs `{"vision_cache": True}`

- [ ] **Step 1: Write the failing test**

Build a `torch.nn.Module` with `forward` returning `args[0] + 1` and a call counter. Set it as `encoder.visual`.

```python
def test_repeated_call_skips_the_tower():
    controller = enable_vision_cache(encoder)
    first = encoder.visual(torch.ones(2))
    second = encoder.visual(torch.ones(2))
    assert torch.equal(first, second)
    assert calls["n"] == 1


def test_a_changed_second_tensor_is_a_miss():
    enable_vision_cache(encoder)
    encoder.visual(torch.ones(2), torch.zeros(2))
    encoder.visual(torch.ones(2), torch.ones(2))
    assert calls["n"] == 2
```

Also assert: a tuple output `(tensor, tensor)` round-trips; after the call the cached leaf `.device.type == "cpu"`; the returned hit is on the input's device (`cpu` in this test). `close()` on a module that had no instance `forward` leaves `"forward" not in module.__dict__`. `close()` on a module that already had `module.forward = custom` restores `custom`. Two controllers constructed with the same `VisionEmbedCache` share one entry. `registry.runner_kwargs_for("vdn-hybrid", ["vision-cache"]) == {"vision_cache": True}`.

Doc sentences in `docs/caches_c3_vision.md`: `The key is one tower call, not one image inside the call.`, `A different prompt with the same image is a hit.`, `C2 owns per-image splitting.`

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vision_cache.py -q`
Expected: FAIL, `enable_vision_cache` missing.

- [ ] **Step 3: Implement the wrapper**

Key tag is `omni-vision-v1`, then positional args, then kwargs in sorted-key order, through `update_hash_for_value`. On hit, `tree_map` tensors to the first tensor device found in the args and kwargs; if no tensor is present, return the cached value unchanged. On miss, call the original, `put` a `detach().to("cpu")` tree, and return the original output. Remember whether `forward` was in `module.__dict__` before replacement. `close()` is idempotent.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vision_cache.py tests/test_cache_contract.py tests/test_registry.py -q`
Expected: PASS

Then ruff check and format check on the new Python files.

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/caches/vision.py tests/test_vision_cache.py docs/caches_c3_vision.md
git commit -m "feat(caches): skip a repeated vision-tower call"
```
