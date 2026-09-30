# C2 Encoder Prefix Cache Deferral Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record why the Qwen3-VL prefix cache is not in this stack, and pin a test that fails if someone adds it quietly.

**Architecture:** No cache module. The document holds the reuse rule and the function signature a later resident-encoder change must implement. This PR's test deletes that signature's file if it appears.

**Tech Stack:** Markdown and pytest. No new dependencies.

**Spec:** Issue [#24](https://github.com/EfficientMoE/Omni-Infinity/issues/24) item C2. C2 is optional and only valid when the Qwen3-VL tower stays resident. It fights the current path, which releases the encoder.

## Global Constraints

- Do not create `omni_infinity/caches/prefix.py`.
- Do not set `use_cache=True` anywhere under `omni_infinity/`.
- Do not reorder `[Shot N]` blocks. Do not vendor ContextPilot or load a ContextPilot `.pth`.
- Do not modify `third_party/`, runners, `attach.py`, `serve/`, or parity tests.
- The only files this plan creates are listed below.
- Fresh context: this plan only. Do not open the other cache plans or draft PR #25.
- New test file starts with `# Copyright (c) EfficientMoE.` and `# SPDX-License-Identifier: Apache-2.0`.

## Review Focus

- A partial trailing block must not be stored. The signature below returns hashes only for complete leading blocks. Task 1.
- Picture tokens stay inside the token prefix. An image match that is not at the start is not a prefix hit. Task 1.
- `omni_infinity/` must not contain `use_cache=True`. Task 1.
- `omni_infinity/caches/prefix.py` must not exist when this PR merges. Task 1.

## File Structure

Create:

- `docs/caches_c2_prefix.md`
- `tests/test_cache_c2_deferred.py`

No other files.

---

### Task 1: Pin the deferral and the future signature

**Files:**
- Create: `docs/caches_c2_prefix.md`
- Test: `tests/test_cache_c2_deferred.py`

**Interfaces:**
- Consumes: nothing.
- Produces: no importable function. The document specifies this signature for a later PR, and this PR must not implement it:

```python
def prefix_blocks(token_ids: tuple[int, ...], block_size: int) -> list[bytes]:
    """Hash complete leading blocks of token_ids.

    Drop the tail if it is shorter than block_size. Picture tokens are
    already inside token_ids. Return an empty list when the tower is not
    resident. Do not reorder shots.
    """
```

- [ ] **Step 1: Write the failing test**

```python
def test_encoder_prefix_cache_is_not_implemented():
    assert not Path("omni_infinity/caches/prefix.py").exists()
    for path in Path("omni_infinity").rglob("*.py"):
        text = path.read_text()
        assert "use_cache=True" not in text
        assert "use_cache = True" not in text


def test_c2_note_states_the_reuse_rule():
    text = Path("docs/caches_c2_prefix.md").read_text()
    for sentence in (
        "Deferred until the Qwen3-VL tower stays resident.",
        "A hit is a complete leading block only.",
        "Vision tokens in front of the text are part of the prefix.",
        "Picture 1 matches only when the image bytes match and it is first.",
        "Do not reorder [Shot N] blocks.",
        "Do not install ContextPilot.",
        "def prefix_blocks(token_ids: tuple[int, ...], block_size: int) -> list[bytes]:",
        "This plan does not implement prefix_blocks.",
    ):
        assert sentence in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache_c2_deferred.py -q`
Expected: FAIL on the missing document. If the `use_cache=True` assertion fails, stop and report the file. Do not weaken the assertion.

- [ ] **Step 3: Write `docs/caches_c2_prefix.md`**

Include every sentence from the test verbatim, plus the signature from Interfaces. State that a later implementation may call C3's `VisionEmbedCache` for images inside a block that still has to run, and that this PR does not import `caches.vision`. State that vLLM's block hash is the reuse rule being copied, and that partial blocks are not stored.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cache_c2_deferred.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docs/caches_c2_prefix.md tests/test_cache_c2_deferred.py
git commit -m "docs: defer the encoder prefix cache until the tower stays resident"
```
