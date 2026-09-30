# C4 VDN Shape-Cache Notes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record how VDN's existing shape caches behave, and leave that code where it is.

**Architecture:** One document and one test that the document says the required facts. No runtime cache, no registry name, no runner flag.

**Tech Stack:** Markdown and pytest. No new dependencies.

**Spec:** Issue [#24](https://github.com/EfficientMoE/Omni-Infinity/issues/24) item C4. C4 is "Leave them. Document that a new caption length misses the inference FLASH compile (one `seq_len` per process) and the BlockMask entry. Not a feature to expand in this issue."

## Global Constraints

- Do not modify `third_party/vdn-minimax-h3` or any file under `third_party/`.
- Do not add `omni_infinity/caches/shape.py` or a registry optimization.
- Do not edit runners, `attach.py`, `serve/`, or parity tests.
- The only files this plan creates are listed below.
- Fresh context: this plan only. Do not open the other cache plans or draft PR #25.
- New test file starts with `# Copyright (c) EfficientMoE.` and `# SPDX-License-Identifier: Apache-2.0`.

## Review Focus

- A reader must see that the caption is not part of the BlockMask key. Task 1.
- A reader must see that a new token length misses a process-wide FLASH compile. Task 1.
- The diff must not contain `third_party/`. Task 1.

## File Structure

Create:

- `docs/caches_c4_shape.md`
- `tests/test_cache_c4_doc.py`

No other files.

---

### Task 1: Write the note and pin its sentences

**Files:**
- Create: `docs/caches_c4_shape.md`
- Test: `tests/test_cache_c4_doc.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the document. No Python API.

- [ ] **Step 1: Write the failing test**

```python
def test_c4_note_states_the_reuse_rule():
    text = Path("docs/caches_c4_shape.md").read_text()
    for sentence in (
        "Flex BlockMask, gather-index, and delta-rule backend caches stay in third_party/vdn-minimax-h3.",
        "The caption is not part of the BlockMask key.",
        "A new caption length misses the inference FLASH compile.",
        "One seq_len is compiled per process.",
        "This issue does not add a shape cache.",
        "This plan does not modify third_party.",
    ):
        assert sentence in text


def test_c4_adds_no_shape_module():
    assert not Path("omni_infinity/caches/shape.py").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache_c4_doc.py -q`
Expected: FAIL, document missing.

- [ ] **Step 3: Write `docs/caches_c4_shape.md`**

Use the sentences verbatim. In the same file, state the operational consequence in prose: attention still runs on a shape hit; only the mask, gather index, and delta-rule backend are reused. A second job whose packed text length differs from the first job in the process misses the FLASH compile and the BlockMask entry. Do not propose a new key, a new cache, or a patch under `third_party/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cache_c4_doc.py -q`
Expected: PASS

Confirm `git diff --name-only origin/main...HEAD` lists only this plan's files plus the contract plan already on the branch, and does not list `third_party/`.

- [ ] **Step 5: Commit**

```bash
git add docs/caches_c4_shape.md tests/test_cache_c4_doc.py
git commit -m "docs: record VDN shape-cache limits without changing them"
```
