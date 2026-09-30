# C5 Denoise-Step Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skip transformer evaluations inside one generation when a calibrated distance says the step output barely changed, and refuse to run until those coefficients exist.

**Architecture:** A context manager wraps `transformer.forward` for one `generate` call. `mode="output"` replays the last prediction. `mode="residual"` replays `input + cached residual` and rejects H3-shaped outputs that have no matching input. There is no default polynomial.

**Tech Stack:** Python 3.10+, PyTorch. `cache-dit` is an optional import, not a project dependency.

**Spec:** Issue [#24](https://github.com/EfficientMoE/Omni-Infinity/issues/24) item C5, and `denoise_step_cache` in `docs/superpowers/plans/2026-09-30-caches-contract.md`.

## Global Constraints

- Not a registry optimization. Do not add `denoise-cache` to `OPTIMIZATIONS`, `load_cache_optimizations`, or `GenerationRequest`.
- Do not claim `rms_rel=0`. Do not edit parity tests.
- Do not add an H3 coefficient table. Do not add a GPU CLIP/SSIM/PSNR test. That gate waits on a calibration the literature does not publish for MiniMax-H3.
- Do not publish text K or V. The wrapper stores the forward output, or a residual of the named inputs, for this generation only.
- Do not modify `third_party/`, runners, `attach.py`, `serve/`, or `examples/fl2va_smoke.py`.
- If `attach.py` does not import `denoise_step_cache`, stop.
- New Python files start with `# Copyright (c) EfficientMoE.` and `# SPDX-License-Identifier: Apache-2.0`. Ruff line length stays 80.
- Fresh context: this plan and the contract Interfaces. Do not open the other cache plans or draft PR #25.

## Review Focus

- `DenoiseCacheConfig(coefficients=(), threshold=0.2)` raises `ValueError` and the message contains `MiniMax-H3`. Task 1.
- With identical inputs, `total_steps=4`, `warmup_steps=1`, `final_steps=1`, the original forward runs exactly twice. Task 2.
- `calls_per_step=2` keeps two slots: a skip in slot 1 must not replay slot 0's output. Task 2.
- `mode="residual"` on an output tensor whose shape differs from `hidden_states` raises `ValueError` matching `shape`. Task 2.
- A skipped call still runs a `register_forward_hook` callback. Task 2.

## File Structure

Create:

- `omni_infinity/caches/denoise.py`
- `tests/test_denoise_cache.py`
- `docs/caches_c5_denoise.md`

No other files.

---

### Task 1: Config refuses an uncalibrated model

**Files:**
- Create: `omni_infinity/caches/denoise.py`
- Test: `tests/test_denoise_cache.py`

**Interfaces:**
- Consumes: nothing from the other caches.
- Produces: frozen `DenoiseCacheConfig(coefficients: tuple[float, ...], threshold: float, mode: str = "output", signal_name: str = "hidden_states", io_names: tuple[str, ...] = ("hidden_states",), calls_per_step: int = 1, warmup_steps: int = 1, final_steps: int = 1)`.

- [ ] **Step 1: Write the failing test**

Empty `coefficients` raises `ValueError` matching `MiniMax-H3`. `threshold=0` and a negative threshold raise `ValueError` matching `threshold`. `mode="kv"` raises `ValueError` matching `output`. `warmup_steps=0` and `final_steps=0` raise `ValueError` matching `first and last`. `calls_per_step=0` raises `ValueError`. A config with `coefficients=(1.0, 0.0)` and `threshold=0.2` can be constructed and is frozen.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_denoise_cache.py -q`
Expected: FAIL

- [ ] **Step 3: Implement `DenoiseCacheConfig.__post_init__`**

Use those exact messages' substrings. Do not supply a default for `coefficients` or `threshold`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_denoise_cache.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/caches/denoise.py tests/test_denoise_cache.py
git commit -m "feat(caches): refuse an uncalibrated denoise-step cache"
```

---

### Task 2: One-generation forward wrapper

**Files:**
- Modify: `omni_infinity/caches/denoise.py`
- Test: `tests/test_denoise_cache.py`
- Create: `docs/caches_c5_denoise.md`

**Interfaces:**
- Consumes: `DenoiseCacheConfig`, `tree_map`, `tree_tensors`.
- Produces:
  - `DenoiseCacheStats` with `computed: int = 0` and `skipped: int = 0`
  - `@contextmanager denoise_step_cache(transformer, config, total_steps: int)` yielding `DenoiseCacheStats`
  - `enable_cache_dit(target, **kwargs)`

- [ ] **Step 1: Write the failing test**

Use a `torch.nn.Module` whose `forward(self, hidden_states)` returns `hidden_states + 1` and counts calls. Relative L1 is `(current - previous).abs().mean() / previous.abs().mean()`. The polynomial is highest-degree first: `(100.0, 0.0)` maps `x` to `100 * x`.

Assert:

- `total_steps=4`, identical `torch.ones(4)` each call, `threshold=0.2`, `coefficients=(1.0, 0.0)`: original runs 2 times, `stats.computed == 2`, `stats.skipped == 2`, and the skipped return equals the first return.
- The same schedule with `coefficients=(100.0, 0.0)` and inputs that move by a relative L1 of about `0.01` computes every step (`100 * 0.01 > 0.2`).
- `calls_per_step=2`, `total_steps=3`: feed slot 0 a vector of ones and slot 1 a vector of twos, repeated. A skipped slot-1 call equals the previous slot-1 output, not slot 0's output.
- `mode="residual"` with output shape `(2,)` and `hidden_states` shape `(3,)` raises `ValueError` matching `shape` on the second call.
- `mode="residual"` with matching shapes returns `current_input + (previous_output - previous_input)` on a skip.
- Entering `denoise_step_cache` twice on the same module raises `RuntimeError` matching `already enabled`. After the context exits, a later call uses the original forward.
- A `register_forward_hook` counter increments on a skipped call.
- `enable_cache_dit(module)` with no `cache_dit` module raises `ImportError` matching `cache-dit is not installed`. Monkeypatch a fake `cache_dit.enable_cache` and assert it receives the module and kwargs.

Doc `docs/caches_c5_denoise.md` contains: `No published TeaCache or cache-dit table covers MiniMax-H3.`, `This plan does not add a CLIP, SSIM, or PSNR gate.`, `Never publish H3 text K/V from a denoise step.`, `mode="output" is the H3 default.`

Also assert `"denoise-cache" not in registry.OPTIMIZATIONS` and `runner_kwargs_for("h3-dense", ["denoise-cache"])` raises `ValueError`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_denoise_cache.py -q`
Expected: FAIL, wrapper missing.

- [ ] **Step 3: Implement the wrapper**

Step index is `calls // calls_per_step`. Slot index is `calls % calls_per_step`. Steps in `[0, warmup_steps)` and steps in `[total_steps - final_steps, total_steps)` always compute. `total_steps < 1` raises `ValueError`. Resolve `signal_name` from kwargs, else from the positional index of that name in `io_names`, else position 0. On compute, store the output (`mode="output"`) or the per-leaf residual against the paired `io_names` inputs (`mode="residual"`). On skip, replay that store. A residual whose leaf shapes differ from the current inputs raises `ValueError`. Exiting the context restores `forward` the same way C3 restores a vision tower: put back an instance forward that was already there, otherwise delete the instance attribute. Set a module attribute `_omni_denoise_cache_active` for the reentry check and clear it on exit.

`enable_cache_dit` imports `cache_dit` and calls `cache_dit.enable_cache(target, **kwargs)`. Missing package or missing attribute raises `ImportError` with `cache-dit is not installed`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_denoise_cache.py tests/test_cache_contract.py tests/test_registry.py -q`
Expected: PASS

Then ruff check and format check on the new Python files.

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/caches/denoise.py tests/test_denoise_cache.py docs/caches_c5_denoise.md
git commit -m "feat(caches): skip denoise steps only with calibrated coefficients"
```
