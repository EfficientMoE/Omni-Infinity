# Opt-in Cache Stack Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the shared hooks the five issue-#24 caches plug into, without implementing any cache.

**Architecture:** Cache modules stay optional imports. `registry.runner_kwargs_for` consults `load_cache_optimizations()` after the static `OPTIMIZATIONS` dict, so `tests/test_registry.py` keeps its exact four-name set. `attach.py` is the only place runners call into a cache. A missing module and a set flag is a loud error, not a silent no-op.

**Tech Stack:** Python 3.10+, existing pytest, no new dependencies.

**Spec:** GitHub issue [#24](https://github.com/EfficientMoE/Omni-Infinity/issues/24). Draft PR #25 is not a base and is not a patch to cherry-pick.

## Global Constraints

- Nothing in this stack is on by default. `ServerSettings.optimizations` stays `("adaln-host-cache", "block-stream", "text-encoder-stream")`.
- Static `registry.OPTIMIZATIONS` stays exactly `adaln-host-cache`, `fp8`, `block-stream`, `text-encoder-stream`. Do not edit `tests/test_registry.py`'s exact-set assertion.
- `condition-cache` and `vision-cache` are the only new `GenerationRequest` optimization names. `denoise-cache` is rejected by that model. C5 is a `generate(..., denoise_cache=)` argument, not a registry optimization.
- C2 and C4 have no registry entry and no runner flag.
- Do not modify `third_party/`. Do not edit `tests/test_reference_parity.py`, `tests/test_ref2va_parity.py`, or `tests/test_vdn_parity.py`.
- CPU tests only. Do not load a checkpoint.
- New Python files start with `# Copyright (c) EfficientMoE.` and `# SPDX-License-Identifier: Apache-2.0`. Ruff line length stays 80.
- This PR implements only the files listed below. Do not add `caches/condition.py`, `caches/vision.py`, `caches/denoise.py`, or `caches/prefix.py`.

## Review Focus

- A default server profile must not pass `condition_cache`, `vision_cache`, or `denoise_cache` into `from_pretrained`. Task 4.
- `runner_kwargs_for("h3-dense", ["condition-cache"])` raises `ValueError` when the loader returns nothing, and succeeds when a stub spec is returned. Task 2.
- Setting `condition_cache=True` or `vision_cache=True` or a non-`None` `denoise_cache` while the module is missing raises `RuntimeError` whose message contains `not installed`. Task 3.
- The same prompt under two checkpoint strings must not share a namespace. Task 3 stores `cache_namespace`.
- Parity test files must not mention these caches. Task 5.

## Stack

Land this branch before any cache branch. The cache PRs use this branch as their base and are parallel with each other:

| Plan | Branch | Owns |
|---|---|---|
| this file | `plan/caches-contract` | hooks below |
| `2026-09-30-cache-c1-condition.md` | `plan/cache-c1-condition` | `caches/condition.py` |
| `2026-09-30-cache-c3-vision.md` | `plan/cache-c3-vision` | `caches/vision.py` |
| `2026-09-30-cache-c5-denoise.md` | `plan/cache-c5-denoise` | `caches/denoise.py` |
| `2026-09-30-cache-c4-shape.md` | `plan/cache-c4-shape` | `docs/caches_c4_shape.md` |
| `2026-09-30-cache-c2-prefix.md` | `plan/cache-c2-prefix` | deferral only |

`Refs #24`. Do not write `Closes #24`.

## File Structure

Create:

- `omni_infinity/caches/__init__.py` — package marker, no cache imports.
- `omni_infinity/caches/_tensor_tree.py` — hash and tensor-tree helpers.
- `omni_infinity/caches/attach.py` — loader calls and generation binding.
- `tests/test_cache_contract.py`
- `tests/test_tensor_tree.py`

Modify:

- `omni_infinity/registry.py` — `load_cache_optimizations`, and `runner_kwargs_for` looks there second.
- `omni_infinity/runner.py` — `ReferenceRunner.from_pretrained` and `generate` gain the kwargs below and call `attach`.
- `omni_infinity/arch/vdn.py` — same for `VdnRunner`.
- `omni_infinity/serve/models.py` — extend the optimization `Literal`.
- `omni_infinity/serve/app.py` — `condition_cache_dir` setting, passed through only when the resolved kwargs already contain `condition_cache=True`.
- `examples/fl2va_smoke.py` — the three flags below.
- `tests/test_server_settings.py` — one assertion, and the env key list.
- `README.md` — one short section. Do not document cache behavior beyond the table in Task 6.

---

### Task 1: Tensor-tree helpers

**Files:**
- Create: `omni_infinity/caches/_tensor_tree.py`
- Create: `omni_infinity/caches/__init__.py`
- Test: `tests/test_tensor_tree.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `update_hash_for_value(hasher, value) -> None`
  - `tree_map(fn, value)`
  - `tree_nbytes(value) -> int`
  - `tree_tensors(value) -> list`

- [ ] **Step 1: Write the failing test**

```python
def test_hash_tags_keep_string_and_int_apart():
    assert _digest("1") != _digest(1)
    assert _digest({"b": 1, "a": 2}) == _digest({"a": 2, "b": 1})


def test_none_slots_do_not_shift():
    assert _digest((None, "x")) != _digest(("x", None))


def test_tree_map_moves_tensor_leaves_only():
    moved = tree_map(lambda t: t + 1, {"a": torch.zeros(1), "b": None})
    assert moved["b"] is None
    assert int(moved["a"]) == 1


def test_unhashable_value_raises_type_error():
    with pytest.raises(TypeError):
        _digest(object())
```

`_digest` hashes with `hashlib.sha256` and `update_hash_for_value`. Also assert a `torch.uint8` tensor's digest changes when one byte changes, and `tree_nbytes` of that tensor equals `tensor.nbytes`. A `PIL.Image` hashes as its PNG bytes: two images with the same pixels match, a recolored image does not.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tensor_tree.py -q`
Expected: FAIL, module or function missing.

- [ ] **Step 3: Implement the helpers**

Type tags, each followed by a payload: `None`; `bool` as `0`/`1`; `int` as decimal text; `float` as `struct.pack("<d", float(value))`; `str` as utf-8; `bytes` as themselves; `list`/`tuple` in order with a length prefix; `dict` with sorted `str` keys. `torch.Tensor` tags `str(dtype)`, `tuple(shape)`, and `cpu().contiguous().numpy().tobytes()`. `PIL.Image.Image` tags PNG bytes from `BytesIO`. Every other type raises `TypeError`.

`tree_map` rebuilds `list`, `tuple`, and `dict`, passes `None` through, calls `fn` on `torch.Tensor`, and returns every other leaf unchanged. `tree_nbytes` sums `tensor.nbytes` and ignores other leaves. `tree_tensors` lists tensor leaves in that same walk order.

`caches/__init__.py` is a docstring only. It must not import a cache module.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tensor_tree.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/caches tests/test_tensor_tree.py
git commit -m "feat(caches): add shared tensor-tree helpers"
```

---

### Task 2: Optional optimization loader

**Files:**
- Modify: `omni_infinity/registry.py`
- Test: `tests/test_cache_contract.py`

**Interfaces:**
- Consumes: `OptimizationSpec` already in `registry.py`.
- Produces: `load_cache_optimizations() -> dict[str, OptimizationSpec]`. `runner_kwargs_for` checks `OPTIMIZATIONS`, then this dict.

- [ ] **Step 1: Write the failing test**

```python
def test_static_optimization_set_excludes_caches():
    assert set(registry.OPTIMIZATIONS) == {
        "adaln-host-cache",
        "fp8",
        "block-stream",
        "text-encoder-stream",
    }


def test_missing_loader_entries_are_unknown(monkeypatch):
    monkeypatch.setattr(registry, "load_cache_optimizations", lambda: {})
    with pytest.raises(ValueError, match="condition-cache"):
        registry.runner_kwargs_for("h3-dense", ["condition-cache"])


def test_loader_spec_merges_like_a_static_optimization(monkeypatch):
    spec = OptimizationSpec(
        name="condition-cache",
        description="stub",
        supported_archs=("h3-dense",),
        runner_kwargs_by_arch=MappingProxyType(
            {"h3-dense": MappingProxyType({"condition_cache": True})}
        ),
    )
    monkeypatch.setattr(
        registry, "load_cache_optimizations", lambda: {"condition-cache": spec}
    )
    assert registry.runner_kwargs_for("h3-dense", ["condition-cache"]) == {
        "condition_cache": True
    }
    with pytest.raises(ValueError, match="vdn-hybrid"):
        registry.runner_kwargs_for("vdn-hybrid", ["condition-cache"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache_contract.py -q`
Expected: FAIL, `load_cache_optimizations` missing.

- [ ] **Step 3: Implement `load_cache_optimizations() -> dict[str, OptimizationSpec]`**

Import these names inside the function, not at module import: `omni_infinity.caches.condition`, `omni_infinity.caches.vision`. `ModuleNotFoundError` skips that module. A module without `OPTIMIZATION` is skipped. Do not import `denoise` or `prefix`. `runner_kwargs_for` looks up the static dict first, then this dict, then raises `ValueError(f"unknown optimization {name!r}")`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cache_contract.py tests/test_registry.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/registry.py tests/test_cache_contract.py
git commit -m "feat(caches): resolve optional cache optimizations"
```

---

### Task 3: Runner attach points

**Files:**
- Create: `omni_infinity/caches/attach.py`
- Modify: `omni_infinity/runner.py` (`ReferenceRunner.from_pretrained`, `ReferenceRunner.generate`)
- Modify: `omni_infinity/arch/vdn.py` (`VdnRunner.from_pretrained`, `VdnRunner.generate`)
- Test: `tests/test_cache_contract.py`

**Interfaces:**
- Consumes: Task 1 helpers are not used here.
- Produces:
  - `attach_caches(runner, *, condition_cache: bool, condition_cache_dir: str | None, vision_cache: bool, cache_namespace: str) -> None`
  - `bind_generation(runner, *, prompt: str, media: tuple, height: int, width: int, num_frames: int, call_kwargs: dict, denoise_cache, total_steps: int, transformer) -> GenerationBinding`
  - `GenerationBinding` with `pipeline`, `call_kwargs: dict`, `observe(state) -> None`, and a context manager `denoise()` that yields `None` when `denoise_cache is None`.

- [ ] **Step 1: Write the failing test**

Use a tiny runner stand-in with `pipeline` and `from_pretrained` left unused. Assert:

- `attach_caches(..., condition_cache=False, vision_cache=False, cache_namespace="ReferenceRunner:/ckpt")` sets `condition_cache` to `None`, `vision_cache_controller` to `None`, and `cache_namespace` to that string.
- The same call with `condition_cache=True` raises `RuntimeError` matching `not installed`.
- `vision_cache=True` raises `RuntimeError` matching `not installed`.
- `bind_generation` with `denoise_cache=None` returns the same `call_kwargs` dict contents, the same pipeline, and `observe` is a no-op. `denoise()` yields `None`.
- `bind_generation` with `denoise_cache=object()` raises `RuntimeError` matching `not installed`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache_contract.py -q`
Expected: FAIL, `attach` missing.

- [ ] **Step 3: Implement attach and call it from both runners**

`attach_caches`: when a flag is false, store `None`. When `condition_cache` is true, import `ConditionCache` from `omni_infinity.caches.condition` and store `ConditionCache(cache_dir=condition_cache_dir)` on `runner.condition_cache`. When `vision_cache` is true, import `enable_vision_cache` from `omni_infinity.caches.vision` and store its return value on `runner.vision_cache_controller`, passing `runner.pipeline.components["text_encoder"]` if present, otherwise `runner.pipeline.text_encoder`. `ModuleNotFoundError` becomes `RuntimeError("condition-cache is not installed")` or `RuntimeError("vision-cache is not installed")`. Always set `runner.cache_namespace`.

`bind_generation`: if `runner.condition_cache` is `None`, keep `runner.pipeline` and `call_kwargs`, and use a no-op `observe`. Otherwise import `prepare` from `omni_infinity.caches.condition` and use its `ConditionReplay` (`pipeline or runner.pipeline`, `call_kwargs()`, `observe`). `denoise()` imports `denoise_step_cache` only when `denoise_cache is not None`; a missing module raises `RuntimeError("denoise-step cache is not installed")`.

`ReferenceRunner.from_pretrained` gains keyword-only `condition_cache: bool = False`, `condition_cache_dir: str | None = None`, `vision_cache: bool = False`. After the pipeline is built, `cache_namespace` is `f"ReferenceRunner:{checkpoint}"`. Call `attach_caches` before `return cls(...)`.

`ReferenceRunner.generate` gains `denoise_cache=None`. Media order is `(image, last_image, *references)`, using `None` for a missing image. Height and width come from `resolve_resolution(resolution)`. `total_steps` is `num_inference_steps`. Transformer is `_transformer_component(binding.pipeline, self.transformer_component)`. Call the pipeline with `binding.call_kwargs` inside `binding.denoise()` and the existing `_denoising_progress` block. Call `binding.observe(state)` after the pipeline returns.

`VdnRunner` gains the same `from_pretrained` kwargs. Namespace is `f"VdnRunner:{checkpoint}"`. `generate` gains `denoise_cache=None`. Media is `(image, last_image)`. Canvas is `height=768`, `width=1344`. `total_steps` is the evaluation count (`num_evaluations or self.default_evaluations`), not the sigma-grid length. Transformer is `binding.pipeline.transformer`.

Do not catch the `not installed` errors.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cache_contract.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/caches/attach.py omni_infinity/runner.py omni_infinity/arch/vdn.py tests/test_cache_contract.py
git commit -m "feat(caches): add runner hooks that fail closed"
```

---

### Task 4: Server and smoke flags

**Files:**
- Modify: `omni_infinity/serve/models.py:31-35`
- Modify: `omni_infinity/serve/app.py` (`ServerSettings`, `from_env`, `load_runner`)
- Modify: `examples/fl2va_smoke.py`
- Modify: `tests/test_server_settings.py`
- Test: `tests/test_cache_contract.py`

**Interfaces:**
- Consumes: `condition_cache`, `condition_cache_dir`, and `vision_cache` kwargs from Task 3.
- Produces: `ServerSettings.condition_cache_dir: str | None = None`. Env `OMNI_CONDITION_CACHE_DIR`. Empty env means `None`. `denoise_config_from_args(args)` in `omni_infinity/caches/attach.py`.

- [ ] **Step 1: Write the failing test**

```python
def test_request_accepts_cache_optimizations_and_rejects_denoise():
    GenerationRequest(
        type="fl2va",
        prompt="a",
        optimizations=["condition-cache", "vision-cache"],
    )
    with pytest.raises(ValidationError):
        GenerationRequest(
            type="fl2va", prompt="a", optimizations=["denoise-cache"]
        )


def test_condition_cache_dir_is_unset_by_default(clean_env):
    assert ServerSettings.from_env().condition_cache_dir is None


def test_load_runner_forwards_the_dir_only_with_the_flag(monkeypatch):
    _CaptureRunner.calls = []
    _stub_profile(monkeypatch, "h3-dense", condition_cache=True)
    load_runner(ServerSettings(condition_cache_dir="/tmp/cond"))
    assert _CaptureRunner.calls[-1][2]["condition_cache_dir"] == "/tmp/cond"

    _CaptureRunner.calls = []
    _stub_profile(monkeypatch, "h3-dense", offload=True)
    load_runner(ServerSettings(condition_cache_dir="/tmp/cond"))
    assert "condition_cache_dir" not in _CaptureRunner.calls[-1][2]
```

Add `"OMNI_CONDITION_CACHE_DIR"` to `_ENV_KEYS`. Add `assert settings.condition_cache_dir is None` to `test_settings_from_env_use_the_streamed_dense_defaults`.

`denoise_config_from_args(args)` lives in `omni_infinity/caches/attach.py`. It returns `None` when both denoise flags are absent. One flag without the other raises `ValueError` matching `both`. Both flags import `DenoiseCacheConfig` from `omni_infinity.caches.denoise` and return that config; a missing module raises `RuntimeError` matching `not installed`. Coefficients are a comma-separated list of floats, highest degree first.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache_contract.py tests/test_server_settings.py -q`
Expected: FAIL

- [ ] **Step 3: Wire settings and the smoke CLI**

Extend the `Literal` with `"condition-cache"` and `"vision-cache"` only. `from_env` reads `OMNI_CONDITION_CACHE_DIR` and stores `None` when unset or blank. `load_runner` copies `condition_cache_dir` into kwargs only when `kwargs.get("condition_cache")` is true.

Add `build_parser()` in `examples/fl2va_smoke.py` and keep `parse_args()` as `return build_parser().parse_args()`. Flags: `--condition-cache` (`store_true`), `--condition-cache-dir`, `--vision-cache` (`store_true`), `--denoise-cache-coefficients`, `--denoise-cache-threshold` (`float`). `main` passes `condition_cache`, `condition_cache_dir`, and `vision_cache` to `from_pretrained`, and `denoise_cache=denoise_config_from_args(args)` to `generate`. Do not change the default invocation. The test imports `build_parser` through `importlib.util.spec_from_file_location("fl2va_smoke", "examples/fl2va_smoke.py")` because `examples` is not a package.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cache_contract.py tests/test_server_settings.py tests/test_job_api.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/caches/attach.py omni_infinity/serve/models.py omni_infinity/serve/app.py examples/fl2va_smoke.py tests/test_server_settings.py tests/test_cache_contract.py
git commit -m "feat(caches): thread opt-in cache flags through serve and smoke"
```

---

### Task 5: Parity stays untouched

**Files:**
- Test: `tests/test_cache_contract.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing new.
- Produces: the README section "Opt-in caches".

- [ ] **Step 1: Write the failing test**

```python
def test_parity_suites_do_not_enable_caches():
    for name in (
        "test_reference_parity.py",
        "test_ref2va_parity.py",
        "test_vdn_parity.py",
    ):
        text = (Path("tests") / name).read_text()
        assert "condition_cache" not in text
        assert "vision_cache" not in text
        assert "denoise_cache" not in text
        assert "condition-cache" not in text
        assert "vision-cache" not in text
```

```python
def test_readme_points_at_the_cache_stack():
    text = Path("README.md").read_text()
    for sentence in (
        "Nothing is on by default.",
        "C2 is deferred.",
        "C5 refuses to run without H3-calibrated coefficients.",
        "C4 is unchanged upstream.",
    ):
        assert sentence in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache_contract.py::test_parity_suites_do_not_enable_caches tests/test_cache_contract.py::test_readme_points_at_the_cache_stack -q`
Expected: FAIL on the README assertion. The parity assertion should already pass; if it fails, stop and report which parity file mentions a cache.

- [ ] **Step 3: Add the README section after the optimization table**

One table, four rows: C1 `condition-cache` opt-in, C2 deferred, C3 `vision-cache` opt-in, C4 unchanged, C5 `--denoise-cache-coefficients` plus `--denoise-cache-threshold`, off the registry. State the four sentences from Step 1. Link this plan and issue #24. Do not describe hit behavior; the cache plans own that.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cache_contract.py tests/test_registry.py tests/test_server_settings.py -q`
Expected: PASS

Then: `python -m ruff check omni_infinity/caches omni_infinity/registry.py omni_infinity/runner.py omni_infinity/arch/vdn.py omni_infinity/serve/app.py omni_infinity/serve/models.py examples/fl2va_smoke.py tests/test_cache_contract.py tests/test_tensor_tree.py tests/test_server_settings.py` and `python -m ruff format --check` on those paths.
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_cache_contract.py
git commit -m "docs: point the README at the opt-in cache stack"
```
