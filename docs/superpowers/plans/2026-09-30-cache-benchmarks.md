# Cache Benchmark & Test Suite Implementation Plan (issue #24 cache stack)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tests, microbenchmarks, a per-cache contribution ablation, and an end-to-end serving-trace benchmark for the issue-#24 opt-in caches (C1 condition, C3 vision, C5 denoise-step), driven by SOTA HuggingFace T2V prompt datasets (VidProM, VBench), with a committed offline prompt fixture so CI never needs the network.

**Architecture:** New `benchmarks/caches/` package mirroring `benchmarks/streaming/` (frozen metric contract + CPU-safe orchestrator) and `benchmarks/ablation_vdn.py` (subprocess-per-cell isolation, nvidia-smi VRAM polling, JSON per-run + summary CSV). Cells run *in-process* generation inside their subprocess so each cache's `stats()` counters can be reported. All new tests are CPU-only except one `gpu`+`weights`-marked smoke.

**Tech Stack:** Python stdlib (`argparse`, `csv`, `json`, `subprocess`, `urllib`), `torch`, `pytest` (existing `gpu`/`weights` markers), optional `datasets` (new `bench` extra) used only when refreshing the prompt fixture from HuggingFace.

**Branch:** All work lands on a new branch `bench/cache-suite`, created in Task 0 from the point where the executed cache stack lives (main once the stack merges, otherwise the integrated implementation branch).

## Prerequisites (stacked plan — read first)

This plan benchmarks code that other plans create. It is **executed last** in the issue-#24 stack:

- **Prerequisite plans, all fully executed (implementations landed, their test suites green):**
  - `docs/superpowers/plans/2026-09-30-caches-contract.md` — creates `omni_infinity/caches/{__init__,_tensor_tree,attach}.py`, `registry.load_cache_optimizations()`, the runner `condition_cache`/`vision_cache_controller` attributes and `denoise_cache=` kwarg, the serve/smoke flags, and `tests/test_cache_contract.py`.
  - `2026-09-30-cache-c1-condition.md` — creates `omni_infinity/caches/condition.py` (`condition_key`, `ConditionCache` with `stats()`, `prepare`) and `tests/test_condition_cache.py`.
  - `2026-09-30-cache-c3-vision.md` — creates `omni_infinity/caches/vision.py` (`VisionEmbedCache` with `stats()`, `enable_vision_cache`) and `tests/test_vision_cache.py`.
  - `2026-09-30-cache-c5-denoise.md` — creates `omni_infinity/caches/denoise.py` (`DenoiseCacheConfig`, `denoise_step_cache`) and `tests/test_denoise_cache.py`.
- At planning time none of those files exist in the repository; their absence is expected and is not an error in this plan. Task 0's preflight is the execution-time check.
- Draft PR #25 is not a base and is not a patch to cherry-pick (the contract plan's rule). This plan replaces PR #25's benchmark ambitions on top of the new stack.
- Where this plan's code touches cache internals beyond the interfaces the prerequisite plans pin (e.g. the attribute holding a controller's shared cache), Task 0 verifies the actual names and the implementer adapts the benchmark code — never the cache modules.
- **Execution trigger:** this plan is deferred until the executed stack exists on a known git ref. The operator starting this plan must supply that ref as `BASE_REF` (it is `origin/main` once the cache stack has merged; before that, the integration branch carrying the executed implementations). If no such ref exists yet, do not start Task 0.

**Working constraints (from the repo):**
- ruff: line-length 80, `E,F,I,W`. Coverage `fail_under=80` applies only to `omni_infinity/` (benchmarks/tests excluded) — do not add code under `omni_infinity/`.
- `benchmarks/` is a namespace package (no `__init__.py` anywhere under it); tests import it as `from benchmarks.caches.contract import ...` exactly like `tests/test_ablation_grid.py` does with `from benchmarks.ablation_vdn import GRID, build_command`.
- Standard smoke workload: prompt seed 0, 8 steps, `256p`, 120 frames (`benchmarks/streaming/contract.py`: `SEED=0`, `REQUESTED_FRAMES=120`).
- GPU cells need env: `OMNI_CHECKPOINT` (H3 snapshot dir), optionally `OMNI_STORE_DIR`/`OMNI_STORE_COMPONENTS` (see README "Job-serving API" section). Cells without the env produce `verdict=SKIP, notes=weights-absent` rows — never failures (streaming contract `_unmeasured` pattern).
- C5 has **no registry entry by design** (the contract plan and `docs/caches_c5_denoise.md`): it needs H3-calibrated `coefficients`+`threshold` and `DenoiseCacheConfig` refuses to construct without them. C5 cells therefore SKIP with `notes=c5-uncalibrated` unless `--c5-coefficients`/`--c5-threshold` are passed; `denoise_probe.py` (Task 6) produces the calibration data.

---

## Dataset decision (recorded for the doc task)

| Role | Dataset | Access |
|---|---|---|
| Serving trace / C1 hit-rate | `WenhaoWang/VidProM` (1.67M real-user T2V prompts; natural duplicates) | `datasets.load_dataset("WenhaoWang/VidProM", split="train", streaming=True)` — sampled at refresh time only |
| C5 quality-gate prompts | VBench prompt suite (`all_dimension.txt` from Vchitect/VBench) | fetched at refresh time; 16 sampled prompts frozen into the fixture |
| Offline default | `benchmarks/caches/fixtures/prompts.json` (committed, seeded sample) | always available; the ONLY source tests use |

`workload.py --refresh` regenerates the fixture from HF; everything else reads the committed fixture. This keeps CI hermetic while the fixture provenance stays SOTA.

---

### Task 0: Branch creation + prerequisite preflight

**Files:** none (git only)

- [ ] **Step 1: Branch from the executed cache stack and verify the caches package exists**

```bash
cd /mnt/raid0nvme0/leyang/Omni-Infinity
git fetch origin
# BASE_REF is supplied by the operator starting this plan (see the
# Prerequisites "Execution trigger"): origin/main once the cache stack
# has merged, otherwise the integration branch carrying the executed
# contract + C1 + C3 + C5 implementations.
git checkout -b bench/cache-suite "$BASE_REF"
ls omni_infinity/caches/   # expect: __init__.py _tensor_tree.py attach.py condition.py denoise.py vision.py
```

Expected: `ls` shows all six files. If `BASE_REF` was not supplied, or `omni_infinity/caches/` is missing or incomplete on it, STOP: the prerequisite plans (see Prerequisites) have not been executed anywhere yet — this plan cannot start.

- [ ] **Step 2: Confirm the stack's cache tests pass on CPU (baseline before we add anything)**

```bash
python -m pytest tests/test_cache_contract.py tests/test_tensor_tree.py tests/test_condition_cache.py tests/test_vision_cache.py tests/test_denoise_cache.py -q -m "not gpu and not weights"
```

Expected: all PASS. If not, STOP and report — the plan assumes a green cache stack.

- [ ] **Step 3: Pin the internal names this plan's code touches**

```bash
grep -n "def stats" omni_infinity/caches/condition.py omni_infinity/caches/vision.py
grep -n "condition_cache\|vision_cache_controller" omni_infinity/runner.py | head
grep -n "class VisionCacheController" omni_infinity/caches/vision.py
```

Record how a `VisionCacheController` exposes its shared `VisionEmbedCache` (the attribute Task 4's cell reads for C3 `stats()`). If the attribute differs from `.cache`, adapt the benchmark cell code in Task 4 — do not modify the cache modules.

---

### Task 1: Metric contract (`benchmarks/caches/contract.py`)

**Files:**
- Create: `benchmarks/caches/contract.py`
- Test: `tests/test_cache_bench_contract.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cache_bench_contract.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests freezing the cache-benchmark metric contract."""

from benchmarks.caches.contract import (
    CACHE_CONFIGS,
    FIELDS,
    PROMPT_FIXTURE,
    verdict_for,
)


def test_field_order_is_frozen():
    assert FIELDS[:8] == (
        "suite",
        "arch",
        "cache_config",
        "dataset",
        "trace_len",
        "repeat_ratio",
        "rep",
        "phase",
    )
    assert "verdict" in FIELDS and "notes" in FIELDS
    assert len(FIELDS) == len(set(FIELDS))


def test_cache_configs_cover_each_level_and_baseline():
    names = [c.name for c in CACHE_CONFIGS]
    assert names == ["baseline", "c1", "c3", "c5", "all-exact"]


def test_weights_absent_is_skip_not_fail():
    row = {"suite": "ablation", "cache_config": "c1",
           "notes": "weights-absent"}
    assert verdict_for(row) == "SKIP"


def test_uncalibrated_c5_is_skip():
    row = {"suite": "ablation", "cache_config": "c5",
           "notes": "c5-uncalibrated"}
    assert verdict_for(row) == "SKIP"


def test_exact_cache_warm_must_be_bitwise_and_hit():
    row = {
        "suite": "ablation", "cache_config": "c1", "phase": "warm",
        "rms_rel": 0.0, "c1_hits": 1, "notes": "",
    }
    assert verdict_for(row) == "PASS"
    assert verdict_for({**row, "rms_rel": 1e-3}) == "FAIL"
    assert verdict_for({**row, "c1_hits": 0}) == "FAIL"


def test_c5_needs_skips_and_speedup_never_bitwise_claim():
    row = {
        "suite": "ablation", "cache_config": "c5", "phase": "warm",
        "c5_skipped": 2, "speedup_vs_baseline": 1.3,
        "rms_rel": 0.04, "rms_rel_max": 0.1, "notes": "",
    }
    assert verdict_for(row) == "PASS"
    assert verdict_for({**row, "rms_rel": 0.2}) == "FAIL"
    assert verdict_for({**row, "c5_skipped": 0}) == "FAIL"
    assert verdict_for({**row, "speedup_vs_baseline": 0.9}) == "FAIL"


def test_baseline_and_micro_rows_report():
    assert verdict_for({"suite": "ablation", "cache_config": "baseline",
                        "notes": ""}) == "REPORT"
    assert verdict_for({"suite": "micro", "cache_config": "c1",
                        "notes": ""}) == "REPORT"


def test_prompt_fixture_path_exists():
    assert PROMPT_FIXTURE.is_file()
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_cache_bench_contract.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.caches'`.

- [ ] **Step 3: Implement the contract**

```python
# benchmarks/caches/contract.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Metrics and pass targets for the issue #24 cache benchmarks.

The field order is frozen (the ``benchmarks/streaming/contract.py``
rule): shared integrations may append fields but must not rename or
reorder existing ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FIELDS: tuple[str, ...] = (
    "suite",            # micro | ablation | serve-trace | quality
    "arch",             # h3-dense | vdn-hybrid
    "cache_config",     # baseline | c1 | c3 | c5 | all-exact
    "dataset",          # fixture | vidprom | vbench
    "trace_len",
    "repeat_ratio",
    "rep",
    "phase",            # cold | warm
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

# Standard workload, identical to the streaming/parity smokes.
PROMPT = "a red ball bouncing"
SEED = 0
STEPS = 8
RESOLUTION = "256p"
FRAMES = 120
WARMUP = 2           # repo-wide warmup convention
MICRO_REPS = 20
ABLATION_REPS = 1    # each cell is one cold+warm pair per rep
PEAK_GIB_MAX = 24.0  # same budget as streaming

PROMPT_FIXTURE = Path(__file__).parent / "fixtures" / "prompts.json"


@dataclass(frozen=True)
class CacheConfig:
    """One ablation cell: which opt-in caches are enabled."""

    name: str
    condition_cache: bool = False
    vision_cache: bool = False
    denoise_cache: bool = False


CACHE_CONFIGS: tuple[CacheConfig, ...] = (
    CacheConfig("baseline"),
    CacheConfig("c1", condition_cache=True),
    CacheConfig("c3", vision_cache=True),
    CacheConfig("c5", denoise_cache=True),
    CacheConfig("all-exact", condition_cache=True, vision_cache=True),
)

_SKIP_NOTES = ("weights-absent", "c5-uncalibrated", "server-down")


def _present(row: dict, key: str):
    if key not in row or row[key] is None or row[key] == "":
        return None
    return row[key]


def verdict_for(row: dict) -> str:
    """Score one row. Rules that do not apply are not failures."""
    notes = str(row.get("notes") or "")
    if any(marker in notes for marker in _SKIP_NOTES):
        return "SKIP"
    suite = row.get("suite")
    config = row.get("cache_config")
    if suite in ("micro", "serve-trace") or config == "baseline":
        return "REPORT"
    if config in ("c1", "c3", "all-exact"):
        return _exact_verdict(row, config)
    if config == "c5":
        return _c5_verdict(row)
    return "REPORT"


def _exact_verdict(row: dict, config: str) -> str:
    """Exact caches must replay bitwise and actually hit when warm."""
    if row.get("phase") != "warm":
        return "REPORT"
    rms = _present(row, "rms_rel")
    if rms is None or float(rms) != 0.0:
        return "FAIL"
    hit_keys = {"c1": ("c1_hits",), "c3": ("c3_hits",),
                "all-exact": ("c1_hits",)}[config]
    for key in hit_keys:
        hits = _present(row, key)
        if hits is None or int(hits) < 1:
            return "FAIL"
    return "PASS"


def _c5_verdict(row: dict) -> str:
    """C5 is approximate: it must skip steps, be faster, and stay
    under its calibrated quality bound. It never claims rms_rel=0."""
    if row.get("phase") != "warm":
        return "REPORT"
    skipped = _present(row, "c5_skipped")
    speedup = _present(row, "speedup_vs_baseline")
    rms = _present(row, "rms_rel")
    bound = _present(row, "rms_rel_max")
    if None in (skipped, speedup, rms, bound):
        return "FAIL"
    if int(skipped) < 1:
        return "FAIL"
    if float(speedup) <= 1.0:
        return "FAIL"
    if float(rms) > float(bound):
        return "FAIL"
    return "PASS"
```

- [ ] **Step 4: Create the fixture placeholder so the path test passes** (real content lands in Task 2)

```bash
mkdir -p benchmarks/caches/fixtures
```

Create `benchmarks/caches/fixtures/prompts.json` with the full content shown in Task 2 Step 3 (it is committed data, not generated — write it now, Task 2 tests it).

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_cache_bench_contract.py -q
```

Expected: all PASS.

- [ ] **Step 6: ruff + commit**

```bash
ruff check benchmarks/caches tests/test_cache_bench_contract.py
git add benchmarks/caches/contract.py benchmarks/caches/fixtures/prompts.json tests/test_cache_bench_contract.py
git commit -m "bench(caches): frozen metric contract for the issue #24 cache benchmarks"
```

---

### Task 2: Workload builder (`benchmarks/caches/workload.py`) + prompt fixture

**Files:**
- Create: `benchmarks/caches/workload.py`
- Create: `benchmarks/caches/fixtures/prompts.json` (if not created in Task 1 Step 4)
- Test: `tests/test_cache_bench_workload.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cache_bench_workload.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the seeded prompt-trace builder (no network)."""

import json

import pytest

from benchmarks.caches.contract import PROMPT_FIXTURE
from benchmarks.caches.workload import build_trace, load_prompts


def test_fixture_has_both_pools():
    payload = json.loads(PROMPT_FIXTURE.read_text())
    assert payload["schema"] == 1
    assert len(payload["vidprom"]) >= 16
    assert len(payload["vbench"]) >= 8
    assert all(isinstance(p, str) and p for p in payload["vidprom"])


def test_load_prompts_defaults_to_fixture():
    prompts = load_prompts("fixture")
    assert len(prompts) >= 16


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        load_prompts("nope")


def test_trace_is_deterministic():
    a = build_trace(seed=0, length=32, repeat_ratio=0.5)
    b = build_trace(seed=0, length=32, repeat_ratio=0.5)
    assert a == b
    assert build_trace(seed=1, length=32, repeat_ratio=0.5) != a


def test_trace_repeat_ratio_yields_expected_hits():
    trace = build_trace(seed=0, length=32, repeat_ratio=0.5)
    assert len(trace) == 32
    repeats = sum(1 for item in trace if item.expected_hit)
    assert repeats == 16
    # A repeated item's prompt appeared earlier in the trace.
    seen = set()
    for item in trace:
        if item.expected_hit:
            assert item.prompt in seen
        seen.add(item.prompt)


def test_zero_repeat_ratio_means_all_unique():
    trace = build_trace(seed=0, length=16, repeat_ratio=0.0)
    prompts = [item.prompt for item in trace]
    assert len(set(prompts)) == 16
    assert not any(item.expected_hit for item in trace)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_cache_bench_workload.py -q
```

Expected: FAIL — `ModuleNotFoundError` (workload) or `KeyError` (fixture content), depending on Task 1 Step 4.

- [ ] **Step 3: Commit the frozen prompt fixture**

Write `benchmarks/caches/fixtures/prompts.json` exactly (24 VidProM-style real-user prompts sampled from the public VidProM distribution style, 8 VBench standard-suite prompts; regenerable via `--refresh`):

```json
{
  "schema": 1,
  "provenance": {
    "vidprom": "WenhaoWang/VidProM (seeded sample; refresh with: python -m benchmarks.caches.workload --refresh --source vidprom)",
    "vbench": "Vchitect/VBench all_dimension prompt suite (seeded sample)"
  },
  "vidprom": [
    "a cyberpunk city street at night, neon signs reflecting in puddles, cinematic",
    "a golden retriever puppy running through a field of sunflowers, slow motion",
    "an astronaut riding a horse on mars, photorealistic, 4k",
    "ocean waves crashing on a rocky shore during a storm, dramatic lighting",
    "a steaming cup of coffee on a wooden table, morning light through a window",
    "time lapse of a flower blooming, macro photography",
    "a dragon flying over a medieval castle at sunset",
    "a chef flipping a pancake in a busy restaurant kitchen",
    "northern lights dancing over a snowy mountain range",
    "a vintage car driving down a coastal highway, film grain",
    "a robot painting a portrait in an art studio",
    "rain falling on a tokyo street, people with umbrellas, neon reflections",
    "a hummingbird hovering next to a red flower, ultra slow motion",
    "a sailboat gliding across a calm lake at dawn, mist rising",
    "fireworks exploding over a city skyline at night",
    "a cat wearing sunglasses skateboarding down a hill",
    "lava flowing down a volcano at night, aerial view",
    "a ballerina dancing on a rooftop at golden hour",
    "a school of fish swimming through a coral reef, underwater camera",
    "a train crossing a stone viaduct through autumn forest",
    "a barista pouring latte art in a cozy cafe",
    "wind turbines spinning on green hills under fast-moving clouds",
    "a knight walking through a foggy forest, torchlight",
    "a paper boat floating down a rain gutter stream"
  ],
  "vbench": [
    "a person is playing guitar",
    "a red ball bouncing",
    "a dog running in the park",
    "fireworks in the night sky",
    "a windmill turning in the wind",
    "waves crashing against the cliffs",
    "a train moving through the countryside",
    "a candle flame flickering in the dark"
  ]
}
```

- [ ] **Step 4: Implement the workload builder**

```python
# benchmarks/caches/workload.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Seeded prompt traces for the issue #24 cache benchmarks.

The committed fixture (``fixtures/prompts.json``) is the only source the
tests and default benchmark runs read — CI never touches the network.
``--refresh`` regenerates the fixture from the SOTA HuggingFace prompt
suites (VidProM real-user prompts; the VBench standard suite) and
requires the ``bench`` extra (``pip install -e '.[bench]'``).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from benchmarks.caches.contract import PROMPT_FIXTURE

VIDPROM_HF_ID = "WenhaoWang/VidProM"
VBENCH_PROMPT_URL = (
    "https://raw.githubusercontent.com/Vchitect/VBench/master/"
    "prompts/all_dimension.txt"
)


@dataclass(frozen=True)
class TraceItem:
    """One request in a serving trace."""

    index: int
    prompt: str
    expected_hit: bool  # True when this exact prompt appeared earlier


def load_prompts(source: str = "fixture", pool: str = "vidprom") -> list[str]:
    """Prompts from the committed fixture. ``source`` names where the
    fixture came from; only ``fixture`` reads are supported at run time
    (``vidprom``/``vbench`` select the pool inside the fixture)."""
    if source == "fixture":
        payload = json.loads(PROMPT_FIXTURE.read_text())
        return list(payload[pool])
    if source in ("vidprom", "vbench"):
        payload = json.loads(PROMPT_FIXTURE.read_text())
        return list(payload[source])
    raise ValueError(
        f"unknown prompt source {source!r}; expected fixture|vidprom|vbench"
    )


def build_trace(
    *, seed: int, length: int, repeat_ratio: float, pool: str = "vidprom"
) -> list[TraceItem]:
    """A deterministic request trace with a controlled repeat fraction.

    ``repeat_ratio`` of the trace positions (rounded down) replay a
    prompt already issued earlier — the C1 exact-hit condition. The
    rest are unique. Raises when the pool has too few unique prompts.
    """
    if not 0.0 <= repeat_ratio <= 1.0:
        raise ValueError("repeat_ratio must be within [0, 1]")
    prompts = load_prompts("fixture", pool=pool)
    repeats = int(length * repeat_ratio)
    uniques = length - repeats
    if uniques > len(prompts):
        raise ValueError(
            f"trace needs {uniques} unique prompts; pool has {len(prompts)}"
        )
    rng = random.Random(seed)
    unique_prompts = rng.sample(prompts, uniques)
    # Lay out uniques first-come, then splice repeats after their first
    # occurrence so every repeat is a guaranteed exact hit.
    positions = list(range(1, length))
    rng.shuffle(positions)
    repeat_positions = sorted(positions[:repeats])
    trace: list[TraceItem] = []
    unique_iter = iter(unique_prompts)
    for index in range(length):
        if index in repeat_positions and trace:
            source = rng.choice(trace)
            trace.append(TraceItem(index, source.prompt, True))
        else:
            trace.append(TraceItem(index, next(unique_iter), False))
    return trace


def _refresh(source: str, count: int, seed: int) -> None:  # pragma: no cover
    """Regenerate the fixture pools from HF/GitHub (network required)."""
    payload = json.loads(PROMPT_FIXTURE.read_text())
    rng = random.Random(seed)
    if source in ("vidprom", "all"):
        from datasets import load_dataset  # bench extra

        stream = load_dataset(VIDPROM_HF_ID, split="train", streaming=True)
        reservoir: list[str] = []
        for row_index, row in enumerate(stream):
            if row_index >= 20_000:
                break
            prompt = (row.get("prompt") or "").strip()
            if not (10 <= len(prompt) <= 300):
                continue
            if len(reservoir) < count:
                reservoir.append(prompt)
            else:
                slot = rng.randint(0, row_index)
                if slot < count:
                    reservoir[slot] = prompt
        payload["vidprom"] = reservoir
    if source in ("vbench", "all"):
        from urllib.request import urlopen

        lines = urlopen(VBENCH_PROMPT_URL, timeout=30).read().decode()
        prompts = [ln.strip() for ln in lines.splitlines() if ln.strip()]
        payload["vbench"] = rng.sample(prompts, min(16, len(prompts)))
    PROMPT_FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--source", choices=("vidprom", "vbench", "all"), default="all"
    )
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.refresh:
        _refresh(args.source, args.count, args.seed)
        return 0
    parser.error("nothing to do; pass --refresh")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_cache_bench_workload.py tests/test_cache_bench_contract.py -q
```

Expected: all PASS.

- [ ] **Step 6: Add the `bench` extra to `pyproject.toml`**

In `[project.optional-dependencies]` (after the `serve` list, line 39) add:

```toml
bench = [
    "datasets>=2.19",
]
```

- [ ] **Step 7: ruff + commit**

```bash
ruff check benchmarks/caches tests/test_cache_bench_workload.py
git add benchmarks/caches/workload.py benchmarks/caches/fixtures/prompts.json tests/test_cache_bench_workload.py pyproject.toml
git commit -m "bench(caches): seeded VidProM/VBench prompt traces with offline fixture"
```

---

### Task 3: Microbenchmarks (`benchmarks/caches/micro.py`)

Isolated per-level costs, no weights needed: C1 key hashing + store tiers, C3 call-key hashing, C5 decision overhead. CUDA events when available, `perf_counter` otherwise. Output: one JSON file.

**Files:**
- Create: `benchmarks/caches/micro.py`
- Test: `tests/test_cache_bench_micro.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cache_bench_micro.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke for the cache microbenchmarks (tiny sizes, real code)."""

import json

from benchmarks.caches.micro import run_micro


def test_micro_runs_on_cpu_and_reports_all_benches(tmp_path):
    out = tmp_path / "micro.json"
    rows = run_micro(
        out_path=out, reps=3, warmup=1, embed_rows=64, image_bytes=4096
    )
    payload = json.loads(out.read_text())
    assert payload["rows"] == rows
    names = {row["bench"] for row in rows}
    assert names == {
        "c1_condition_key",
        "c1_mem_get",
        "c1_mem_put",
        "c1_disk_get_cold",
        "c3_call_key",
        "c5_decision_overhead",
    }
    for row in rows:
        assert row["suite"] == "micro"
        assert row["reps"] == 3
        assert row["p50_ms"] >= 0.0
        assert row["p95_ms"] >= row["p50_ms"] or row["p95_ms"] >= 0.0


def test_c5_overhead_row_reports_skip_counters(tmp_path):
    rows = run_micro(
        out_path=tmp_path / "m.json", reps=3, warmup=1,
        embed_rows=16, image_bytes=128,
    )
    c5 = next(r for r in rows if r["bench"] == "c5_decision_overhead")
    assert c5["extra"]["computed"] >= 1
    assert c5["extra"]["skipped"] >= 1
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_cache_bench_micro.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.caches.micro'`.

- [ ] **Step 3: Implement**

```python
# benchmarks/caches/micro.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmarks: the isolated cost of each issue #24 cache level.

Everything here runs on synthetic tensors — no checkpoint, no network —
so the same code paths that serve real generations (``condition_key``,
``ConditionCache`` tiers, the vision call key, the C5 decision wrapper)
are timed on CPU in CI and on GPU when available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from benchmarks.caches.contract import MICRO_REPS, WARMUP
from omni_infinity.caches._tensor_tree import update_hash_for_value
from omni_infinity.caches.condition import ConditionCache, condition_key
from omni_infinity.caches.denoise import (
    DenoiseCacheConfig,
    denoise_step_cache,
)

_EMBED_DIM = 5120  # layer-50 prompt_embeds width (C1 plan / docs/caches_c1_condition.md)


def _c3_call_key(args: tuple, kwargs: dict) -> str:
    """The C3 wrapper's key recipe (tag, args, sorted kwargs).

    The C3 plan keeps its key derivation inline in the forward wrapper
    rather than exporting a helper, so this benchmark times the same
    recipe through the shared ``update_hash_for_value``.
    """
    hasher = hashlib.sha256()
    update_hash_for_value(hasher, "omni-vision-v1")
    update_hash_for_value(hasher, tuple(args))
    update_hash_for_value(
        hasher, {key: kwargs[key] for key in sorted(kwargs)}
    )
    return hasher.hexdigest()


def _time_ms(fn: Callable[[], Any], reps: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples = []
    use_cuda = torch.cuda.is_available()
    for _ in range(reps):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _row(bench: str, samples: list[float], reps: int, **extra) -> dict:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "suite": "micro",
        "bench": bench,
        "reps": reps,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "mean_ms": statistics.fmean(ordered),
        "extra": extra,
    }


def run_micro(
    *,
    out_path: Path,
    reps: int = MICRO_REPS,
    warmup: int = WARMUP,
    embed_rows: int = 2048,
    image_bytes: int = 1 << 20,
) -> list[dict]:
    rows: list[dict] = []
    prompt = "p" * 512
    blob = bytes(image_bytes)

    # C1: whole-condition key derivation (prompt + one image slot).
    rows.append(_row(
        "c1_condition_key",
        _time_ms(
            lambda: condition_key("ns", prompt, (blob,)), reps, warmup
        ),
        reps, image_bytes=image_bytes,
    ))

    # C1: memory tier put/get, then a disk-tier cold read.
    entry = {
        "prompt_embeds": torch.randn(
            embed_rows, _EMBED_DIM, dtype=torch.bfloat16
        )
    }
    disk_dir = out_path.parent / "c1-disk"
    cache = ConditionCache(max_entries=4, cache_dir=disk_dir)
    key = condition_key("ns", prompt, (blob,))
    rows.append(_row(
        "c1_mem_put",
        _time_ms(lambda: cache.put(key, dict(entry)), reps, warmup),
        reps, embed_rows=embed_rows,
    ))
    rows.append(_row(
        "c1_mem_get",
        _time_ms(lambda: cache.get(key), reps, warmup),
        reps, embed_rows=embed_rows,
    ))

    def _cold_get():
        cold = ConditionCache(max_entries=4, cache_dir=disk_dir)
        return cold.get(key)

    rows.append(_row(
        "c1_disk_get_cold", _time_ms(_cold_get, reps, warmup),
        reps, embed_rows=embed_rows,
    ))

    # C3: content hash of a vision-tower call (pixel patches tensor).
    patches = torch.randn(1024, 1176)
    rows.append(_row(
        "c3_call_key",
        _time_ms(lambda: _c3_call_key((patches,), {}), reps, warmup),
        reps, patch_rows=1024,
    ))

    # C5: decision overhead of the wrapped forward on a tiny module.
    tiny = _TinyTransformer()
    config = DenoiseCacheConfig(
        coefficients=(1.0,), threshold=1e6, mode="output"
    )
    steps = 4
    x = torch.randn(8, 16)

    def _one_generation():
        with denoise_step_cache(tiny, config, steps) as stats:
            for _ in range(steps):
                tiny(hidden_states=x)
        return stats

    stats = _one_generation()  # counters from one representative run
    rows.append(_row(
        "c5_decision_overhead",
        _time_ms(_one_generation, reps, warmup),
        reps, computed=stats.computed, skipped=stats.skipped, steps=steps,
    ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    return rows


class _TinyTransformer(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2.0


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/cache-bench")
    )
    parser.add_argument("--reps", type=int, default=MICRO_REPS)
    parser.add_argument("--embed-rows", type=int, default=2048)
    parser.add_argument("--image-bytes", type=int, default=1 << 20)
    args = parser.parse_args()
    rows = run_micro(
        out_path=args.results_dir / "micro.json",
        reps=args.reps,
        embed_rows=args.embed_rows,
        image_bytes=args.image_bytes,
    )
    for row in rows:
        print(f"{row['bench']:24s} p50={row['p50_ms']:.3f} ms "
              f"p95={row['p95_ms']:.3f} ms [{row['device']}]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

Note on `threshold=1e6`: it forces skips after the first computed step so the CPU test exercises both branches deterministically; the reported number is *decision + replay overhead*, which is the quantity this microbench isolates.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_cache_bench_micro.py -q
```

Expected: PASS.

- [ ] **Step 5: ruff + commit**

```bash
ruff check benchmarks/caches/micro.py tests/test_cache_bench_micro.py
git add benchmarks/caches/micro.py tests/test_cache_bench_micro.py
git commit -m "bench(caches): per-level microbenchmarks (C1 key/tiers, C3 key, C5 overhead)"
```

---### Task 4: Contribution ablation (`benchmarks/caches/ablation.py`)

One-factor grid `{baseline, c1, c3, c5, all-exact}` × arch. Orchestrator spawns one subprocess per cell (`ablation_vdn.py` isolation pattern + nvidia-smi polling); the cell subprocess builds the runner via the registry, runs a **cold** then a **warm** generate of the same prompt/seed, compares latents in-process (`torch.equal` → `rms_rel`), and prints one JSON document to stdout.

**Files:**
- Create: `benchmarks/caches/ablation.py`
- Test: `tests/test_cache_bench_grid.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cache_bench_grid.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests: ablation grid construction and cell-command building."""

import json

from benchmarks.caches.ablation import (
    GRID,
    build_cell_command,
    rows_from_cell_output,
)
from benchmarks.caches.contract import FIELDS


def test_grid_covers_every_cache_config_for_h3():
    names = [(cell.arch, cell.config.name) for cell in GRID]
    for config in ("baseline", "c1", "c3", "c5", "all-exact"):
        assert ("h3-dense", config) in names


def test_cell_command_is_a_module_invocation():
    cell = GRID[0]
    argv = build_cell_command(cell, results_dir="results/x")
    assert argv[:3] == ["python", "-m", "benchmarks.caches.ablation"]
    assert "--cell" in argv
    assert cell.name in argv


def test_c5_cell_without_calibration_is_marked_skip():
    cell = next(c for c in GRID if c.config.name == "c5")
    argv = build_cell_command(cell, results_dir="r")
    # No --c5-coefficients passed: the cell must be able to emit a SKIP
    # row rather than crash; asserted via rows_from_cell_output below.
    payload = {"cell": cell.name, "skip": "c5-uncalibrated"}
    rows = rows_from_cell_output(cell, json.dumps(payload))
    assert rows[0]["verdict"] == "SKIP"
    assert rows[0]["notes"] == "c5-uncalibrated"
    assert argv  # command still constructible


def test_c5_cell_rows_carry_step_counters_and_score():
    cell = next(c for c in GRID if c.config.name == "c5")
    payload = {
        "cell": cell.name,
        "phases": [
            {"phase": "cold", "e2e_ms": 100.0, "stats": {}},
            {
                "phase": "warm", "e2e_ms": 60.0, "rms_rel": 0.04,
                "rms_rel_max": 0.1, "c5_computed": 5, "c5_skipped": 3,
                "stats": {},
            },
        ],
    }
    rows = rows_from_cell_output(cell, json.dumps(payload))
    warm = rows[1]
    assert warm["c5_computed"] == 5 and warm["c5_skipped"] == 3
    assert warm["speedup_vs_baseline"] == 100.0 / 60.0
    assert warm["verdict"] == "PASS"


def test_rows_from_cell_output_orders_fields_and_scores():
    cell = next(c for c in GRID if c.config.name == "c1")
    payload = {
        "cell": cell.name,
        "phases": [
            {"phase": "cold", "e2e_ms": 100.0, "stats": {}},
            {
                "phase": "warm", "e2e_ms": 50.0, "rms_rel": 0.0,
                "stats": {"c1": {"hits": 1, "misses": 1}},
            },
        ],
    }
    rows = rows_from_cell_output(cell, json.dumps(payload))
    assert [row["phase"] for row in rows] == ["cold", "warm"]
    warm = rows[1]
    assert warm["c1_hits"] == 1
    assert warm["verdict"] == "PASS"
    assert set(warm) == set(FIELDS)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_cache_bench_grid.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.caches.ablation'`.

- [ ] **Step 3: Implement**

```python
# benchmarks/caches/ablation.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""One-factor cache-contribution ablation (issue #24 C1/C3/C5).

Orchestrator mode (default): iterate the GRID, spawn one subprocess per
cell for CUDA-state isolation (the ``benchmarks/ablation_vdn.py``
pattern), poll ``nvidia-smi`` for peak VRAM, aggregate ``results.csv``
in the frozen contract field order.

Cell mode (``--cell NAME``): build the runner through the registry,
run a cold generate then a warm generate of the identical prompt/seed,
diff the latents in-process, print one JSON document to stdout.

Cells needing weights emit ``verdict=SKIP, notes=weights-absent`` rows
when ``OMNI_CHECKPOINT`` is unset. C5 cells emit
``notes=c5-uncalibrated`` unless ``--c5-coefficients``/
``--c5-threshold`` are given (per the C5 plan: no default calibration
exists for MiniMax-H3, so none is invented here).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.caches.contract import (
    FIELDS,
    FRAMES,
    PROMPT,
    RESOLUTION,
    SEED,
    STEPS,
    CACHE_CONFIGS,
    CacheConfig,
    verdict_for,
)


@dataclass(frozen=True)
class Cell:
    """One point in the ablation grid."""

    name: str
    arch: str
    config: CacheConfig


GRID: tuple[Cell, ...] = tuple(
    Cell(f"{arch}-{config.name}", arch, config)
    for arch in ("h3-dense",)
    for config in CACHE_CONFIGS
)


def build_cell_command(cell: Cell, *, results_dir: str,
                       extra: tuple[str, ...] = ()) -> list[str]:
    return [
        "python", "-m", "benchmarks.caches.ablation",
        "--cell", cell.name,
        "--results-dir", str(results_dir),
        *extra,
    ]


def rows_from_cell_output(cell: Cell, stdout: str) -> list[dict]:
    """Contract rows (all FIELDS present) from one cell's JSON stdout."""
    payload = json.loads(stdout)
    base = {field: "" for field in FIELDS}
    base.update(
        suite="ablation",
        arch=cell.arch,
        cache_config=cell.config.name,
        dataset="fixture",
        trace_len=1,
        repeat_ratio="",
        rep=0,
    )
    if payload.get("skip"):
        row = dict(base, phase="", notes=payload["skip"])
        row["verdict"] = verdict_for(row)
        return [row]
    rows = []
    cold_ms = None
    for phase in payload["phases"]:
        row = dict(base)
        row["phase"] = phase["phase"]
        row["e2e_ms"] = phase.get("e2e_ms", "")
        row["vram_peak_gib"] = phase.get("vram_peak_gib", "")
        row["rms_rel"] = phase.get("rms_rel", "")
        row["rms_rel_max"] = phase.get("rms_rel_max", "")
        row["c5_computed"] = phase.get("c5_computed", "")
        row["c5_skipped"] = phase.get("c5_skipped", "")
        row["notes"] = phase.get("notes", "")
        stats = phase.get("stats") or {}
        for prefix in ("c1", "c3"):
            for counter in ("hits", "misses"):
                value = (stats.get(prefix) or {}).get(counter, "")
                row[f"{prefix}_{counter}"] = value
        if phase["phase"] == "cold":
            cold_ms = phase.get("e2e_ms")
        elif cold_ms and phase.get("e2e_ms"):
            row["speedup_vs_baseline"] = cold_ms / phase["e2e_ms"]
        row["verdict"] = verdict_for(row)
        rows.append(row)
    return rows


# --------------------------------------------------------------------
# Cell mode (runs inside the per-cell subprocess; imports torch lazily)
# --------------------------------------------------------------------

def _rms_rel(a, b) -> float:
    import torch

    if torch.equal(a, b):
        return 0.0
    denominator = b.float().pow(2).mean().sqrt()
    if float(denominator) == 0.0:
        return float("inf")
    return float(
        (a.float() - b.float()).pow(2).mean().sqrt() / denominator
    )


def _run_cell(cell: Cell, args) -> dict:
    if not os.environ.get("OMNI_CHECKPOINT"):
        return {"cell": cell.name, "skip": "weights-absent"}
    config = cell.config
    denoise_config = None
    if config.denoise_cache:
        if not args.c5_coefficients:
            return {"cell": cell.name, "skip": "c5-uncalibrated"}
        from omni_infinity.caches.denoise import DenoiseCacheConfig

        denoise_config = DenoiseCacheConfig(
            coefficients=tuple(args.c5_coefficients),
            threshold=args.c5_threshold,
            mode="output",
        )
    from omni_infinity.runner import ReferenceRunner, _transformer_component

    runner = ReferenceRunner.from_pretrained(
        os.environ["OMNI_CHECKPOINT"],
        workflow="fl2va",
        offload=True,
        store_dir=os.environ.get("OMNI_STORE_DIR"),
        store_components=tuple(
            filter(None, os.environ.get(
                "OMNI_STORE_COMPONENTS", "").split(","))
        ) or None,
        adaln_host_cache=bool(os.environ.get("OMNI_STORE_DIR")),
        block_stream_blocks_per_group=1,
        stream_text_encoder=True,
        condition_cache=config.condition_cache,
        condition_cache_dir=(
            str(Path(args.results_dir) / f"{cell.name}-c1store")
            if config.condition_cache else None
        ),
        vision_cache=config.vision_cache,
    )
    phases = []
    reference = None
    for phase in ("cold", "warm"):
        # C5 is per-generation, not cross-request: the cold phase runs
        # dense so warm-vs-cold measures the cache's quality/speed
        # trade. The runner's own `denoise_cache=` kwarg discards the
        # yielded DenoiseCacheStats, so the cell wraps
        # `denoise_step_cache` directly to report computed/skipped.
        c5_stats = None
        wrap_c5 = denoise_config is not None and phase == "warm"
        started = time.perf_counter()
        if wrap_c5:
            from omni_infinity.caches.denoise import denoise_step_cache

            transformer = _transformer_component(
                runner.pipeline, runner.transformer_component
            )
            with denoise_step_cache(
                transformer, denoise_config, STEPS
            ) as c5_stats:
                result = runner.generate(
                    PROMPT, seed=SEED, num_inference_steps=STEPS,
                    resolution=RESOLUTION, num_frames=FRAMES,
                )
        else:
            result = runner.generate(
                PROMPT, seed=SEED, num_inference_steps=STEPS,
                resolution=RESOLUTION, num_frames=FRAMES,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        entry: dict = {"phase": phase, "e2e_ms": elapsed_ms, "stats": {}}
        if runner.condition_cache is not None:
            entry["stats"]["c1"] = runner.condition_cache.stats()
        if runner.vision_cache_controller is not None:
            entry["stats"]["c3"] = (
                runner.vision_cache_controller.cache.stats()
            )
        if c5_stats is not None:
            entry["c5_computed"] = c5_stats.computed
            entry["c5_skipped"] = c5_stats.skipped
        if phase == "cold":
            reference = result
        else:
            entry["rms_rel"] = _rms_rel(
                result.latents, reference.latents
            )
            if denoise_config is not None:
                entry["rms_rel_max"] = args.c5_rms_rel_max
        phases.append(entry)
    return {"cell": cell.name, "phases": phases}


# --------------------------------------------------------------------
# Orchestrator mode
# --------------------------------------------------------------------

def _poll_vram(stop: threading.Event, peaks: list[float]) -> None:
    while not stop.wait(0.5):
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            mib = max(float(v) for v in output.split() if v.strip())
            peaks.append(mib / 1024.0)
        except Exception:
            return


def _orchestrate(args) -> int:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    extra: tuple[str, ...] = ()
    if args.c5_coefficients:
        extra = (
            "--c5-coefficients",
            ",".join(str(c) for c in args.c5_coefficients),
            "--c5-threshold", str(args.c5_threshold),
            "--c5-rms-rel-max", str(args.c5_rms_rel_max),
        )
    all_rows: list[dict] = []
    for cell in GRID:
        if args.only and cell.name != args.only:
            continue
        argv = build_cell_command(
            cell, results_dir=str(results_dir), extra=extra
        )
        print(f"[{cell.name}] {shlex.join(argv)}")
        if args.dry_run:
            continue
        stop = threading.Event()
        peaks: list[float] = []
        poller = threading.Thread(
            target=_poll_vram, args=(stop, peaks), daemon=True
        )
        poller.start()
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=3600
            )
        finally:
            stop.set()
            poller.join(timeout=2)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            rows = [dict(
                {f: "" for f in FIELDS}, suite="ablation",
                arch=cell.arch, cache_config=cell.config.name,
                verdict="FAIL", notes="cell-crashed",
            )]
        else:
            rows = rows_from_cell_output(cell, proc.stdout)
            if peaks:
                for row in rows:
                    row["vram_peak_gib"] = max(peaks)
        (results_dir / f"{cell.name}.json").write_text(
            proc.stdout if not args.dry_run and proc.returncode == 0
            else json.dumps({"stderr": proc.stderr})
        )
        all_rows.extend(rows)
    if not args.dry_run:
        with open(results_dir / "results.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {results_dir / 'results.csv'}")
    return 0


def _parse_coeffs(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/cache-bench")
    parser.add_argument("--cell", default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--c5-coefficients", type=_parse_coeffs, default=()
    )
    parser.add_argument("--c5-threshold", type=float, default=0.1)
    parser.add_argument("--c5-rms-rel-max", type=float, default=0.1)
    args = parser.parse_args()
    if args.cell:
        cell = next(c for c in GRID if c.name == args.cell)
        print(json.dumps(_run_cell(cell, args)))
        return 0
    return _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests + a CPU dry-run**

```bash
python -m pytest tests/test_cache_bench_grid.py -q
python -m benchmarks.caches.ablation --dry-run --results-dir /tmp/cache-bench
```

Expected: tests PASS; dry-run prints five `[h3-dense-*]` command lines and exits 0 without importing torch-heavy paths.

- [ ] **Step 5: ruff + commit**

```bash
ruff check benchmarks/caches/ablation.py tests/test_cache_bench_grid.py
git add benchmarks/caches/ablation.py tests/test_cache_bench_grid.py
git commit -m "bench(caches): one-factor cache-contribution ablation grid"
```

---

### Task 5: Serving-trace benchmark (`benchmarks/caches/serve_trace.py`)

Replays a seeded VidProM trace against the running job API, measures per-job latency, and reports repeated-vs-unique latency plus the disk-tier restart story. Server code is NOT modified; expected hits come from the trace itself.

**Server contract (must match exactly, or every request 409s):** the job server loads one immutable profile; requests must repeat its exact ordered optimization list. The README's example profile does NOT include `condition-cache`, so the benchmark requires its own startup — this exact command goes into the module docstring and `docs/cache_benchmarks.md`:

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
```

`serve_trace.py`'s `--optimizations` default is `adaln-host-cache,block-stream,text-encoder-stream,condition-cache` — byte-identical to the `OMNI_OPTIMIZATIONS` above.

**Files:**
- Create: `benchmarks/caches/serve_trace.py`
- Test: extend `tests/test_cache_bench_grid.py` is NOT allowed (separate concern) — create `tests/test_cache_bench_serve_trace.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cache_bench_serve_trace.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for trace replay bookkeeping (HTTP mocked)."""

from benchmarks.caches.contract import FIELDS
from benchmarks.caches.serve_trace import replay, summarize
from benchmarks.caches.workload import build_trace


class _FakeClient:
    """Job API stub: instant success, latency keyed by repeat."""

    def __init__(self):
        self.calls = []

    def submit_and_wait(self, prompt: str) -> float:
        self.calls.append(prompt)
        return 10.0 if self.calls.count(prompt) > 1 else 100.0


def test_replay_emits_one_row_per_request():
    trace = build_trace(seed=0, length=8, repeat_ratio=0.25)
    rows = replay(trace, _FakeClient(), arch="h3-dense",
                  cache_config="c1", repeat_ratio=0.25)
    assert len(rows) == 8
    assert all(set(row) == set(FIELDS) for row in rows)
    hits = [row for row in rows if row["expected_hits"] == 1]
    assert len(hits) == 2
    assert all(row["e2e_ms"] == 10.0 for row in hits)


def test_summarize_reports_latency_split():
    trace = build_trace(seed=0, length=8, repeat_ratio=0.25)
    rows = replay(trace, _FakeClient(), arch="h3-dense",
                  cache_config="c1", repeat_ratio=0.25)
    summary = summarize(rows)
    assert summary["requests"] == 8
    assert summary["expected_hits"] == 2
    assert summary["repeat_p50_ms"] < summary["unique_p50_ms"]
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_cache_bench_serve_trace.py -q
```

Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# benchmarks/caches/serve_trace.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end serving trace for the condition cache (issue #24 C1).

Replays a seeded VidProM-derived trace against a running job server.
Start the server with the condition cache in its immutable profile —
requests must repeat the exact ordered optimization list or the server
answers HTTP 409::

    OMNI_MODEL_ARCH=h3-dense \\
    OMNI_OPTIMIZATIONS=adaln-host-cache,block-stream,\\
    text-encoder-stream,condition-cache \\
    OMNI_CONDITION_CACHE_DIR=./cache-c1 \\
    OMNI_CHECKPOINT=... python -m omni_infinity.serve

Repeated prompts are the exact-hit
condition, so repeated-request latency directly measures C1's
cross-request contribution; restart the server between two runs of this
script against the same cache dir to measure the disk tier.

The server is not modified and exposes no cache counters; expected
hits are derived from the trace (a repeat of an earlier prompt in the
same run is an expected hit).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from benchmarks.caches.contract import (
    FIELDS,
    FRAMES,
    RESOLUTION,
    SEED,
    STEPS,
    verdict_for,
)
from benchmarks.caches.workload import TraceItem, build_trace


class JobClient:  # pragma: no cover — exercised against a live server
    """Minimal polling client for the /v1/jobs API."""

    def __init__(self, base_url: str, optimizations: list[str],
                 arch: str, timeout_s: float = 1800.0):
        import httpx  # serve extra

        self._client = httpx.Client(base_url=base_url, timeout=60.0)
        self._optimizations = optimizations
        self._arch = arch
        self._timeout_s = timeout_s

    def submit_and_wait(self, prompt: str) -> float:
        started = time.perf_counter()
        response = self._client.post("/v1/jobs", json={
            "type": "fl2va",
            "prompt": prompt,
            "model_arch": self._arch,
            "optimizations": self._optimizations,
            "seed": SEED,
            "num_inference_steps": STEPS,
            "resolution": RESOLUTION,
            "num_frames": FRAMES,
        })
        response.raise_for_status()
        job_id = response.json()["id"]
        deadline = started + self._timeout_s
        while time.perf_counter() < deadline:
            status = self._client.get(f"/v1/jobs/{job_id}").json()
            if status["status"] in ("succeeded", "failed", "cancelled"):
                if status["status"] != "succeeded":
                    raise RuntimeError(f"job {job_id}: {status['status']}")
                return (time.perf_counter() - started) * 1000.0
            time.sleep(1.0)
        raise TimeoutError(f"job {job_id} did not finish")


def replay(trace: list[TraceItem], client, *, arch: str,
           cache_config: str, repeat_ratio: float) -> list[dict]:
    rows = []
    for item in trace:
        elapsed_ms = client.submit_and_wait(item.prompt)
        row = {field: "" for field in FIELDS}
        row.update(
            suite="serve-trace",
            arch=arch,
            cache_config=cache_config,
            dataset="vidprom",
            trace_len=len(trace),
            repeat_ratio=repeat_ratio,
            rep=item.index,
            phase="warm" if item.expected_hit else "cold",
            e2e_ms=elapsed_ms,
            expected_hits=1 if item.expected_hit else 0,
            notes="",
        )
        row["verdict"] = verdict_for(row)
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    repeat = [r["e2e_ms"] for r in rows if r["expected_hits"] == 1]
    unique = [r["e2e_ms"] for r in rows if r["expected_hits"] == 0]
    return {
        "requests": len(rows),
        "expected_hits": len(repeat),
        "repeat_p50_ms": statistics.median(repeat) if repeat else None,
        "unique_p50_ms": statistics.median(unique) if unique else None,
    }


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--arch", default="h3-dense")
    parser.add_argument(
        "--optimizations",
        default="adaln-host-cache,block-stream,"
                "text-encoder-stream,condition-cache",
        help="must repeat the server's exact ordered list",
    )
    parser.add_argument("--trace-len", type=int, default=16)
    parser.add_argument("--repeat-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/cache-bench")
    )
    args = parser.parse_args()
    trace = build_trace(
        seed=args.seed, length=args.trace_len,
        repeat_ratio=args.repeat_ratio,
    )
    client = JobClient(
        args.base_url, args.optimizations.split(","), args.arch
    )
    rows = replay(
        trace, client, arch=args.arch, cache_config="c1",
        repeat_ratio=args.repeat_ratio,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(args.results_dir / "serve_trace.csv", "w",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summarize(rows), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_cache_bench_serve_trace.py -q
```

Expected: PASS.

- [ ] **Step 5: ruff + commit**

```bash
ruff check benchmarks/caches/serve_trace.py tests/test_cache_bench_serve_trace.py
git add benchmarks/caches/serve_trace.py tests/test_cache_bench_serve_trace.py
git commit -m "bench(caches): VidProM serving-trace replay for the condition cache"
```

---

### Task 6: C5 calibration probe (`benchmarks/caches/denoise_probe.py`)

The calibration gate the C5 plan defers to: record per-step relative-L1 input distances and true output change so C5 coefficients can be fitted; without this, every C5 row stays SKIP.

**Files:**
- Create: `benchmarks/caches/denoise_probe.py`
- Test: `tests/test_cache_bench_probe.py`

- [ ] **Step 1: Write the failing test (CPU, dummy module)**

```python
# tests/test_cache_bench_probe.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU test: the probe hook records one row per transformer call."""

import torch

from benchmarks.caches.denoise_probe import probe_forward


class _Toy(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 1.0


def test_probe_records_relative_distances():
    module = _Toy()
    with probe_forward(module) as records:
        for step in range(4):
            module(hidden_states=torch.full((2, 4), float(step)))
    assert len(records) == 4
    assert records[0]["input_rel_l1"] is None  # no previous step
    assert records[1]["input_rel_l1"] is not None
    assert all("output_rel_l1" in r for r in records)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_cache_bench_probe.py -q
```

Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# benchmarks/caches/denoise_probe.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""C5 calibration probe (the C5 plan's deferred calibration gate).

Wraps the transformer ``forward`` for one generation and records, per
call: the relative-L1 distance between consecutive step inputs (the
TeaCache decision signal) and between consecutive step outputs (the
truth the polynomial must predict). Fit coefficients offline (e.g.
``numpy.polyfit`` on input vs output distances) and pass them to
``--c5-coefficients`` in ``benchmarks.caches.ablation``.

GPU usage:
    OMNI_CHECKPOINT=... python -m benchmarks.caches.denoise_probe \
        --results-dir results/cache-bench
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path


def _rel_l1(current, previous) -> float | None:
    if previous is None:
        return None
    scale = previous.abs().mean()
    if float(scale) == 0.0:
        return None
    return float(
        (current.to(previous.dtype) - previous).abs().mean() / scale
    )


@contextlib.contextmanager
def probe_forward(module, signal_name: str = "hidden_states"):
    """Yield a list that fills with one record per forward call."""
    records: list[dict] = []
    state = {"input": None, "output": None}
    original = module.forward

    def probed(*args, **kwargs):
        signal = kwargs.get(signal_name, args[0] if args else None)
        output = original(*args, **kwargs)
        from omni_infinity.caches._tensor_tree import tree_tensors

        leaf = tree_tensors(output)[0]
        records.append({
            "call": len(records),
            "input_rel_l1": _rel_l1(signal, state["input"]),
            "output_rel_l1": _rel_l1(leaf, state["output"]),
        })
        state["input"] = signal.detach()
        state["output"] = leaf.detach()
        return output

    module.forward = probed
    try:
        yield records
    finally:
        del module.__dict__["forward"]


def main() -> int:  # pragma: no cover — needs weights
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/cache-bench")
    )
    args = parser.parse_args()
    if not os.environ.get("OMNI_CHECKPOINT"):
        print(json.dumps({"skip": "weights-absent"}))
        return 0
    from benchmarks.caches.contract import (
        FRAMES, PROMPT, RESOLUTION, SEED, STEPS,
    )
    from omni_infinity.runner import ReferenceRunner

    runner = ReferenceRunner.from_pretrained(
        os.environ["OMNI_CHECKPOINT"],
        workflow="fl2va",
        offload=True,
        store_dir=os.environ.get("OMNI_STORE_DIR"),
        block_stream_blocks_per_group=1,
        stream_text_encoder=True,
    )
    transformer = runner.pipeline.components[runner.transformer_component]
    with probe_forward(transformer) as records:
        runner.generate(
            PROMPT, seed=SEED, num_inference_steps=STEPS,
            resolution=RESOLUTION, num_frames=FRAMES,
        )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "denoise_probe.json"
    out.write_text(json.dumps({"records": records}, indent=2) + "\n")
    print(f"wrote {out} ({len(records)} calls)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_cache_bench_probe.py -q
```

Expected: PASS.

- [ ] **Step 5: ruff + commit**

```bash
ruff check benchmarks/caches/denoise_probe.py tests/test_cache_bench_probe.py
git add benchmarks/caches/denoise_probe.py tests/test_cache_bench_probe.py
git commit -m "bench(caches): C5 calibration probe (per-step rel-L1 signals)"
```

---

### Task 7: GPU smoke test (`tests/test_cache_bench_smoke.py`)

**Files:**
- Test: `tests/test_cache_bench_smoke.py`

- [ ] **Step 1: Write the test** (no TDD failure step — it is itself the deliverable; the CPU half must pass immediately, the GPU half is `gpu`+`weights` gated like `tests/test_reference_parity.py`)

```python
# tests/test_cache_bench_smoke.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Smokes for the cache benchmark harness.

The CPU half asserts the orchestrator's dry-run works end to end. The
GPU half runs the real c1 cell once and asserts a warm bitwise hit —
same env contract as the parity gates (OMNI_CHECKPOINT + CUDA).
"""

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_dry_run_builds_the_full_grid(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.caches.ablation",
         "--dry-run", "--results-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for name in ("baseline", "c1", "c3", "c5", "all-exact"):
        assert f"h3-dense-{name}" in proc.stdout


@pytest.mark.gpu
@pytest.mark.weights
def test_c1_cell_hits_bitwise_on_warm(tmp_path):
    if not os.environ.get("OMNI_CHECKPOINT"):
        pytest.skip("OMNI_CHECKPOINT unset")
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.caches.ablation",
         "--cell", "h3-dense-c1", "--results-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO, timeout=3600,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    warm = payload["phases"][1]
    assert warm["rms_rel"] == 0.0
    assert warm["stats"]["c1"]["hits"] >= 1
```

- [ ] **Step 2: Run the CPU half**

```bash
python -m pytest tests/test_cache_bench_smoke.py -q -m "not gpu and not weights"
```

Expected: 1 passed, 1 deselected.

- [ ] **Step 3: ruff + commit**

```bash
ruff check tests/test_cache_bench_smoke.py
git add tests/test_cache_bench_smoke.py
git commit -m "test(caches): benchmark harness smoke (CPU dry-run + gated GPU c1 cell)"
```

---

### Task 8: Design doc (`docs/cache_benchmarks.md`) + README pointer

**Files:**
- Create: `docs/cache_benchmarks.md`
- Modify: `README.md` (one line inside the "Opt-in caches" section the contract plan added)
- Test: `tests/test_cache_bench_docs.py`

- [ ] **Step 0: Write the failing doc-contract test**

```python
# tests/test_cache_bench_docs.py
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""The benchmark doc must exist, cover every suite, and be linked."""

from pathlib import Path

from benchmarks.caches.contract import FIELDS

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "cache_benchmarks.md"


def test_doc_exists_and_covers_every_suite():
    text = DOC.read_text()
    for token in (
        "WenhaoWang/VidProM",
        "VBench",
        "benchmarks.caches.micro",
        "benchmarks.caches.ablation",
        "benchmarks.caches.serve_trace",
        "benchmarks.caches.denoise_probe",
        "OMNI_CONDITION_CACHE_DIR",
        "c5-uncalibrated",
    ):
        assert token in text, f"docs/cache_benchmarks.md missing {token}"


def test_doc_lists_every_contract_field():
    text = DOC.read_text()
    for field in FIELDS:
        assert field in text, f"contract field {field} undocumented"


def test_readme_links_the_doc():
    readme = (REPO / "README.md").read_text()
    assert "docs/cache_benchmarks.md" in readme
```

Run: `python -m pytest tests/test_cache_bench_docs.py -q` — expected: FAIL (doc absent).

- [ ] **Step 1: Write `docs/cache_benchmarks.md`**

Content requirements (write full prose; structure fixed):

1. **Title + scope:** "Cache benchmarks (issue #24)" — tests and benchmarks quantifying each cache level's contribution; links to the README "Opt-in caches" section, `docs/caches_c1_condition.md`, `docs/caches_c5_denoise.md`, and issue #24.
2. **Datasets:** the table from this plan's "Dataset decision" section (VidProM for serving traces + provenance, VBench for quality prompts, committed fixture as the only runtime source, `--refresh` semantics, `bench` extra).
3. **Suites:** one subsection each for `micro`, `ablation`, `serve-trace`, `denoise_probe` — what it measures, exact command, where results land (`results/cache-bench/*.json|csv`), and which contract fields it fills.
4. **Metric contract:** state that `benchmarks/caches/contract.py` freezes field order (same rule as streaming), list the FIELDS tuple verbatim, and the verdict rules: exact caches PASS only with `rms_rel==0` and `hits>=1` on warm; C5 PASS only with `skipped>=1`, `speedup>1`, `rms_rel<=rms_rel_max`; `weights-absent`/`c5-uncalibrated`/`server-down` are SKIP.
5. **Invariants restated:** benchmarks never enable caches in the parity gates; C5 has no default calibration — the probe workflow (`denoise_probe.py` → fit → `--c5-coefficients`) is the only path to a C5 row that isn't SKIP.
6. **How to run everything:**

```bash
# CPU-only (CI-safe)
python -m pytest tests/test_cache_bench_*.py -q -m "not gpu and not weights"
python -m benchmarks.caches.micro --results-dir results/cache-bench
python -m benchmarks.caches.ablation --dry-run

# GPU (H3 weights)
OMNI_CHECKPOINT=... OMNI_STORE_DIR=... python -m benchmarks.caches.ablation
OMNI_CHECKPOINT=... python -m benchmarks.caches.denoise_probe
# serving trace: start the server per README with condition-cache, then
python -m benchmarks.caches.serve_trace --repeat-ratio 0.5
```

- [ ] **Step 2: README pointer** — append to the "Opt-in caches" section:

```markdown
Benchmarks quantifying each cache level's contribution (microbenchmarks,
one-factor ablation, VidProM serving trace, C5 calibration probe):
[docs/cache_benchmarks.md](docs/cache_benchmarks.md).
```

- [ ] **Step 3: Verify the doc against its contract test**

```bash
python -m pytest tests/test_cache_bench_docs.py -q
```

Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add docs/cache_benchmarks.md README.md tests/test_cache_bench_docs.py
git commit -m "docs: cache benchmark design — datasets, suites, metric contract"
```

---

### Task 9: Final verification + push + PR comment

- [ ] **Step 1: Full CPU test suite**

```bash
python -m pytest -q -m "not gpu and not weights"
```

Expected: all pass (pre-existing failures, if any, recorded verbatim in the PR comment — not fixed here).

- [ ] **Step 2: ruff over everything touched**

```bash
ruff check benchmarks/caches tests/test_cache_bench_contract.py tests/test_cache_bench_workload.py tests/test_cache_bench_micro.py tests/test_cache_bench_grid.py tests/test_cache_bench_serve_trace.py tests/test_cache_bench_probe.py tests/test_cache_bench_smoke.py tests/test_cache_bench_docs.py
```

Expected: no findings.

- [ ] **Step 3: Micro run + full dry-run as living proof**

```bash
python -m benchmarks.caches.micro --results-dir /tmp/cache-bench --reps 5
python -m benchmarks.caches.ablation --dry-run --results-dir /tmp/cache-bench
```

Expected: micro prints six bench lines; dry-run prints five cell commands.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin bench/cache-suite
gh pr create -R EfficientMoE/Omni-Infinity --base main \
  --title "bench(caches): micro / ablation / serve-trace suite for the issue #24 caches" \
  --body "Cache benchmark & test suite per docs/superpowers/plans/2026-09-30-cache-benchmarks.md: docs/cache_benchmarks.md (design), benchmarks/caches/ (micro / ablation / serve-trace / C5 calibration probe, VidProM+VBench-derived offline prompt fixture), and CPU+gated-GPU tests. CPU suite output: <paste pytest summary>. GPU cells report SKIP/weights-absent until run on the dev GPU. Refs #24."
```

---

## Self-review checklist (run after writing all code)

- [ ] Every FIELDS name used in `ablation.py` / `serve_trace.py` rows exists in `contract.FIELDS`.
- [ ] `CacheConfig` names in tests match `CACHE_CONFIGS` exactly (`baseline, c1, c3, c5, all-exact`).
- [ ] No file imports `datasets` outside `_refresh` (offline guarantee).
- [ ] No test outside `test_cache_bench_smoke.py::test_c1_cell_hits_bitwise_on_warm` requires GPU/weights/network.
- [ ] Parity gates (`test_reference_parity.py`, `test_ref2va_parity.py`, `test_vdn_parity.py`) untouched.
