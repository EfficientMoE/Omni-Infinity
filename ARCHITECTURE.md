# Architecture

A contributor-oriented map of the Omni-Infinity codebase: the core inference
path, the job-serving and streaming paths, the module layout, and the request
lifecycle. For the user-facing operator guide see the
[README](README.md); for the longer design notes see the
[Documentation Hub](docs/README.md).

## Big picture

```text
                    client (curl / examples/*.py / omni_infinity.client)
                                      |
                       HTTP / WebSocket (omni_infinity/serve)
                                      |
  +-----------------------------------------------------------------------+
  |  serve/app.py  create_app + ServerSettings (one immutable profile)    |
  |     |                                                                 |
  |  serve/service.py  JobService  (ThreadPoolExecutor, max_workers=1)    |
  |     |                      \                                          |
  |  serve/store.py JobStore     serve/stream.py StreamService / demo     |
  +-----|-----------------------------------------------------------------+
        |  resolve_profile()                      one serialized GPU worker
        v
  registry.py  (model-arch x optimization)  ---> runner
        |                                          |
        |                       +------------------+------------------+
        v                       v                                     v
  runner.ReferenceRunner   arch.vdn.VdnRunner                 (same memory levers)
  (h3-dense)               (vdn-hybrid)
        |
        |  wraps the UNMODIFIED diffusers MiniMax-H3 modular pipeline
        v
  memory & compute levers (opt-in, resolved from the profile):
    store.py          StoreComponentSource  (moe-store v2, one-read groups)
    adaln.py          HostResidentAdaLN      (host-resident AdaLN branches)
    fp8.py / kernels/ ScaledFp8Linear + fused_fp8_gemm (Triton | reference)
    step_overlap.py   StepOverlapController  (cross-step weight prefetch)
    caches/           condition (C1) · vision (C3) · denoise (C5)
        |
        v
  serve/artifacts.py  write_artifacts -> jobs/<uuid>/output.wav + output.mp4
```

The serving process loads **one** registry profile at startup and reuses that
runner for every request; GPU work is serialized through a single in-process
worker. Streaming and the multi-prompt demo sit beside the job API and share
that same worker.

## Package layout

| Module | Responsibility |
|---|---|
| `omni_infinity/__init__.py` | Exposes `__version__` only (via `importlib.metadata`). |
| `registry.py` | Category registry — `ArchSpec`, `OptimizationSpec`, `ResolvedProfile`; `resolve_profile`, `runner_class`, `runner_kwargs_for`, `load_cache_optimizations`. |
| `runner.py` | `ReferenceRunner` (model-arch `h3-dense`) wrapping the unmodified H3-Base modular pipeline; `GenerationResult`, `resolve_resolution`. |
| `arch/vdn.py` | `VdnRunner` (model-arch `vdn-hybrid`) for VDN-Minimax-H3 hybrid attention. |
| `store.py` | `StoreComponentSource` — component weights from a moe-store v2 store with one-read group fetches; `load_diffusers_component`, `load_transformer_with_adaln_cache`. |
| `adaln.py` | `HostResidentAdaLN` / `AdaLNEntry` — host-resident AdaLN branch cache. |
| `fp8.py` | `ScaledFp8Linear`, `quantize_per_row_fp8`, `apply_scaled_fp8_casting` — opt-in FP8 weight path. |
| `kernels/` | Op-centric facade: `fused_fp8_gemm` dispatching to `_fused_fp8_gemm` (Triton) or `_reference` (pure torch); `_quant` block-wise 128×128 quantization. |
| `step_overlap.py` | `StepOverlapController` / `enable_step_overlap` — cross-step prefetch adapter for Diffusers 0.40.x group-offloading hooks. |
| `caches/` | Opt-in inference caches: `attach` (fail-closed binding), `condition` (C1), `vision` (C3), `denoise` (C5), `_tensor_tree` helpers. |
| `serve/` | Job-serving + streaming HTTP app (see below). |
| `streaming/` | Model-agnostic chunk layer: `chunks` (`MediaChunk`, `ActionCue`, `ChunkSource`), `clip` (`ClipChunker`), `native` (`NativeChunker` stub), `fragment` (fMP4). |
| `demo/` | Multi-prompt demo: `models`, `service`, `schedule`, `stitch`, `store`, `pipeline`. |
| `client/` | `player.py` — reference WebSocket client used by `examples/stream_play.py`. |
| `optim/` | Optimization-category marker package. |

## Serving path (`omni_infinity/serve/`)

| File | Responsibility |
|---|---|
| `app.py` | `ServerSettings`, `create_app`, `load_runner`; resolves one immutable profile per process. |
| `service.py` | `JobService` and its errors (`ProfileConflict`, `Ref2VANotImplemented`, `SessionLimit`, `InvalidMedia`); owns the single-worker executor. |
| `store.py` | `JobStore` + error/transition types — atomic on-disk job persistence and state transitions. |
| `models.py` | Pydantic schemas: `GenerationRequest`, `JobRecord`, `JobResponse`, `Progress`, `JobStatus`, `ArtifactMetadata`. |
| `artifacts.py` | `write_artifacts` → `output.wav` (SoundFile) + `output.mp4` (`diffusers … encode_video`). |
| `stream.py` | `StreamService` / `StreamSession`, `DemoStreamService` / `DemoStreamSession`, `ChunkPipe`, `chunk_message`. |
| `__main__.py` | `python -m omni_infinity.serve` entry point. |

## Request lifecycle — job

1. `create_app` resolves one `ResolvedProfile` (model-arch + ordered
   optimizations) and loads the matching runner once.
2. `POST /v1/jobs` validates the body against `GenerationRequest`. A request
   whose `model_arch` or optimization list differs from the loaded profile
   returns **HTTP 409** (`ProfileConflict`).
3. The job is persisted `queued` and returned as **HTTP 202** with a
   `Location: /v1/jobs/<uuid>` header.
4. The single worker runs the runner. `Progress` advances after each denoising
   transformer forward (`0/N → N/N`).
5. On success, `write_artifacts` writes `output.wav` and `output.mp4`.
   `GET /v1/jobs/{id}/artifacts` returns **HTTP 409** until then, and the MP4
   afterwards. Unknown IDs return **HTTP 404**.
6. `type="ref2va"` is schema-valid but returns **HTTP 501**
   (`Ref2VANotImplemented`); the supported Ref2VA path is
   `examples/ref2va_smoke.py`. On restart, persisted `running` jobs become
   `failed`.

## Request lifecycle — streaming & demo

- `POST /v1/streams` creates a passive `StreamSession` and returns a
  `stream_id`. `WS /v1/streams/{id}/ws` emits `init` → `chunk*` → `end`.
- `source="clip"` runs the one-shot runner and fragments the decoded result
  via `ClipChunker` + `fragment`; `source="native"` uses the 16×16
  `NativeChunker` stub. Neither overlaps generation with playback.
- `POST /v1/demos` drives `DemoStreamService` for the multi-prompt timeline
  (`demo/schedule` + `demo/stitch`), calling the loaded runner once per prompt.
- Every finished stream also publishes the ordinary job artifact.

## Registry model

Two orthogonal axes in `registry.py` describe every configuration:

- **model-arch** (`arch/`): `h3-dense` → `runner.ReferenceRunner`,
  `vdn-hybrid` → `arch.vdn.VdnRunner`.
- **optimization** (`optim/`): `adaln-host-cache`, `fp8`, `block-stream`,
  `text-encoder-stream`, plus the opt-in caches (`condition-cache`,
  `vision-cache`). `resolve_profile` turns a `(model_arch, optimizations)`
  pair into a `ResolvedProfile` that the runner and server consume.

## Public surface

- `omni_infinity.__version__`.
- Runners: `omni_infinity.runner.ReferenceRunner`,
  `omni_infinity.arch.vdn.VdnRunner`.
- Profile resolution: `omni_infinity.registry.resolve_profile` (and helpers).
- Serving: `omni_infinity.serve.create_app`, `python -m omni_infinity.serve`.
- CLIs: `examples/fl2va_smoke.py`, `examples/ref2va_smoke.py`,
  `examples/vdn_smoke.py`, `examples/stream_play.py`.

## Testing & CI

CPU-only unit tests and the Ruff gate run in GitHub Actions
(`.github/workflows/ci.yml`); GPU/weights tests are marked and excluded there.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the commands and conventions.
