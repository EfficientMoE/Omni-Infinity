# Multi-prompt demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept an ordered prompt list and play one stitched video at 15s, 1min, 2min, or 5min, where every prompt is one 124-frame generation.

**Architecture:** A new `omni_infinity/demo/` package owns the duration table, the inclusive prompt clock, frame/audio stitch, and a three-stage pipeline. Each clip flows through three pipeline slots named `encoder`, `backbone`, and `decoder`. The loaded runners are monolithic (`ReferenceRunner.generate`, `VdnRunner.generate` — one synchronous call runs text encoding, denoising, and decode), so the runner is called once per prompt inside the `backbone` slot; the `encoder` slot is the admission gate for that prompt's sampled second, and the `decoder` slot trims, fragments, pushes, and hands the last frame to the next clip. The slots stay on the one job worker, but a later prompt's encoder slot may run before the previous clip's decoder slot. The first clip is produced before playback. When its fragments are sent, the playback clock starts at 0. Each later prompt draws a whole second uniformly from X through Y inclusive and its encoder does not start before that second. The finished file is still one trimmed MP4. `POST /v1/jobs` and `GenerationRequest` stay single-prompt. The stream monitor picks the length from a dropdown, sets X and Y, and highlights the prompt whose interval contains `video.currentTime`.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, Pillow, NumPy, PyTorch, pytest, the existing `write_artifacts` / `fragment_clip` helpers. No new dependencies. No H3-World weights.

**Spec:** https://github.com/EfficientMoE/Omni-Infinity/issues/34

## Global Constraints

- Each segment is exactly 124 frames at 24 fps (`17*7+5`, 124/24 s). `num_frames` is not a demo request field.
- Presets and the smallest covering prompt counts are exactly: `15s` → 3 prompts (372 generated frames, play 360), `1min` → 12 (1488 generated, play 1440), `2min` → 24 (2976 generated, play 2880), `5min` → 59 (7316 generated, play 7200). 23 prompts is 118.8 s and is rejected for `2min`. 58 prompts is 299.7 s and is rejected for `5min`.
- Playback stops at the nominal duration: 15 s, 60 s, 120 s, 300 s. The final MP4 is trimmed to that duration. Fragments for the last clip stop at the same cut, before they are sent.
- The first clip is encoded, denoised, and decoded before any later prompt starts. Sending its fragments starts the playback clock at 0. The browser plays that clip while later clips are still running.
- `schedule_start` (X) and `schedule_end` (Y) are integers, `0 <= X <= Y`. For each prompt index `i >= 1`, the start second is `random.Random(seed).randint(X, Y)` drawn in index order. `randint` is inclusive on both ends. Prompt 0 has no draw. The encoder for prompt `i` does not start until the playback clock is at least that second.
- Encoder, backbone, and decoder are pipeline slots, not a refactor of the runners. `ReferenceRunner.generate` (`omni_infinity/runner.py`) and `VdnRunner.generate` (`omni_infinity/arch/vdn.py`) stay untouched; the `backbone` slot makes exactly one `generate()` call per prompt. The `encoder` slot does no model work — it admits the prompt at its sampled second. The `decoder` slot trims the last clip, fragments, pushes, and computes the handoff frame. On the single worker a tick runs every ready slot in the order backbone, encoder, decoder, each to completion. A due encoder slot may therefore run after the previous clip's backbone and before that clip's decoder slot. The backbone for clip `i > 0` still waits for clip `i - 1`'s decoder slot, which supplies the handoff frame. Two copies of the same slot are never in flight.
- The pipeline reads time from an injected `PlaybackClock` (`now()` / `wait_until(second)`). The streaming service anchors a monotonic clock when clip 0's fragments are pushed and its `wait_until` sleeps; tests and the non-streaming path use `SimClock`, whose `wait_until` fast-forwards instantly. `run_pipeline` itself never sleeps and, when no stage is ready, waits the clock to the earliest undrawn encoder second, so it terminates for any `X <= Y`.
- `fragment_clip` embeds fMP4 media timestamps that start at zero for every clip it encodes; the websocket `pts` field is metadata only. The demo therefore fragments each clip separately and puts the timeline shift on the client: chunk messages carry `clip: i` and `timestamp_offset: i * 124 / 24`, the server sends each clip's own `init` segment tagged with that `clip`, and the player sets `sourceBuffer.timestampOffset` to the clip's offset before appending that clip's init segment and fragments. Message `pts` is timeline seconds: `timestamp_offset` plus the fragment's clip-local time.
- Segment 0 uses the required first frame. Segment `i > 0` uses the last decoded frame of segment `i - 1` as its `image`. `last_image` is always `None`.
- Segment `i` is called with `seed + i`.
- `POST /v1/jobs` still accepts one `prompt` and `num_frames` in `120..360`. A demo id is HTTP 404 on `GET /v1/jobs/{id}`.
- Demo and job `generate()` share `JobService.executor` (`max_workers=1`). Do not construct a second pool.
- Profile mismatch is HTTP 409 with detail `request profile does not match the loaded server profile`. A bad prompt count is HTTP 422 with detail `duration {duration} requires {n} prompts, got {got}`. A missing first frame is HTTP 422 with detail `first frame is required`. Invalid image bytes reuse `JobService._decode_image` errors.
- `vdn-hybrid` with `resolution != "768p"` raises `ValueError("vdn-hybrid requires the 768p canvas")` before any `generate()`.
- `OMNI_STREAM_ENABLED` unset leaves `WS /v1/demos/{id}/ws` unregistered. `POST /v1/demos` stays registered either way.
- The length control is `<select id="duration">` with options `15s`, `1min`, `2min`, `5min`. It is not a free-text field. X and Y are `<input id="schedule-start">` and `<input id="schedule-end">`, integer seconds, defaults 1 and 5. The prompt row whose half-open interval contains `video.currentTime` has class `active`. Rows for clips the server has not sent yet stay visible and are marked rendering.
- CPU tests, fake runners, no checkpoint loads. Do not modify `third_party/`.
- New Python files start with `# Copyright (c) EfficientMoE.` and `# SPDX-License-Identifier: Apache-2.0`. Ruff line length stays 80.

## Review Focus

- `2min` with 23 prompts and `5min` with 58 prompts are 422; the covering counts are 24 and 59. Task 3.
- No played frame or fragment has `pts` at or past the nominal duration. Task 2 and Task 7.
- The first clip's websocket `chunk` is sent before any later encoder starts, and the player calls `video.play()` on that first chunk. Task 7 and Task 8.
- With X = Y = 4, every later prompt's encoder waits until playback second 4. With X = 0 and Y = 0, those encoders may start at second 0. Task 6.
- For three due clips, the stage order contains encoder of clip 2 before decoder of clip 1. Task 6.
- `run_pipeline` finishes when `schedule_start > 0` because the injected clock's `wait_until` advances it: `SimClock` fast-forwards instantly, the streaming clock sleeps until the playback second. Task 6 and Task 7.
- `ReferenceRunner.generate` and `VdnRunner.generate` are not modified; the backbone slot calls the monolithic runner once per prompt, and the encoder slot does no model work. Task 7.
- Clip `i > 0` keeps zero-based timestamps inside its fMP4 bytes; the message carries `timestamp_offset = i * 124 / 24` and the player sets `sourceBuffer.timestampOffset` before appending that clip's init segment, so the browser timeline is contiguous. Task 7 and Task 8.
- Segment 1's `image` is the last frame of segment 0, not the user first frame. Task 4.
- `POST /v1/jobs` still rejects a body that has `prompts` and no `prompt`. Task 5.
- A demo submit and a job submit share one executor, so the two `generate()` calls cannot overlap. Task 5.

---

## File Structure

Create:

- `omni_infinity/demo/__init__.py` — re-exports `segment_count`, `DemoRequest`, `DemoService`.
- `omni_infinity/demo/schedule.py` — preset table, frame counts, and inclusive prompt times.
- `omni_infinity/demo/pipeline.py` — encoder, backbone, and decoder tick order.
- `omni_infinity/demo/stitch.py` — last-frame image, concatenate, trim.
- `omni_infinity/demo/models.py` — `DemoRequest`, `DemoRecord`, `DemoResponse`.
- `omni_infinity/demo/store.py` — `DemoStore` under `{jobs_dir.parent}/demos`.
- `omni_infinity/demo/service.py` — `DemoService`.
- `tests/test_demo_schedule.py`
- `tests/test_demo_pipeline.py`
- `tests/test_demo_stitch.py`
- `tests/test_demo_service.py`
- `tests/test_demo_api.py`

Modify:

- `omni_infinity/serve/app.py` — demo routes, pass `service.executor` into `DemoService`.
- `omni_infinity/serve/stream.py` — export `chunk_message` (current `_chunk_message` body).
- `omni_infinity/serve/webui/index.html` — length `<select id="duration">` and `#prompt-stack`.
- `omni_infinity/serve/webui/player.js` — demo submit, play on the first chunk, highlight the playhead prompt.
- `README.md` — one section after Streaming playback.

Do not edit `GenerationRequest`, `JobStore.create`, or `POST /v1/jobs`.

---

### Task 1: Duration schedule

**Files:**
- Create: `omni_infinity/demo/__init__.py`
- Create: `omni_infinity/demo/schedule.py`
- Test: `tests/test_demo_schedule.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `FPS: int = 24`, `SEGMENT_FRAMES: int = 124`
  - `PRESETS: dict[str, int]` mapping `15s`→15, `1min`→60, `2min`→120, `5min`→300
  - `segment_count(duration_s: int) -> int` = `math.ceil(duration_s * FPS / SEGMENT_FRAMES)`
  - `playback_frames(duration_s: int) -> int` = `duration_s * FPS`
  - `generated_frames(duration_s: int) -> int` = `segment_count(duration_s) * SEGMENT_FRAMES`

- [x] **Step 1: Write the failing test**

```python
import pytest

from omni_infinity.demo.schedule import (
    generated_frames,
    playback_frames,
    segment_count,
)

@pytest.mark.parametrize(
    ("seconds", "segments", "play", "generated"),
    [
        (15, 3, 360, 372),
        (60, 12, 1440, 1488),
        (120, 24, 2880, 2976),
        (300, 59, 7200, 7316),
    ],
)
def test_covering_counts(seconds, segments, play, generated):
    assert segment_count(seconds) == segments
    assert playback_frames(seconds) == play
    assert generated_frames(seconds) == generated


def test_short_counts_do_not_cover():
    assert segment_count(120) != 23
    assert segment_count(300) != 58
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError` or import error for `omni_infinity.demo.schedule`

- [x] **Step 3: Implement the four names in `omni_infinity/demo/schedule.py`**

Use the formulas in Interfaces. Re-export `segment_count` from `omni_infinity/demo/__init__.py`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_demo_schedule.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add omni_infinity/demo/__init__.py omni_infinity/demo/schedule.py tests/test_demo_schedule.py
git commit -m "feat: add demo duration schedule"
```

---

### Task 2: Stitch and trim

**Files:**
- Create: `omni_infinity/demo/stitch.py`
- Test: `tests/test_demo_stitch.py`

**Interfaces:**
- Consumes: `GenerationResult` from `omni_infinity.runner`. `playback_frames` from Task 1.
- Produces:
  - `last_frame_image(frames) -> PIL.Image.Image`. Float frames are clipped to `[0, 1]`, multiplied by 255, and rounded to uint8. uint8 frames are copied. Mode is `RGB`.
  - `stitch_results(results: Sequence[GenerationResult], *, playback_frames: int) -> GenerationResult`. Concatenate `_video_frames` on axis 0 and `_stereo_audio` on the sample axis. Trim video to `playback_frames` rows. Trim audio to `round(playback_frames / 24 * sampling_rate)` samples. `sampling_rate` is the first result's rate; a later mismatch raises `ValueError`. `latents` and `audio_latents` on the return value are `None`. If the concatenated video has fewer than `playback_frames` rows, raise `ValueError`.

- [x] **Step 1: Write the failing test**

Build two `GenerationResult`s. Each video is `np.arange` float32 frames of shape `(4, 2, 2, 3)` scaled into `[0, 1]`. Each audio is `torch.ones(2, 8)`. Sampling rate 24 so one frame is one audio sample. Call `stitch_results(..., playback_frames=6)`.

Assert the video has 6 frames, the first 4 equal result 0, the next 2 equal result 1's first two frames, audio shape is `(2, 6)`, and `latents is None`. Assert `last_frame_image` on a `(1, 1, 3)` float frame of `1.0` is RGB `(255, 255, 255)`. Assert a second result with `sampling_rate=16_000` raises `ValueError`.

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo_stitch.py -v`
Expected: FAIL with import error for `omni_infinity.demo.stitch`

- [x] **Step 3: Implement `last_frame_image` and `stitch_results`**

Use `_video_frames` and `_stereo_audio` from `omni_infinity.serve.artifacts`. Do not mux here.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_demo_stitch.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add omni_infinity/demo/stitch.py tests/test_demo_stitch.py
git commit -m "feat: stitch and trim demo segments"
```

---

### Task 3: Demo request validation

**Files:**
- Create: `omni_infinity/demo/models.py`
- Test: `tests/test_demo_api.py` (validation cases only in this task)

**Interfaces:**
- Consumes: `PRESETS`, `segment_count` from Task 1. Optimization literals match `GenerationRequest.optimizations`.
- Produces:
  - `DemoRequest` fields: `prompts: list[str]` (each length 1..20000), `duration: Literal["15s","1min","2min","5min"]`, `schedule_start: int = Field(default=1, ge=0)`, `schedule_end: int = Field(default=5, ge=0)`, `model_arch: Literal["h3-dense","vdn-hybrid"] = "h3-dense"`, `optimizations` (same names and unique-name rule as `GenerationRequest`), `seed: int = 0`, `num_inference_steps: int = Field(default=8, ge=1, le=100)`, `resolution: Literal["256p","512p","768p"] = "256p"`, `first_frame_base64: str | None = None`.
  - If `schedule_end < schedule_start`, `ValueError` with message `schedule end is before schedule start`.
  - Model validator: `len(prompts) == segment_count(PRESETS[duration])`, else `ValueError` with message `duration {duration} requires {n} prompts, got {got}`.
  - `DemoRecord` and `DemoResponse` mirror `JobRecord` / `JobResponse` with `request: DemoRequest`.

- [x] **Step 1: Write the failing test**

Name the tests `test_prompt_count_rejects_short_lists` and `test_duplicate_optimizations_rejected`.

`test_prompt_count_rejects_short_lists`: `DemoRequest(prompts=["a", "b"], duration="15s")` raises `ValidationError` matching `duration 15s requires 3 prompts, got 2`. 23 prompts at `2min` matches `requires 24 prompts, got 23`. 58 prompts at `5min` matches `requires 59 prompts, got 58`. A 3-prompt `15s` request validates.

`test_duplicate_optimizations_rejected`: `optimizations=["fp8", "fp8"]` raises `ValidationError` matching `optimization names must be unique`.

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo_api.py -v -k "prompt_count or duplicate"`
Expected: FAIL with import error for `DemoRequest`

- [x] **Step 3: Implement `DemoRequest` in `omni_infinity/demo/models.py`**

No `num_frames` field. No `type` field. No `last_frame_base64`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_demo_api.py -v -k "prompt_count or duplicate"`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add omni_infinity/demo/models.py tests/test_demo_api.py
git commit -m "feat: validate demo prompt counts"
```

---

### Task 4: Sequential generation

**Files:**
- Create: `omni_infinity/demo/service.py`
- Test: `tests/test_demo_service.py`

**Interfaces:**
- Consumes: `segment_count`, `playback_frames`, `PRESETS`, `last_frame_image`, `stitch_results`, `DemoRequest`.
- Produces:
  - `DemoService._segment(i: int, request: DemoRequest, image: Image.Image, *, step_callback) -> GenerationResult` — the single per-clip runner call. Task 7's backbone slot reuses it unchanged.
  - `DemoService.run(request: DemoRequest, first: Image.Image, *, step_callback) -> GenerationResult` — a sequential loop over `_segment` with the last-frame handoff.
  - `_segment` calls `runner.generate(prompt, seed=request.seed + i, num_frames=124, image=frame, last_image=None, num_inference_steps=..., resolution=..., step_callback=...)` for `h3-dense`.
  - For `vdn-hybrid`, pass `num_evaluations=request.num_inference_steps` and omit `resolution` from the runner kwargs, matching `JobService._generate`. If `request.resolution != "768p"`, raise `ValueError("vdn-hybrid requires the 768p canvas")` before the first call.
  - `step_callback(completed, total)` uses `total = segment_count * num_inference_steps` and `completed = i * num_inference_steps + segment_completed`.
  - Returns `stitch_results(..., playback_frames=playback_frames(PRESETS[request.duration]))`.

- [x] **Step 1: Write the failing test**

Fake runner records `(prompt, seed, num_frames, image, last_image)` and returns 124 frames. Frame `t` of segment `i` is filled with scalar `(i + 1) / 10`. Audio is `torch.zeros(2, 124 * 48000 // 24)`. Sampling rate 48000.

Call `run` with three prompts, `duration="15s"`, `seed=7`, and a red first image. Assert three calls, seeds `7, 8, 9`, `num_frames == 124`, `last_image is None`, call 0's image is the red image, and call 1's image pixel equals segment 0's last frame via `last_frame_image`. Assert the stitched video has 360 frames. Assert a `vdn-hybrid` request at `256p` raises `ValueError` matching `768p` and the fake `generate` was not called.

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo_service.py -v`
Expected: FAIL with import error for `DemoService`

- [x] **Step 3: Implement `DemoService.run`**

The constructor takes `runner` and `model_arch: str`. Do not open a thread pool in this task.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_demo_service.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add omni_infinity/demo/service.py tests/test_demo_service.py
git commit -m "feat: generate demo segments in order"
```

---

### Task 5: HTTP demo API

**Files:**
- Create: `omni_infinity/demo/store.py`
- Modify: `omni_infinity/demo/service.py`
- Modify: `omni_infinity/serve/app.py`
- Test: `tests/test_demo_api.py`

**Interfaces:**
- Consumes: `DemoService.run`, `JobService.executor`, `JobService._decode_image`, `write_artifacts`, `JobStatus` transitions from `omni_infinity.serve.store.ALLOWED_TRANSITIONS`.
- Produces:
  - `DemoStore` rooted at `jobs_dir.parent / "demos"`, with `create`, `get`, `transition`, `update_progress`, `video_path`. Record file is `demo.json`. `create` sets `progress.total_steps` to `segment_count(PRESETS[duration]) * num_inference_steps`.
  - `DemoService.submit(request) -> DemoRecord` decodes the first frame, stores it as `input-first.png`, and submits `_execute` on the injected executor.
  - `POST /v1/demos` → HTTP 202, body `DemoResponse`, header `Location: /v1/demos/{id}`.
  - `GET /v1/demos/{id}` returns that record. Unknown id is HTTP 404.
  - `GET /v1/demos/{id}/artifacts` is HTTP 409 with detail `artifact is not ready` until `SUCCEEDED` and `output.mp4` exists, then a `video/mp4` file.
  - `GET /v1/jobs/{demo_id}` is HTTP 404.

- [x] **Step 1: Write the failing test**

Use `TestClient` and a fake runner, same pattern as `tests/test_job_api.py`. A 3-prompt `15s` post returns 202 and a 32-hex id. After the executor finishes, the artifact response is `video/mp4` and `av` counts 360 frames. `GET /v1/jobs/{id}` is 404. `POST /v1/jobs` with `{"type":"fl2va","prompts":["a","b","c"],"duration":"15s"}` is 422. Two-prompt `15s` is 422 matching `requires 3 prompts, got 2`. Missing `first_frame_base64` is 422 matching `first frame is required`. A profile whose `model_arch` differs from the server is 409 matching `request profile does not match the loaded server profile`.

Shared-executor test: construct `JobService` and `DemoService` in-process, assert `demo.executor is jobs.executor`. Submit a job whose fake `generate` sets a threading event and waits on a second event, then submit a demo; assert the demo's `generate` has not started until the job releases. `max` overlapping calls is 1.

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo_api.py -v`
Expected: FAIL with 404 on `POST /v1/demos`

- [x] **Step 3: Implement store, submit, and routes**

`create_app` builds `DemoStore(configured.jobs_dir.parent / "demos")` and `DemoService(..., executor=service.executor)`. `_execute` mirrors `JobService._execute`: `RUNNING`, `run`, `write_artifacts` with `artifact_url=f"/v1/demos/{id}/artifacts"`, `SUCCEEDED`; on exception unlink partial media and transition `FAILED`. Decode the first frame with `JobService._decode_image`. Empty `first_frame_base64` raises `InvalidMedia("first frame is required")` before decode.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_demo_api.py tests/test_job_api.py -v`
Expected: PASS, including the existing job tests

- [x] **Step 5: Commit**

```bash
git add omni_infinity/demo/store.py omni_infinity/demo/service.py omni_infinity/serve/app.py tests/test_demo_api.py
git commit -m "feat: serve the multi-prompt demo"
```

---

### Task 6: Inclusive prompt clock and module pipeline

**Files:**
- Modify: `omni_infinity/demo/schedule.py`
- Create: `omni_infinity/demo/pipeline.py`
- Test: `tests/test_demo_schedule.py`
- Test: `tests/test_demo_pipeline.py`

**Interfaces:**
- Consumes: `random.Random.randint`, which includes both endpoints.
- Produces:
  - `prompt_times(n: int, start: int, end: int, seed: int) -> list[int | None]` in `schedule.py`. Length `n`. Index 0 is `None`. Each later item is one `randint(start, end)` from `random.Random(seed)`, drawn in index order from that same generator.
  - `PlaybackClock` protocol in `pipeline.py` with `now() -> float` and `wait_until(second: float) -> None`.
  - `SimClock` implements `PlaybackClock`: `now()` starts at `0.0`; `wait_until(s)` sets the stored second to `max(now(), s)` and returns immediately. It never sleeps.
  - `ready_stages(done: set[tuple[str, int]], times: list[int | None], now: float) -> list[tuple[str, int]]`. Stage names are `encoder`, `backbone`, and `decoder`. Returns stages not in `done` whose prerequisites are met, in backbone, encoder, decoder order. `("encoder", i)` with `i > 0` is ready only when `now >= times[i]`. `("backbone", i)` needs `("encoder", i)` done and, for `i > 0`, `("decoder", i - 1)` done. `("decoder", i)` needs `("backbone", i)` done.
  - `run_pipeline(n: int, times: list[int | None], *, clock: PlaybackClock | None = None, run_stage: Callable[[str, int], None] | None = None) -> list[tuple[str, int]]`. `clock=None` means a fresh `SimClock`. `run_stage=None` means a no-op. Clip 0 runs `encoder`, `backbone`, `decoder` without consulting the clock; from `("decoder", 0)` on, the clock is live. Each tick evaluates `ready_stages` once with `clock.now()`, then runs every returned stage to completion in that order, calling `run_stage(name, i)` and appending `(name, i)` to the result. A stage finished mid-tick becomes ready in the next tick, so two copies of one module are never in flight. If no stage is ready and stages remain, call `clock.wait_until(t)` where `t` is the smallest `times[i]` over clips whose encoder has not run, then tick again. With `SimClock` that wait is an instant fast-forward, so `run_pipeline` terminates without sleeping. Task 7 supplies the real clock.

- [x] **Step 1: Write the failing tests**

`test_prompt_times_are_inclusive` (in `tests/test_demo_schedule.py`): `prompt_times(4, 4, 4, seed=0) == [None, 4, 4, 4]`. `prompt_times(3, 0, 0, seed=1) == [None, 0, 0]`. Every value of `prompt_times(8, 2, 5, seed=7)` is `None` or in `range(2, 6)`.

`test_later_encoder_waits_for_its_second` (in `tests/test_demo_pipeline.py`): `ready_stages(done={("encoder", 0), ("backbone", 0), ("decoder", 0)}, times=[None, 4], now=3)` does not contain `("encoder", 1)`. The same call with `now=4` does. `clock = SimClock()`; `run_pipeline(2, [None, 4], clock=clock)` emits `("decoder", 0)` before `("encoder", 1)` and finishes with `clock.now() == 4`.

`test_next_encoder_runs_before_previous_decoder`: with `times=[None, 0, 0]`, the returned stage list contains `("encoder", 2)` before `("decoder", 1)`, and `("backbone", 2)` after `("decoder", 1)`.

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_demo_schedule.py::test_prompt_times_are_inclusive tests/test_demo_pipeline.py -v`
Expected: FAIL with import errors

- [x] **Step 3: Implement `prompt_times`, the clock protocol, `SimClock`, `ready_stages`, and `run_pipeline`**

Readiness is evaluated once at the start of each tick with `clock.now()`. `run_pipeline` itself never sleeps; any waiting lives in the injected clock's `wait_until`.

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_demo_schedule.py tests/test_demo_pipeline.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add omni_infinity/demo/schedule.py omni_infinity/demo/pipeline.py tests/test_demo_schedule.py tests/test_demo_pipeline.py
git commit -m "feat: schedule later prompts on an inclusive clock"
```

---

### Task 7: Play the first clip while later clips generate

**Files:**
- Modify: `omni_infinity/serve/stream.py`
- Modify: `omni_infinity/serve/app.py`
- Modify: `omni_infinity/demo/service.py`
- Test: `tests/test_demo_api.py`

**Interfaces:**
- Consumes: `fragment_clip` from `omni_infinity.streaming.fragment`. `prompt_times`, `run_pipeline`, `PlaybackClock`, and `SimClock` from Task 6. `chunk_message` is `stream.py`'s current `_chunk_message` made public; keep `_chunk_message = chunk_message`.
- Produces: when `stream_enabled` is true, `WS /v1/demos/{id}/ws` sends clip 0's `init` and `chunk` messages before any later encoder slot runs. `DemoService` drives clips through `run_pipeline` with a real `PlaybackClock` and this `run_stage` mapping:
  - `("encoder", i)`: no model work — record that prompt `i` is admitted. The loaded runners are monolithic, so admission is the only encoder-slot effect.
  - `("backbone", i)`: one `DemoService._segment(i, ...)` call (Task 4). Clip 0's `image` is the request first frame; clip `i > 0` uses the handoff frame stored by decoder slot `i - 1`.
  - `("decoder", i)`: trim clip `i` to the nominal cut if it is the last clip, call `fragment_clip` on it, send one `init` message with that clip's init segment and `clip: i`, then that clip's `chunk` messages, then store `last_frame_image` of the untrimmed clip as the next handoff frame. The `run_stage` for `("decoder", 0)` records `t0 = time.monotonic()` immediately after clip 0's messages are pushed.
  - The clock: `now()` returns `time.monotonic() - t0`, and `wait_until(s)` sleeps in increments of at most 0.05 s until `now() >= s`, so a later encoder slot runs only once the playback clock reaches that prompt's sampled second.
  - Messages: every `chunk` carries `clip: i`, `timestamp_offset: i * 124 / 24`, and `pts` equal to `timestamp_offset` plus the fragment's clip-local time — the fMP4 bytes themselves keep clip-local timestamps starting at zero, and the browser applies the shift (Task 8). Each chunk's `prompt` is that clip's prompt. `instruction` and `action` are `None`. Build the dict with `chunk_message` and add the demo keys. `end` is sent only after the last clip. The last chunk has `done: true`. No chunk has `pts >=` nominal seconds.
  - The final MP4 is the stitched, trimmed timeline, written before `SUCCEEDED`. When `stream_enabled` is false, that websocket route is unregistered (404), `POST /v1/demos` still works, the service passes `SimClock()` so nothing sleeps, and the MP4 appears only after every clip has finished.

- [x] **Step 1: Write the failing test**

Name it `test_stream_starts_at_the_first_clip`. With `stream_enabled=True` and `stream_chunk_frames=124`, a 15s demo posts `schedule_start=0` and `schedule_end=0` so later encoders are due at playback second 0 and the test never waits on the real clock. The fake runner's second `generate()` blocks on a `threading.Event`. The websocket, opened immediately after HTTP 202, receives an `init` with `clip == 0` and a chunk with `prompts[0]` while that event is still unset. Releasing the event lets clips 2 and 3 finish. Collected `pts` values are all `< 15`. Each clip `i > 0` is preceded by its own `init` with `clip == i`. The chunk at `pts == 124/24` has `prompts[1]`, `clip == 1`, and `timestamp_offset == 124/24`; decoding that clip's `init` bytes plus its fragment bytes with `av` yields a first video packet at clip-local time 0, which proves the timeline shift lives in `timestamp_offset`, not in the media. The last chunk has `done: true`. With `stream_enabled=False`, `POST /v1/demos` returns 202 and the websocket path is 404.

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo_api.py -v -k stream`
Expected: FAIL because the websocket route is missing

- [x] **Step 3: Implement per-clip send**

Drive clips through `run_pipeline` from Task 6 with the real `PlaybackClock` and the `run_stage` mapping described in Interfaces. After decoder slot 0, that clip's messages are pushed before any later encoder slot. Do not modify `ReferenceRunner.generate` or `VdnRunner.generate`. Do not wait for `stitch_results` before the first chunk. Do not call `ClipChunker`. Do not register the route unless `configured.stream_enabled`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_demo_api.py tests/test_stream_api.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add omni_infinity/serve/stream.py omni_infinity/serve/app.py omni_infinity/demo/service.py tests/test_demo_api.py
git commit -m "feat: play the first demo clip while generating the rest"
```

---

### Task 8: Length dropdown and playhead highlight

**Files:**
- Modify: `omni_infinity/serve/webui/index.html`
- Modify: `omni_infinity/serve/webui/player.js`
- Modify: `README.md` (after the Streaming playback section)
- Test: `tests/test_stream_api.py` (`test_webui_is_served_only_when_streaming_is_enabled`)

**Interfaces:**
- Consumes: preset ids `15s`, `1min`, `2min`, `5min` and clip counts 3, 12, 24, 59 from Task 1. Messages from Task 7: per-clip `init` messages tagged `clip`, and `chunk` messages carrying `pts`, `duration`, `prompt`, `clip`, and `timestamp_offset`.
- Produces:
  - `<select id="duration" aria-label="Playback length">` with those four values. Option labels name the length and the clip count (`15 seconds — 3 clips`, `1 minute — 12 clips`, `2 minutes — 24 clips`, `5 minutes — 59 clips`).
  - Source option `value="demo"`. Choosing it reveals `#prompt-stack`: one text field per clip. Changing the dropdown rebuilds the stack and keeps text already typed.
  - On submit, demo mode `POST`s `/v1/demos` with `prompts`, `duration`, and `first_frame_base64`, then opens `WS /v1/demos/{id}/ws`. Clip and native keep `POST /v1/streams`.
  - In demo mode, an `init` message with `clip > 0` waits for pending appends to drain, sets `sourceBuffer.timestampOffset` to the message's `timestamp_offset` (`clip * 124 / 24`), appends that clip's init segment bytes, then appends its fragments. Clip 0 keeps the existing init path with `timestampOffset` 0. This is what places every clip's zero-based fMP4 timestamps onto the shared browser timeline.
  - Before any chunk, the rail lists every prompt as rendering. `highlightCue` adds `active` only to the row whose half-open `[pts, pts + duration)` contains `video.currentTime`. The first `chunk` calls `video.play()`. Until `end`, the status reads `live · model running`.

- [x] **Step 1: Extend the web UI test**

In `test_webui_is_served_only_when_streaming_is_enabled`, assert the page contains `id="duration"`, `id="schedule-start"`, `id="schedule-end"`, `value="15s"`, `value="1min"`, `value="2min"`, `value="5min"`, and `value="demo"`. Assert `player.js` contains `video.play`, `model running`, `highlightCue`, `schedule_start`, `schedule_end`, and `timestampOffset`.

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_api.py::test_webui_is_served_only_when_streaming_is_enabled -v`
Expected: FAIL on `id="duration"`

- [x] **Step 3: Implement the monitor controls**

Keep the existing dark monitor. The length control is the styled `<select>`, not a new text field. Prompt rows use the existing `.cue` / `.cue.active` treatment. Add `.cue.rendering` for clips not yet received. Do not add a font that requires a network fetch.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_api.py::test_webui_is_served_only_when_streaming_is_enabled -v`
Expected: PASS

- [x] **Step 5: README**

Add a "Multi-prompt demo" section with the four rows (prompts, generated frames, played frames) and a `curl` `POST /v1/demos` example for `15s` with three prompts and `first_frame_base64`. State that playback starts when the first clip is generated, later clips still run on the same worker, the length control is the dropdown, and the prompt under the playhead is highlighted. `POST /v1/jobs` is unchanged. H3-World weights are not in this change.

Run: `rg -n "POST /v1/demos" README.md`
Expected: one match, and nearby text lists `59` prompts for `5min` and `first clip`

- [x] **Step 6: Commit**

```bash
git add omni_infinity/serve/webui/index.html omni_infinity/serve/webui/player.js README.md tests/test_stream_api.py
git commit -m "feat: pick demo length from a dropdown and highlight the playhead prompt"
```

---

## Test plan

Run from the repo root after Task 8:

```bash
pytest tests/test_demo_schedule.py tests/test_demo_pipeline.py tests/test_demo_stitch.py tests/test_demo_service.py tests/test_demo_api.py tests/test_job_api.py tests/test_stream_api.py -v
ruff check omni_infinity/demo omni_infinity/serve/app.py omni_infinity/serve/stream.py tests/test_demo_schedule.py tests/test_demo_stitch.py tests/test_demo_service.py tests/test_demo_api.py
ruff format --check omni_infinity/demo omni_infinity/serve/app.py omni_infinity/serve/stream.py tests/test_demo_schedule.py tests/test_demo_stitch.py tests/test_demo_service.py tests/test_demo_api.py
```

Expected: pytest PASS, ruff clean.

Out of scope for these tests: loading H3-World or MiniMax-H3 weights, and a real GPU generation. The 1min, 2min, and 5min presets are covered by the schedule, the validation tests, and the dropdown markup, not by generating 59 fake clips. The HTTP generation test uses the 15s preset only. `test_stream_starts_at_the_first_clip` is the check that playback media leaves the server before a later encoder starts. `test_prompt_times_are_inclusive` and `test_next_encoder_runs_before_previous_decoder` cover the X–Y clock and the encoder/backbone/decoder order. The page test checks the length dropdown, the X and Y inputs, and that `highlightCue` still keys off `video.currentTime`.
