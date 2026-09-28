# Streaming Chunk Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-agnostic fMP4 stream beside the existing job API so a client can play chunks and read the prompt before the finished MP4 is downloadable, including a stub input channel for a later H3-World runner.

**Architecture:** `omni_infinity/streaming/` turns a `GenerationResult` or a scripted stub into `MediaChunk`s. `StreamService` accepts sessions through the same profile checks and the same `JobService` executor as jobs, then pushes one WebSocket message per chunk. The finished file is still `write_artifacts` under `jobs/<id>/`. Local and browser players consume that socket. H3-World itself is not in this plan.

**Tech Stack:** Python 3.10+, FastAPI, Starlette `TestClient` websockets, Pydantic v2, PyAV (`av`) already in the `serve` extra, pytest. No new dependencies.

**Spec:** `docs/streaming_chunk_playback.md`

## Global Constraints

- Branch the implementation from `docs/streaming-chunk-playback` (spec commit), as `feat/streaming-chunk-playback`. Do not commit onto the docs PR branch and do not start from `main` without that spec file.
- `OMNI_STREAM_ENABLED` defaults off. Unset, empty, and `0` leave `POST /v1/streams` and `/` unregistered. Truthy values are `1`, `true`, and `yes`, case insensitive.
- `OMNI_STREAM_MAX_SESSIONS` defaults to `1`. `OMNI_STREAM_CHUNK_FRAMES` defaults to `24`. `OMNI_STREAM_FALLBACK_HLS` defaults off, same truthy set as the enable flag.
- `OMNI_WORKERS` stays `1`. A stream and a job share `JobService.executor` (`max_workers=1`).
- Profile mismatch is HTTP 409 with detail `request profile does not match the loaded server profile`. `type="ref2va"` is HTTP 501 with detail `Ref2VA requires Task 3 (issue #8), which is not merged`. Invalid media is HTTP 422. A session over the cap is HTTP 409 with detail `stream session limit reached`.
- `POST /v1/streams` returns HTTP 202 and `{"stream_id": "<32 lowercase hex>"}`. That id is a `JobStore` id. `GET /v1/jobs/{id}/artifacts` is unchanged.
- Codec string is exactly `video/mp4; codecs="avc1.42E01E,mp4a.40.2"`. Frame rate is 24. Audio is muxed inside each video fragment; `MediaChunk.audio_bytes` is `None`.
- Every fragment is a keyframe. `ClipChunker` finishes `generate()` before it yields the first chunk.
- An `input` message changes a later chunk, never a chunk already sent. `source="clip"` ignores `input`. `source="native"` is the stub, not H3-World.
- CPU tests only, fake runners, no checkpoint loads. No real-GPU gate. Do not modify `third_party/` or `/v1/jobs` handler behavior.
- New Python files start with `# Copyright (c) EfficientMoE.` and `# SPDX-License-Identifier: Apache-2.0`. Ruff line length stays 80.

## Review Focus

- A WebSocket `chunk` must be readable before `jobs/<id>/output.mp4` exists; after `end`, the artifact bytes are that one `write_artifacts` file. Task 6.
- A blocking job and a stream must not overlap `generate()` (`max_active == 1`). Task 6.
- With streaming left at the default, `POST /v1/streams` is 404 and the existing job lifecycle test still passes. Task 5.
- An `input` received after chunk 0 changes chunk 1 only. Task 7.
- `active_cue` uses a half-open interval: time equal to the next cue's `pts` selects the next cue, and time equal to the last cue's end selects nothing. Task 1.

---

## File Structure

Create:

- `omni_infinity/streaming/__init__.py` — re-exports the public names below.
- `omni_infinity/streaming/chunks.py` — `MediaChunk`, `ActionCue`, `StreamRequest`, `StreamInput`, `ChunkSource`, `active_cue`.
- `omni_infinity/streaming/fragment.py` — `CODEC`, `MediaFragment`, `fragment_clip`.
- `omni_infinity/streaming/clip.py` — `ClipChunker`.
- `omni_infinity/streaming/native.py` — `NativeChunker`.
- `omni_infinity/serve/stream.py` — `SessionLimit`, `StreamService`.
- `omni_infinity/client/__init__.py` — package marker.
- `omni_infinity/client/player.py` — message decode, key map, prompt list.
- `omni_infinity/serve/webui/index.html` — page shell.
- `omni_infinity/serve/webui/player.js` — MSE, cue highlight, keys.
- `examples/stream_play.py` — CLI.
- `tests/test_streaming_chunks.py`
- `tests/test_streaming_fragment.py`
- `tests/test_streaming_clip.py`
- `tests/test_streaming_native.py`
- `tests/test_stream_api.py`
- `tests/test_stream_player.py`

Modify:

- `omni_infinity/serve/app.py` — four settings, route registration.
- `omni_infinity/serve/service.py` — extract `validate_request` and `decode_inputs` so the stream path does not queue a second generation. `submit` keeps its current behavior.
- `README.md` — one section after Job-serving API.

`setuptools` already includes `omni_infinity*`. Do not add a package list entry.

---

### Task 1: Chunk types and cue selection

**Files:**
- Create: `omni_infinity/streaming/__init__.py`
- Create: `omni_infinity/streaming/chunks.py`
- Test: `tests/test_streaming_chunks.py`

**Interfaces:**
- Consumes: `GenerationRequest` in `omni_infinity/serve/models.py`.
- Produces:
  - `MediaChunk` frozen dataclass: `index: int`, `pts: float`, `duration: float`, `keyframe: bool`, `video_bytes: bytes | None`, `audio_bytes: bytes | None`, `prompt: str`, `instruction: str | None = None`, `action: str | None = None`, `done: bool = False`.
  - `ActionCue` pydantic model: `t: float` (`ge=0`), `action: str` (length 1..64), `instruction: str` (length 1..2000).
  - `StreamRequest(GenerationRequest)` with `action_script: list[ActionCue]` default `[]` max length 64, and `source: Literal["clip", "native"] = "clip"`.
  - `StreamInput` pydantic model: `type: Literal["input"] = "input"`, `key: str | None = None`, `down: bool = True`, `action: str | None = None`, `prompt: str | None = None`.
  - `ChunkSource` protocol with `iter_chunks(self, request: StreamRequest, *, step_callback: Callable[[int, int], None] | None = None, image: Any = None, last_image: Any = None) -> Iterator[MediaChunk]`.
  - `active_cue(cues: Sequence[MediaChunk], time: float) -> MediaChunk | None`. Interval is `[pts, pts + duration)`.

- [ ] **Step 1: Write the failing test**

```python
def test_active_cue_is_half_open_and_empty_is_none():
    cues = [
        MediaChunk(0, 0.0, 1.0, True, None, None, "a"),
        MediaChunk(1, 1.0, 1.0, True, None, None, "b"),
    ]
    assert active_cue([], 0.0) is None
    assert active_cue(cues, -0.1) is None
    assert active_cue(cues, 0.0).prompt == "a"
    assert active_cue(cues, 0.999).prompt == "a"
    assert active_cue(cues, 1.0).prompt == "b"
    assert active_cue(cues, 2.0) is None


def test_stream_request_defaults_and_rejects_a_long_script():
    request = StreamRequest(type="fl2va", prompt="a red ball bouncing")
    assert request.source == "clip"
    assert request.action_script == []
    assert request.resolution == "256p"
    with pytest.raises(ValidationError):
        StreamRequest(
            type="fl2va",
            prompt="p",
            action_script=[
                {"t": 0, "action": "x" * 65, "instruction": "go"}
            ],
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_streaming_chunks.py -v`
Expected: FAIL with import error for `omni_infinity.streaming`

- [ ] **Step 3: Implement the types and `active_cue`**

Export those names from `omni_infinity/streaming/__init__.py`. `active_cue` returns the first cue whose half-open interval contains `time`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_streaming_chunks.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/streaming/__init__.py omni_infinity/streaming/chunks.py tests/test_streaming_chunks.py
git commit -m "feat(streaming): add chunk types and cue selection"
```

---

### Task 2: fMP4 fragmentation

**Files:**
- Create: `omni_infinity/streaming/fragment.py`
- Modify: `omni_infinity/streaming/__init__.py`
- Test: `tests/test_streaming_fragment.py`

**Interfaces:**
- Consumes: nothing from Task 1. Frames are `numpy` float32 `(N, H, W, 3)` in `[0, 1]`. Audio is a float32 torch tensor shaped `(2, samples)`.
- Produces:
  - `CODEC: str = 'video/mp4; codecs="avc1.42E01E,mp4a.40.2"'`
  - `MediaFragment` frozen dataclass: `index: int`, `pts: float`, `duration: float`, `keyframe: bool`, `video_bytes: bytes`.
  - `fragment_clip(frames, audio, sample_rate: int, *, fps: int = 24, chunk_frames: int = 24) -> tuple[bytes, tuple[MediaFragment, ...]]`.

- [ ] **Step 1: Write the failing test**

```python
def test_fragment_clip_keyframes_and_round_trips():
    frames = np.zeros((5, 16, 16, 3), dtype=np.float32)
    frames[:, :, :, 1] = np.linspace(0.0, 1.0, 5)[:, None, None]
    audio = torch.zeros(2, 48000, dtype=torch.float32)
    init, fragments = fragment_clip(
        frames, audio, 48000, fps=24, chunk_frames=2
    )
    assert init.startswith(b"\x00\x00\x00")
    assert len(fragments) == 3
    assert [item.keyframe for item in fragments] == [True, True, True]
    assert fragments[0].pts == 0.0
    assert fragments[1].pts == pytest.approx(2 / 24)
    assert fragments[2].duration == pytest.approx(1 / 24)
    assert all(item.video_bytes.startswith(b"\x00\x00") for item in fragments)
    container = av.open(io.BytesIO(init + b"".join(
        item.video_bytes for item in fragments
    )))
    decoded = sum(1 for _ in container.decode(video=0))
    container.close()
    assert decoded == 5
    assert b"moov" in init
    assert all(b"moof" in item.video_bytes for item in fragments)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_streaming_fragment.py::test_fragment_clip_keyframes_and_round_trips -v`
Expected: FAIL with import error

- [ ] **Step 3: Implement `fragment_clip`**

Encode with PyAV to fragmented MP4: H.264 `avc1` and AAC, `fps=24`, GOP size `chunk_frames`, muxer flags `frag_keyframe+empty_moov+default_base_moof`. Split the file into the init segment (through the first `moov`) and one `moof` fragment per GOP. The last fragment holds the remainder. `pts = index * chunk_frames / fps`. `duration = frames_in_fragment / fps`. Set `CODEC` even though it is not returned; callers import it.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_streaming_fragment.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/streaming/fragment.py omni_infinity/streaming/__init__.py tests/test_streaming_fragment.py
git commit -m "feat(streaming): fragment clips into fMP4"
```

---

### Task 3: ClipChunker

**Files:**
- Create: `omni_infinity/streaming/clip.py`
- Modify: `omni_infinity/streaming/__init__.py`
- Test: `tests/test_streaming_clip.py`

**Interfaces:**
- Consumes: `StreamRequest`, `MediaChunk`, `fragment_clip`, `GenerationResult`.
- Produces: `ClipChunker(runner, *, arch: str, chunk_frames: int = 24)` with `result: GenerationResult | None`, `init: bytes | None`, and `iter_chunks(...)` from `ChunkSource`. `init` is set before the first yield.

- [ ] **Step 1: Write the failing test**

```python
def test_clip_chunker_finishes_generate_before_the_first_yield(artifact_result):
    seen = {}

    class FakeRunner:
        def generate(self, prompt, *, step_callback, **kwargs):
            seen["kwargs"] = kwargs
            step_callback(kwargs["num_inference_steps"], kwargs["num_inference_steps"])
            seen["returned"] = True
            return artifact_result

    request = StreamRequest(
        type="fl2va",
        prompt="a red ball bouncing",
        num_frames=6,
        action_script=[{"t": 0.0, "action": "wait", "instruction": "hold"}],
    )
    chunker = ClipChunker(FakeRunner(), arch="h3-dense", chunk_frames=4)
    iterator = chunker.iter_chunks(request)
    first = next(iterator)
    rest = list(iterator)
    chunks = [first, *rest]
    assert seen["returned"] is True
    assert seen["kwargs"]["num_inference_steps"] == 8
    assert seen["kwargs"]["resolution"] == "256p"
    assert seen["kwargs"]["num_frames"] == 6
    assert chunker.result is artifact_result
    assert chunker.init
    assert first.prompt == "a red ball bouncing"
    assert first.instruction == "hold"
    assert first.action == "wait"
    assert chunks[-1].done is True
    assert all(chunk.keyframe and chunk.audio_bytes is None for chunk in chunks)


def test_clip_chunker_rejects_vdn_below_768p(artifact_result):
    class FakeRunner:
        def generate(self, *args, **kwargs):
            raise AssertionError("generate must not run")

    request = StreamRequest(type="fl2va", prompt="p", resolution="256p")
    chunker = ClipChunker(FakeRunner(), arch="vdn-hybrid", chunk_frames=4)
    with pytest.raises(ValueError, match="768p"):
        next(chunker.iter_chunks(request))
```

`artifact_result` is the same shape as `tests/test_job_api.py`: 6 frames of float32 `(6, 16, 16, 3)` and stereo audio. Copy that fixture into this file. The dense call also receives `seed`, `image`, and `last_image`. The VDN call is not reached here.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_streaming_clip.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Implement `ClipChunker.iter_chunks`**

Call `generate` to completion, assign `self.result`, run `fragment_clip`, and assign `self.init` before the first yield. Yield one `MediaChunk` per fragment. Attach the `ActionCue` whose `t` lies in `[pts, pts+duration)`; if several match, the last one wins. `done=True` only on the last chunk. For `arch="vdn-hybrid"`, require `request.resolution == "768p"` and call `generate(..., num_evaluations=request.num_inference_steps)` instead of `num_inference_steps`. Otherwise use the dense kwargs. Raise `ValueError("vdn-hybrid requires the 768p canvas")` before `generate`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_streaming_clip.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/streaming/clip.py omni_infinity/streaming/__init__.py tests/test_streaming_clip.py
git commit -m "feat(streaming): chunk a finished generate() into fMP4"
```

---

### Task 4: NativeChunker stub

**Files:**
- Create: `omni_infinity/streaming/native.py`
- Modify: `omni_infinity/streaming/__init__.py`
- Test: `tests/test_streaming_native.py`

**Interfaces:**
- Consumes: `StreamRequest`, `StreamInput`, `MediaChunk`, `fragment_clip`.
- Produces: `NativeChunker(*, chunk_frames: int = 2, chunks: int = 2)` with `push_input(incoming: StreamInput) -> None`, `iter_chunks(...)`, `init: bytes | None`, and `result: GenerationResult | None`. `result` is the synthetic frames and audio passed to `fragment_clip`, set before the first yield.

- [ ] **Step 1: Write the failing test**

```python
def test_native_input_changes_only_the_next_chunk():
    request = StreamRequest(type="fl2va", prompt="idle", source="native")
    chunker = NativeChunker(chunk_frames=2, chunks=2)
    iterator = chunker.iter_chunks(request)
    first = next(iterator)
    chunker.push_input(StreamInput(action="forward", prompt="run"))
    second = next(iterator)
    assert first.prompt == "idle"
    assert first.instruction is None
    assert second.prompt == "run"
    assert second.action == "forward"
    assert second.instruction == "forward"
    assert second.done is True
    assert first.video_bytes and second.video_bytes
    assert chunker.result is not None
    assert chunker.init
    with pytest.raises(StopIteration):
        next(iterator)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_streaming_native.py::test_native_input_changes_only_the_next_chunk -v`
Expected: FAIL with import error

- [ ] **Step 3: Implement `NativeChunker`**

Build one synthetic clip of `chunk_frames * chunks` solid frames and silence, store it as `self.result`, and `fragment_clip` it once. Assign `self.init` before the first yield. Do not sleep. Before each yield, apply the newest `StreamInput` since the previous yield: `action` sets both `action` and `instruction`; `prompt` replaces `prompt`. Inputs pushed before the first `next()` apply to chunk 0. There is no model call.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_streaming_native.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/streaming/native.py omni_infinity/streaming/__init__.py tests/test_streaming_native.py
git commit -m "feat(streaming): add the interactive NativeChunker stub"
```

---

### Task 5: Session HTTP, dark by default

**Files:**
- Create: `omni_infinity/serve/stream.py`
- Modify: `omni_infinity/serve/service.py`
- Modify: `omni_infinity/serve/app.py`
- Test: `tests/test_stream_api.py`

**Interfaces:**
- Consumes: `JobService`, `JobStore`, `StreamRequest`, existing `ProfileConflict`, `Ref2VANotImplemented`, `InvalidMedia`.
- Produces:
  - `JobService.validate_request(request: GenerationRequest) -> None`
  - `JobService.decode_inputs(request: GenerationRequest) -> tuple[bytes | None, bytes | None]`
  - `ServerSettings.stream_enabled: bool = False`, `stream_max_sessions: int = 1`, `stream_chunk_frames: int = 24`, `stream_fallback_hls: bool = False`, filled by `from_env`.
  - `SessionLimit(JobServiceError)`.
  - `StreamService(jobs: JobService, *, max_sessions: int, chunk_frames: int, hls: bool)` with `open_session(request: StreamRequest) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
def test_streams_are_absent_unless_enabled(tmp_path):
    settings = ServerSettings(jobs_dir=tmp_path, optimizations=())
    with TestClient(create_app(settings, lambda: object())) as client:
        assert client.post("/v1/streams", json=_body()).status_code == 404


def test_open_session_caps_and_reuses_job_errors(tmp_path):
    settings = ServerSettings(
        jobs_dir=tmp_path, optimizations=(), stream_enabled=True
    )
    with TestClient(create_app(settings, lambda: object())) as client:
        first = client.post("/v1/streams", json=_body())
        assert first.status_code == 202
        assert len(first.json()["stream_id"]) == 32
        second = client.post("/v1/streams", json=_body())
        assert second.status_code == 409
        assert second.json()["detail"] == "stream session limit reached"
        ref = client.post("/v1/streams", json=_body(type="ref2va"))
        assert ref.status_code == 501
        other = client.post(
            "/v1/streams", json=_body(model_arch="vdn-hybrid")
        )
        assert other.status_code == 409
        assert other.json()["detail"] == (
            "request profile does not match the loaded server profile"
        )
        assert len(list(tmp_path.iterdir())) == 1
```

`_body(**overrides)` returns a FL2VA JSON object with `prompt="a red ball bouncing"` and `optimizations: []`. The cap test needs the first session to stay open; do not connect a socket in this task. `validate_request` runs before the session cap, so the ref2va and profile calls return 501 and 409 while the first session is still held.

Add `test_stream_enabled_env_parses_bools` for `ServerSettings.from_env`: unset is false, `OMNI_STREAM_ENABLED=true` is true, `OMNI_STREAM_ENABLED=0` is false, `OMNI_STREAM_CHUNK_FRAMES=12` is 12.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_api.py -v`
Expected: FAIL because the route and settings fields do not exist

- [ ] **Step 3: Implement session open without starting generation**

Move the ref2va check, profile check, and image decode out of `JobService.submit` into `validate_request` and `decode_inputs`. `submit` calls them and then creates and enqueues as it does now. `open_session` calls `validate_request`, then `decode_inputs`, then rejects when the live session count is `>= max_sessions`, and only then calls `store.create`. stores a `GenerationRequest` (drop `action_script` and `source` from the job record), saves images the way `submit` does, and returns the id. Count that session until `StreamService.finish(stream_id)` which this task also adds so tests can release the slot. Register `POST /v1/streams` only when `settings.stream_enabled` is true. Map `SessionLimit` and `ProfileConflict` to 409, `Ref2VANotImplemented` to 501, `InvalidMedia` to 422.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stream_api.py tests/test_job_api.py -v`
Expected: PASS, including the existing job lifecycle tests

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/serve/stream.py omni_infinity/serve/service.py omni_infinity/serve/app.py tests/test_stream_api.py
git commit -m "feat(serve): open stream sessions without exposing them by default"
```

---

### Task 6: WebSocket playback and shared worker

**Files:**
- Modify: `omni_infinity/serve/stream.py`
- Modify: `omni_infinity/serve/app.py`
- Test: `tests/test_stream_api.py`

**Interfaces:**
- Consumes: `ClipChunker`, `NativeChunker`, `CODEC`, `JobService.executor`, `write_artifacts`, `ChunkSource.iter_chunks`.
- Produces: websocket `WS /v1/streams/{id}/ws` and in-order JSON messages:
  - `{"type": "init", "codec": CODEC, "init_b64": "<standard base64>", "action_script": [{"t", "action", "instruction"}]}`
  - `{"type": "chunk", "index", "pts", "duration", "keyframe", "video_b64", "audio_b64": null, "prompt", "instruction", "action", "done"}`
  - `{"type": "end", "artifact_url": "/v1/jobs/<id>/artifacts"}`
  - `{"type": "error", "detail": "<text>"}` then close

- [ ] **Step 1: Write the failing tests**

```python
def test_socket_sends_first_chunk_before_the_artifact(tmp_path, artifact_result):
    # FakeRunner.generate records entry on a threading.Event and returns
    # artifact_result. POST /v1/streams, then websocket_connect.
    # After receive_json() returns a chunk, output.mp4 does not exist.
    # After the end message, GET /v1/jobs/{id}/artifacts is 200 and its
    # body equals jobs/<id>/output.mp4. av.open on those bytes shows
    # exactly one video stream and one audio stream.
    # init is the first message and its codec equals CODEC.
    # The last chunk has done True. Prompt equals the request prompt.


def test_socket_does_not_pull_the_next_chunk_early(tmp_path):
    # ChunkSource.iter_chunks yields two MediaChunk values.
    # The second next() sets a threading.Event.
    # receive_json of the first chunk happens while that event is unset.


def test_stream_waits_for_the_job_worker(tmp_path, artifact_result):
    # One FakeRunner whose generate increments max_active.
    # submit a /v1/jobs request that blocks inside generate.
    # Connect the stream socket on another thread.
    # Assert max_active stays 1 until the job is released.
```

Use `stream_chunk_frames=4` so the 6-frame fixture yields two fragments. Inject the runner with `create_app(settings, lambda: runner)`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_api.py::test_socket_sends_first_chunk_before_the_artifact -v`
Expected: FAIL with 404 on the websocket

- [ ] **Step 3: Implement the socket on the job executor**

Run the chunk iterator on `jobs.executor`. The socket thread and that worker share two `queue.Queue(maxsize=1)` objects: the socket puts a token to ask for the next chunk, and the worker blocks on that token before every `next()`. The first `next()` is what runs `generate()` and sets `chunker.init`. After the first result, send `init` (include `action_script`) and then the chunk. Ask for another chunk only after that send returns. On `StopIteration`, call `write_artifacts(chunker.result, store.job_dir(id), artifact_url=f"/v1/jobs/{id}/artifacts")`, transition to succeeded, and send `end`. On failure, unlink `output.mp4` and `output.wav` if present, transition to failed, and send `error`. Call `finish` in a `finally`, including disconnect. Do not prefetch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stream_api.py tests/test_job_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/serve/stream.py omni_infinity/serve/app.py tests/test_stream_api.py
git commit -m "feat(serve): stream fMP4 chunks on one GPU worker"
```

---

### Task 7: WebSocket input channel

**Files:**
- Modify: `omni_infinity/serve/stream.py`
- Test: `tests/test_stream_api.py`

**Interfaces:**
- Consumes: `StreamInput`, `NativeChunker.push_input`.
- Produces: the socket reads `{"type": "input", "key", "down", "action", "prompt"}`. For `source="native"` only, after sending a chunk and before asking for the next one, wait up to that chunk's `duration` seconds for an input. `source="clip"` does not wait and ignores input.

- [ ] **Step 1: Write the failing test**

```python
def test_input_changes_the_next_native_chunk_only(tmp_path):
    # POST source=native, stream_chunk_frames=2.
    # Receive init and chunk 0. Send
    # {"type":"input","action":"forward","prompt":"run","key":"ArrowUp","down":true}.
    # Chunk 0 prompt stays "idle". Chunk 1 prompt is "run",
    # action and instruction are "forward".
    # A clip session that receives the same input keeps the original prompt
    # on every chunk.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_api.py::test_input_changes_the_next_native_chunk_only -v`
Expected: FAIL because chunk 1 is still the original prompt

- [ ] **Step 3: Implement input forwarding**

Make the socket endpoint `async def`. After each native chunk is sent, `await` the next client message with a timeout of that chunk's `duration` (use `anyio.fail_after`). On `input`, call `push_input` and then ask the worker for the next chunk. On timeout, ask immediately. `source="clip"` never enters that wait. A malformed native input sends `error` and still continues playback.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/serve/stream.py tests/test_stream_api.py
git commit -m "feat(serve): accept stream input for the native stub"
```

---

### Task 8: HLS fallback

**Files:**
- Modify: `omni_infinity/serve/stream.py`
- Modify: `omni_infinity/serve/app.py`
- Test: `tests/test_stream_api.py`

**Interfaces:**
- Consumes: the session's init bytes and fragment list from Task 6.
- Produces, only when `stream_fallback_hls` is true:
  - `GET /v1/streams/{id}/playlist.m3u8`
  - `GET /v1/streams/{id}/init.mp4`
  - `GET /v1/streams/{id}/seg/{n}.m4s`
  - `GET /v1/streams/{id}/prompts.json` as `{"cues": [{"index", "pts", "duration", "prompt", "instruction", "action"}]}`

- [ ] **Step 1: Write the failing test**

```python
def test_hls_stays_404_until_the_fallback_flag(tmp_path, artifact_result):
    # stream_enabled true, stream_fallback_hls false:
    # GET playlist.m3u8 is 404.
    # With the flag true, playlist before the socket is 409.
    # After the end message, playlist contains #EXT-X-MAP:URI="init.mp4",
    # seg/0.m4s, and #EXT-X-ENDLIST.
    # prompts.json cues[0].prompt equals the request prompt.
    # seg/0.m4s body equals the first fragment video_bytes.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_api.py::test_hls_stays_404_until_the_fallback_flag -v`
Expected: FAIL with 404 while the flag is on, or missing playlist text

- [ ] **Step 3: Implement the four GET handlers**

Store init and fragments on the session as the socket sends them. Return 409 with detail `stream media is not ready` until init exists. Append `#EXT-X-ENDLIST` only after `end`. Do not register the routes when the flag is off.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/serve/stream.py omni_infinity/serve/app.py tests/test_stream_api.py
git commit -m "feat(serve): add the optional fMP4 playlist fallback"
```

---

### Task 9: Local player

**Files:**
- Create: `omni_infinity/client/__init__.py`
- Create: `omni_infinity/client/player.py`
- Create: `examples/stream_play.py`
- Test: `tests/test_stream_player.py`

**Interfaces:**
- Consumes: the JSON message shapes from Task 6 and `StreamInput`.
- Produces:
  - `KEY_ACTIONS = {"ArrowUp": "forward", "ArrowDown": "back", "ArrowLeft": "left", "ArrowRight": "right", " ": "jump"}`
  - `key_to_input(key: str, down: bool) -> StreamInput | None` (`None` for unmapped keys).
  - `prompts_from_messages(messages: list[dict]) -> list[tuple[float, str, str | None]]` using each chunk's `pts`, `prompt`, and `instruction`.
  - `decode_fragmented(messages: list[dict]) -> int` concatenates `init_b64` plus each `video_b64` and returns the decoded video frame count.
  - `examples/stream_play.py` arguments `--url` (required), `--player {av,ffplay,mpv}` default `av`, `--interactive`.

- [ ] **Step 1: Write the failing test**

```python
def test_key_map_and_prompt_timeline():
    assert key_to_input("ArrowUp", True).action == "forward"
    assert key_to_input("x", True) is None
    messages = [
        {"type": "init", "codec": CODEC, "init_b64": base64.b64encode(init).decode()},
        {"type": "chunk", "pts": 0.0, "prompt": "idle", "instruction": None,
         "video_b64": base64.b64encode(fragments[0].video_bytes).decode(),
         "audio_b64": None, "done": False},
        {"type": "chunk", "pts": fragments[1].pts, "prompt": "run",
         "instruction": "forward",
         "video_b64": base64.b64encode(fragments[1].video_bytes).decode(),
         "audio_b64": None, "done": True},
    ]
    assert prompts_from_messages(messages) == [
        (0.0, "idle", None),
        (fragments[1].pts, "run", "forward"),
    ]
    assert decode_fragmented(messages) == 4
```

Build `init, fragments` with `fragment_clip` on 4 frames and `chunk_frames=2`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_player.py::test_key_map_and_prompt_timeline -v`
Expected: FAIL with import error

- [ ] **Step 3: Implement the player helpers and the example CLI**

`stream_play.py` connects to `--url`, prints `f"{pts:.3f} {prompt} {instruction or ''}"` for each chunk, and in `--interactive` mode sends `key_to_input` for stdin keys. `--player av` decodes through `decode_fragmented`'s same byte assembly. `--player ffplay` and `mpv` pipe those bytes to the executable. Tests must not spawn a player.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_player.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/client examples/stream_play.py tests/test_stream_player.py
git commit -m "feat(client): play stream chunks and map keys to input"
```

---

### Task 10: Browser player

**Files:**
- Create: `omni_infinity/serve/webui/index.html`
- Create: `omni_infinity/serve/webui/player.js`
- Modify: `omni_infinity/serve/app.py`
- Test: `tests/test_stream_api.py`

**Interfaces:**
- Consumes: `active_cue` semantics and `KEY_ACTIONS` from Task 9.
- Produces: `GET /` and `GET /player.js` only when `stream_enabled` is true. The page has a `<video>` element and a prompt panel element with id `prompt-panel`.

- [ ] **Step 1: Write the failing test**

```python
def test_webui_is_served_only_when_streaming_is_enabled(tmp_path):
    dark = ServerSettings(jobs_dir=tmp_path, optimizations=())
    with TestClient(create_app(dark, lambda: object())) as client:
        assert client.get("/").status_code == 404
    live = ServerSettings(
        jobs_dir=tmp_path, optimizations=(), stream_enabled=True
    )
    with TestClient(create_app(live, lambda: object())) as client:
        page = client.get("/")
        script = client.get("/player.js")
    assert page.status_code == 200
    assert 'id="prompt-panel"' in page.text
    assert "<video" in page.text
    body = script.text
    for needle in ("SourceBuffer", "currentTime", "keydown", "keyup",
                   "forward", "activeCue"):
        assert needle in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_api.py::test_webui_is_served_only_when_streaming_is_enabled -v`
Expected: FAIL with 404 on the enabled app

- [ ] **Step 3: Implement the static page**

`player.js` defines `activeCue(cues, time)` with the same half-open rule as `active_cue`. On `init`, render every `action_script` entry into `#prompt-panel`. Append `init` then each `chunk` into one MSE `SourceBuffer` for the `codec` string, and mark the entry whose interval contains `video.currentTime`. `keydown` and `keyup` send `input` using the same five key names as `KEY_ACTIONS`. Serve the two files with `FileResponse`. No build step and no new package.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_api.py::test_webui_is_served_only_when_streaming_is_enabled -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add omni_infinity/serve/webui omni_infinity/serve/app.py tests/test_stream_api.py
git commit -m "feat(serve): add the static MSE player"
```

---

### Task 11: README

**Files:**
- Modify: `README.md`
- Test: `tests/test_stream_api.py`

**Interfaces:**
- Consumes: the routes and settings from Tasks 5–10.
- Produces: a `## Streaming playback` section immediately after the Job-serving API section and before `## Deferred: shared`.

- [ ] **Step 1: Write the failing test**

```python
def test_readme_documents_pseudo_streaming():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "ClipChunker does not overlap generation with playback." in readme
    for needle in (
        "OMNI_STREAM_ENABLED",
        "POST /v1/streams",
        "WS /v1/streams/{id}/ws",
        "OMNI_STREAM_FALLBACK_HLS",
    ):
        assert needle in readme
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_api.py::test_readme_documents_pseudo_streaming -v`
Expected: FAIL on the missing sentence

- [ ] **Step 3: Document the contract**

State that the flag defaults off, list the four settings, show a `POST /v1/streams` body that is a job body plus optional `action_script` and `source`, name the `init`, `chunk`, `end`, and `input` messages, and point artifact download at `GET /v1/jobs/{id}/artifacts`. Include the sentence `ClipChunker does not overlap generation with playback.` exactly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stream_api.py tests/test_streaming_chunks.py tests/test_streaming_fragment.py tests/test_streaming_clip.py tests/test_streaming_native.py tests/test_stream_player.py tests/test_job_api.py -v`
Expected: PASS

Run: `ruff check omni_infinity/streaming omni_infinity/client omni_infinity/serve tests/test_streaming_chunks.py tests/test_streaming_fragment.py tests/test_streaming_clip.py tests/test_streaming_native.py tests/test_stream_api.py tests/test_stream_player.py && ruff format --check omni_infinity/streaming omni_infinity/client omni_infinity/serve tests/test_streaming_chunks.py tests/test_streaming_fragment.py tests/test_streaming_clip.py tests/test_streaming_native.py tests/test_stream_api.py tests/test_stream_player.py`
Expected: exit 0

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_stream_api.py
git commit -m "docs: document streaming playback next to the job API"
```
