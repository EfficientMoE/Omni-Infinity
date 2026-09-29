# Streaming chunk playback

Issue #14 adds a model-agnostic streaming layer beside the Task-4 job API.
`GET /v1/jobs/{id}/artifacts` stays a finished-file download: it returns HTTP
409 until denoising, decode, and mux complete. A 5–15 s H3 clip therefore
shows no frames until the whole MP4 exists, and progress is only a step
counter. There is also no player, and no place to show the prompt or
per-interval instruction next to the video.

This document is the implementation design and the deliverable list. It does
not land the feature. The H3-World model itself (LoRA, directed attention,
`h3-world` registry entry, native chunk production) is a follow-up that this
layer is shaped to accept.

## What stays

The job API in `omni_infinity/serve/` is unchanged:

- `ServerSettings` and `create_app` still load one registry profile for the
  process lifetime. A request whose `model_arch` or optimization list differs
  from that profile returns HTTP 409.
- `JobService` keeps a `ThreadPoolExecutor(max_workers=1)`. A stream and a job
  share that worker, so the GPU never runs two generations at once.
- `GenerationRequest` remains the generation schema. `POST /v1/streams` uses
  the same fields plus an optional `action_script`.
- `write_artifacts` still publishes `jobs/<uuid>/output.wav` with SoundFile
  and `jobs/<uuid>/output.mp4` with
  `diffusers.utils.export_utils.encode_video`. A finished stream is also a
  normal job artifact, so download-the-file keeps working and stays
  comparable to `/v1/jobs`.
- `type="ref2va"` stays HTTP 501.
- `av`, FastAPI, and Uvicorn already ship in the `serve` extra. Add
  `websockets` to both `serve` and the mirrored `dev` dependency list for the
  live Uvicorn transport and local player. Fragmented MP4 is new code on
  `av`; it does not replace the finished-file muxer.

## Shape

```text
player (local or browser)
        |
        |  WS /v1/streams/{id}/ws
        v
  StreamService  ---->  existing single GPU worker
        |                      |
        |                      +-- ClipChunker --> ReferenceRunner / VdnRunner
        |                      +-- NativeChunker   (interface only)
        |
        +-- JobStore artifact  jobs/<uuid>/output.mp4
```

`ClipChunker` is the Phase-1 adapter. It calls the existing one-shot
`generate()` (FL2VA through `ReferenceRunner`, or `VdnRunner`), forwards the
current `step_callback` for denoising progress, then splits the decoded frames
and stereo audio into time-ordered fMP4 fragments. Playback can start before
the client would have been allowed to download the finished MP4. Denoising
itself does not change. This is pseudo-streaming.

`NativeChunker` implements the same `ChunkSource` protocol but is not backed
by a model in this issue. A test stub emits scripted chunks and, in Phase 2,
consumes client `input` messages. Real autoregressive latent intervals, each
tied to a language instruction, land with H3-World.

## Chunk model

New package `omni_infinity/streaming/`.

`MediaChunk` fields:

- `index`, `pts`, `duration`, `keyframe`
- `video_bytes`: one fMP4 media fragment, or `None` before media exists
- `audio_bytes`: optional AAC fragment for the same interval
- `prompt`: the active prompt string
- `instruction`, `action`: optional, `None` for a static FL2VA prompt
- `done`: true on the last chunk of the session

`ChunkSource` is a protocol: `iter_chunks(request) -> Iterator[MediaChunk]`.
The iterator runs on the existing serialized GPU worker, not on the request
thread.

Adapters:

- `ClipChunker` wraps `ReferenceRunner.generate()` / `VdnRunner.generate()`.
  After generation it fragments video and audio with `av` (H.264 / AAC, same
  codecs as the job muxer) into an initialization segment plus media
  fragments. Every chunk carries the static request prompt. `step_callback`
  still updates job progress while denoising runs, before the first fragment.
- `NativeChunker` is the protocol implementation point for a later runner
  that yields one decoded fMP4 fragment per autoregressive interval, with
  that interval's instruction. This issue ships the class and a deterministic
  stub, not the H3-World forward.

Fragmentation lives in one helper used by both adapters: init segment, then
media fragments sized by `OMNI_STREAM_CHUNK_FRAMES`. The first fragment of a
session is a keyframe so a client can start, or resync, without a prior
fragment.

## HTTP and WebSocket

Extend `omni_infinity/serve/`. Routes exist only when `OMNI_STREAM_ENABLED`
is set. Default is off, so current servers keep today's surface: no `/`, no
`/v1/streams`.

`POST /v1/streams` creates a session and a corresponding `JobRecord` through
`JobStore.create`; its generated 32-hex job id is also the `stream_id`. The
record follows the existing `queued -> running -> succeeded|failed|cancelled`
state machine, and progress callbacks run only after it reaches `running`.
Body is `GenerationRequest` plus optional `action_script` (ordered
`{t, action, instruction}` entries for scripted Phase 1). Response is
`{stream_id}`. Profile mismatch is HTTP 409, invalid media is HTTP 422,
`ref2va` is HTTP 501, matching jobs. A second session while one is active
returns HTTP 409 when `OMNI_STREAM_MAX_SESSIONS` is 1 (the default, matching
the single worker).

`WS /v1/streams/{id}/ws` is the primary transport.

Server to client:

- `init`: fMP4 initialization segment and the MSE codec string
  (`video/mp4; codecs="avc1....,mp4a...."`).
- `chunk`: one `MediaChunk` (fragment bytes, `pts`, `duration`, `keyframe`,
  `prompt`, optional `instruction` / `action`, `done`).
- `end`: session finished; the muxed MP4 is published as a job artifact.

Client to server, Phase 2:

- `input`: a key or action, or a prompt update. Phase 1 accepts the socket
  and ignores these messages. `NativeChunker` consumes them when the stub
  (and later H3-World) is selected.

The GPU worker writes fragments to a bounded per-session queue. A separate
async WebSocket sender drains that queue, so a slow network client cannot
block the sole GPU executor thread and prevent queued jobs from running.
`ClipChunker` materializes the request-bounded clip before delivery. The
socket hand-off queue is bounded and applies backpressure without dropping
fragments, so slow clients receive every fragment in order while the GPU
worker remains free for the next job.
The MSE client can therefore resync without a custom timeline.

Optional passive fallback, only if `OMNI_STREAM_FALLBACK_HLS` is set:

- `GET /v1/streams/{id}/playlist.m3u8`
- `GET /v1/streams/{id}/seg/{n}.m4s`
- `GET /v1/streams/{id}/prompts.json` (sidecar timeline of prompt /
  instruction per fragment)

That path is for a plain `<video>` element, hls.js, or a CDN. It is not the
interactive channel. Timed metadata inside the media is not required.

When the stream finishes, `StreamService` calls `write_artifacts` in the
directory of the existing `JobRecord`, then transitions that record to
`succeeded` with the artifact metadata the job API already returns.
`GET /v1/jobs/{id}/artifacts` on the shared stream/job id returns the MP4.
Existing job routes are not rewritten.

## Settings

Added on `ServerSettings.from_env`:

| Variable | Default | Role |
| --- | --- | --- |
| `OMNI_STREAM_ENABLED` | off | Register stream routes and `/` |
| `OMNI_STREAM_MAX_SESSIONS` | 1 | Cap concurrent streams |
| `OMNI_STREAM_CHUNK_FRAMES` | 24 | Frames per fMP4 fragment (about 1 s at 24 fps) |
| `OMNI_STREAM_QUEUE_CHUNKS` | 8 | Bounded async sender queue capacity |
| `OMNI_STREAM_FALLBACK_HLS` | off | Publish the playlist / segment / prompt sidecar |

`OMNI_WORKERS` stays 1.

## Clients

Local player: `omni_infinity/client/` plus `examples/stream_play.py`. A
WebSocket client appends fMP4 fragments and plays them with PyAV, or pipes
the same byte stream to `ffplay` / `mpv`. It prints or overlays the active
prompt and instruction. Phase 2 reads the keyboard and sends `input`. No new
heavy dependency is needed beyond the lightweight `websockets` runtime added
to `serve`; `av` is already present.

Browser player: `omni_infinity/serve/webui/`, static files, no build step.
Served at `/` when streaming is enabled. A `<video>` element plus an MSE
`SourceBuffer` consumes `init` and `chunk`. A prompt panel beside the video
highlights the entry whose `[pts, pts+duration)` contains
`video.currentTime`. Phase 1 can show the full `action_script` with the live
segment highlighted. Phase 2 maps `keydown` / `keyup` to `input` messages.

## Deliverables

### Phase 1 — passive playback

- `omni_infinity/streaming/`: `MediaChunk`, `ChunkSource`, `ClipChunker`,
  fMP4 helper, `NativeChunker` interface.
- Stream routes on the existing FastAPI app, sharing the one GPU worker and
  the `JobStore` artifact layout.
- `examples/stream_play.py`: WebSocket client, PyAV or `ffplay` / `mpv`,
  prompt printed beside playback.
- `omni_infinity/serve/webui/`: static page, MSE playback, prompt panel.
- CPU tests with a fake chunk source: session create, WS `init` / `chunk` /
  `end`, HTTP 409 on profile mismatch and on session overflow, jobs API
  behavior unchanged, and equivalent video/audio streams in the stream and
  job-path artifacts for the same frames and audio.
- A live-socket integration test launches Uvicorn and connects with
  `websockets`, proving the installed `serve` extra can complete a real
  WebSocket upgrade rather than only an in-process ASGI test.
- README section next to Job-serving API: enable flag, `POST /v1/streams`,
  WebSocket message types, artifact download, and the statement that
  `ClipChunker` does not overlap generation with playback.

Acceptance: a `256p` / 120-frame FL2VA stream emits its first media fragment
before `GET /v1/jobs/{id}/artifacts` would have returned 200 for that same
clip. The player shows the prompt beside the video. The downloaded MP4
matches the existing job artifact for the same generation result.

### Phase 2 — interactive control

Still this issue, still without the H3-World model.

- The same WebSocket accepts `input`.
- The `NativeChunker` stub changes the next chunk from a key or action.
- The browser captures `keydown` / `keyup` and highlights the live
  instruction.

Acceptance: with the stub, an `input` changes the next streamed chunk inside
a bounded per-chunk latency (one fragment interval, not a full clip). The
prompt panel shows that chunk's instruction.

## Tests

CPU CI already installs `.[dev]`; its dependency list mirrors `serve` and
must add `websockets` with it. New tests follow `tests/test_job_api.py`:
inject a fake runner or fake `ChunkSource`, drive the app with `httpx` and a
WebSocket test client, and do not load weights. One CPU test also starts a
live Uvicorn server and connects through `websockets` to cover the real
upgrade path. A real FL2VA stream gate, if added, stays `pytest.mark.skipif`
on CUDA plus an explicit env flag, same pattern as
`test_real_fl2va_job_api_returns_muxed_mp4`.

Default `OMNI_STREAM_ENABLED` unset must leave `POST /v1/streams` unregistered
so existing job tests keep passing without new settings.

## Out of scope

- H3-World weights, LoRA, directed attention, and an `h3-world` registry
  entry.
- Auth and multi-tenant streaming.
- Adaptive bitrate and sparse attention.
- Replacing `/v1/jobs` or changing FP8. FP8 remains the existing opt-in
  memory tradeoff.

## Alternatives not taken

- Segmented HTTP alone (fMP4 / LL-HLS) is the optional fallback, not the
  primary transport, because it has no client-to-server input channel.
- Streaming raw latents or RGB is rejected: the browser cannot run the VAE,
  and raw RGB is the wrong bandwidth and the wrong player model.
- Chunked transfer of a growing MP4 on `/v1/jobs` is rejected: progressive
  MP4 is not reliably appendable mid-mux, and it carries no per-chunk prompt
  or upstream input.
