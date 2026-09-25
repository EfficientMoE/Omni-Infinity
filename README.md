# Omni-Infinity

Cost-effective single-server inference for **dense omni-modal generative
models** — starting with **MiniMax-H3-Base** — on memory-constrained GPUs.

Sibling project of [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity)
(same philosophy: run models bigger than your GPU by exploiting structure +
fast storage), scoped by the RFC in
[MoE-Infinity#222](https://github.com/EfficientMoE/MoE-Infinity/issues/222).
Where MoE-Infinity's lever is routed-expert offloading, H3-class models have
no routed experts; Omni-Infinity's levers are:

1. **Component-level offloading** — the H3-Base pipeline spans a 33B
   transformer, a Qwen3-VL-32B text encoder, and two VAEs that never need to
   be co-resident: encode text → release encoder → denoise → decode latents,
   swapping components through the GPU with prefetch overlap.
2. **AdaLN branch caching** — ~13B of H3's 33B params are AdaLN branches
   that are cacheable at inference; keep them in host memory/SSD and
   materialize per-modality on demand.
3. **Denoising-step scheduling** — overlap weight transfers with denoising
   compute across steps; step-granular checkpointing for preemption.
4. **Latent/VRAM budgeting** — paged latent buffers instead of paged KV.
5. **Job-oriented serving** — generation jobs with progress/webhooks, not
   OpenAI chat completions.

## Storage

Omni-Infinity consumes [moe-store](https://github.com/EfficientMoE/moe-store)
(>= 0.2.1) for checkpoint conversion and the arch-aware v2 on-disk store.
`moe-store convert MiniMaxAI/MiniMax-H3 <store>` ingests the full modular
pipeline: per-block non-AdaLN groups, per-block AdaLN bundle groups, and one
group per VAE/encoder shard. The transfer engine is Omni-Infinity's own
step-synchronous loop, not shared with MoE-Infinity.

## Scope (v0.1)

- **Target:** MiniMax-H3-Base only (`MiniMaxAI/MiniMax-H3`, FL2VA + Ref2VA,
  768p). H3-Context-IR and H3-Regenerate-2K are hosted/API-only and out of
  local scope until (if ever) their weights are released.
- **Memory story:** component offload + AdaLN caching + FP8 weights (FP4
  where kernels allow).
- **Non-goals:** multimodal understanding on MoE backbones (that is
  MoE-Infinity, [#221](https://github.com/EfficientMoE/MoE-Infinity/issues/221));
  sparse-attention inference (the H3 open release supports full attention
  only).

## Model archs & optimizations

Two orthogonal category axes (`omni_infinity/registry.py`) describe every
supported configuration; the smoke CLIs and the ablation harness
(`benchmarks/ablation_vdn.py`) resolve their flags through the registry.

| model-arch | runner | checkpoint |
|---|---|---|
| `h3-dense` | `omni_infinity.runner.ReferenceRunner` | `MiniMaxAI/MiniMax-H3` |
| `vdn-hybrid` | `omni_infinity.arch.vdn.VdnRunner` | `OpenVDN/vdn-minimax-h3` (Hub repo-id only — see the cross-repo `trust_remote_code` note in [docs/repro_vdn.md](docs/repro_vdn.md)) |

| optimization | h3-dense | vdn-hybrid |
|---|:---:|:---:|
| `adaln-host-cache` (moe-store AdaLN branch cache) | ✓ | — |
| `fp8` (weight-only FP8 on wide Linears) | ✓ | ✓ |
| `block-stream` (transformer block_level group offload) | ✓ | ✓ |
| `text-encoder-stream` (Qwen3-VL leaf_level streaming) | ✓ | ✓ |

`vdn-hybrid` is **VDN-Minimax-H3** ("Video DeltaNet",
[OpenVDN/vdn-minimax-h3](https://github.com/OpenVDN/vdn-minimax-h3),
pinned at `third_party/vdn-minimax-h3`): a frame-wise linear-attention
branch + window softmax + two merged LoRA adapters on the frozen H3
backbone. Reproduction on RTX PRO 6000 Blackwell (sm120):
[docs/repro_vdn.md](docs/repro_vdn.md) — dense→hybrid+fp8 **2.15×**
per-NFE, bitwise golden parity (`tests/test_vdn_parity.py`), block
streaming ~20 GiB peak. Ablation study:
[docs/ablation_vdn.md](docs/ablation_vdn.md). Speedup attribution study:
[docs/attribution_vdn.md](docs/attribution_vdn.md). Tracking:
[#10](https://github.com/EfficientMoE/Omni-Infinity/issues/10).

## Status

Bootstrap in progress — see the
[task list](https://github.com/EfficientMoE/MoE-Infinity/issues/222):

- [x] Task 0 — moe-store multi-component H3 converter with AdaLN bundle
      groups ([moe-store#9](https://github.com/EfficientMoE/moe-store/pull/9),
      released in v0.2.1)
- [x] Task 1 — repo skeleton + reference-parity harness: runner API
      (`omni_infinity/runner.py`), smoke CLI (`examples/fl2va_smoke.py`),
      and the bitwise parity gate (`tests/test_reference_parity.py`) —
      validated on real H3-Base weights: a fresh generation reproduces the
      committed golden latents bitwise
      ([#1](https://github.com/EfficientMoE/Omni-Infinity/issues/1))
- [~] Task 2 — component offload + AdaLN caching + FP8 on a single 24 GB GPU
      ([#2](https://github.com/EfficientMoE/Omni-Infinity/issues/2)):
  - [x] Store-sourced components (`StoreComponentSource`, one-read group
        fetches) reproduce the goldens bitwise (inc 1, 2, 2b)
  - [x] Host-resident AdaLN branch cache — 66.28 → 40.26 GB GPU-resident,
        bitwise parity (inc 3)
  - [x] FP8 weight path (opt-in) + offline QA harness — validated 40.26 →
        20.19 GB, but H3 is not FP8-native so it can't meet `rtol=2e-2`
        (~16-25% latent error); memory tradeoff only (inc 4)
  - [x] bf16 block-streaming — native `enable_group_offload(block_level)`
        streams the transformer blocks (attn/ff) with prefetch; the AdaLN
        host cache is untouched. Reproduces the goldens **bitwise**
        (`rms_rel=0.0000`) with a trivial transformer weight footprint (inc 5)
  - [x] Text-encoder streaming — the 64-layer Qwen3-VL encoder is
        leaf-level group-offloaded (bf16), so its layers stream through the GPU
        instead of the whole 64 GB staying resident. Reproduces the goldens
        **bitwise** with the transformer block-streamed too (inc 6). `leaf_level`
        (not `block_level`) is required so `embed_tokens` self-onloads.
  - [x] Whole-pipeline ≤22 GiB gate — the full encode+denoise+decode window
        peaks at **9.96 GiB** under the emulated 22 GiB offload envelope, with
        `rms_rel=0.0000`, elementwise `allclose(rtol=2e-2)`, and 38.1 s
        wall-clock (inc 7)
  - [x] Fused block-scaled FP8 weight-only kernel — vendored+hardened
        Triton GEMM behind an `omni_infinity/kernels/` facade (op-centric,
        pure-torch reference fallback). `ScaledFp8Linear` and the AdaLN FP8
        branch dequantize *inside* the GEMM (no bf16 weight
        materialization); block-wise 128×128 scaling. Benchmark: FP8
        weight bytes halved and lower peak memory on large Linears (e.g.
        N=28672: 3330→2600 MB); latency ~matches/trails bf16 at large M
        (bf16 MMA, no fp8-TC on the weight-only path). **Accuracy gate
        (256p/120f vs goldens): block `rms_rel=0.2135`, per-row `0.2373` —
        both FAIL `allclose(rtol=2e-2)`. H3 is not FP8-native; block-wise
        beats per-row but does not close the gap.** FP8 remains an
        **opt-in memory/bandwidth tradeoff** (select via `--fp8-scale
        block`); bf16 block-streaming stays the accuracy-preserving
        default (inc 5-7). (inc 8)
  - [ ] Confirm the emulated envelope on a true 24 GB card before release
- [x] Task 3 — Ref2VA store path + denoising-step prefetch overlap
      ([#8](https://github.com/EfficientMoE/Omni-Infinity/issues/8)):
  - Ref2VA selects `transformer_ref` and passes an ordered
    `MiniMaxH3ImageReference` list. Its packed reference rows now use the same
    store-backed, host-AdaLN, bf16 block-streamed, and text-encoder-streamed
    execution path as FL2VA.
  - The final transformer weight group now prefetches the first group for the
    next denoising step. CUDA-event QA measured H2D/compute overlap ratios of
    **0.637** and **0.599** for destination steps 2 and 3.
  - The required 256p/120-frame smoke reproduced the full-resident video and
    audio latents bitwise in **77.0 s**, peaking at **10.72 GiB**.

Required Ref2VA smoke gate:

```bash
python examples/ref2va_smoke.py --ref tests/fixtures/ref.png --seed 0 --steps 8 --resolution 256p --frames 120 --max-vram 22GiB
```

See [docs/ref2va_step_overlap.md](docs/ref2va_step_overlap.md) for golden
provenance, store inspection, parity, and overlap commands.

Reference smoke (diffusers >= 0.40; `--offload` runs components
sequentially when the ~144 GB FL2VA set exceeds one GPU; H3 generates
5-15 s at 24 fps, so `--frames` must be >= 120):

```bash
python examples/fl2va_smoke.py --prompt "a red ball bouncing" \
    --seed 0 --steps 8 --resolution 256p --frames 120 --offload \
    --record-goldens tests/fixtures/goldens
```

Memory-constrained path — host-resident AdaLN cache + bf16 block-streaming
from a moe-store, bitwise-identical latents (the transformer's blocks stream
one at a time; parity is preserved because bf16 weights move device-only):

```bash
python examples/fl2va_smoke.py --prompt "a red ball bouncing" \
    --seed 0 --steps 8 --resolution 256p --frames 120 \
    --checkpoint <local-H3-snapshot-dir> --offload \
    --store-dir <moe-store> --store-components transformer,vae,audio_vae \
    --adaln-host-cache --block-stream-blocks-per-group 1 \
    --goldens tests/fixtures/goldens/fl2va_goldens.pt
```

## Job-serving API

Install the optional serving stack:

```bash
pip install -e '.[serve]'
```

The server loads one immutable registry profile at startup and reuses that
runner for every job. GPU generation is serialized through one in-process
worker; requests for a different model architecture or optimization set
return HTTP 409 rather than loading another pipeline.

Example startup for the streamed dense profile and a 22 GiB budget:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/mnt/raid0nvme0/leyang/hf-home \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMNI_CHECKPOINT=/mnt/raid0nvme0/leyang/hf-home/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe \
OMNI_STORE_DIR=/mnt/raid0nvme0/leyang/h3-store-v2 \
OMNI_STORE_COMPONENTS=transformer,vae,audio_vae \
OMNI_MODEL_ARCH=h3-dense \
OMNI_OPTIMIZATIONS=adaln-host-cache,block-stream,text-encoder-stream \
OMNI_MAX_VRAM=22GiB \
OMNI_JOBS_DIR=./jobs \
OMNI_HOST=127.0.0.1 \
OMNI_PORT=8000 \
python -m omni_infinity.serve
```

`OMNI_HOST`, `OMNI_PORT`, and `OMNI_JOBS_DIR` default to `127.0.0.1`,
`8000`, and `./jobs`. The checkpoint and store variables are optional in the
configuration schema but are required for the offline streamed recipe above.

Create and poll an FL2VA job:

```bash
curl -i http://127.0.0.1:8000/v1/jobs \
  -H 'content-type: application/json' \
  -d '{
    "type": "fl2va",
    "prompt": "a red ball bouncing",
    "model_arch": "h3-dense",
    "optimizations": [
      "adaln-host-cache", "block-stream", "text-encoder-stream"
    ],
    "seed": 0,
    "num_inference_steps": 8,
    "resolution": "256p",
    "num_frames": 120,
    "first_frame_base64": "<optional-base64-PNG>",
    "last_frame_base64": null
  }'

# HTTP 202, Location: /v1/jobs/0123456789abcdef0123456789abcdef
# {
#   "id": "0123456789abcdef0123456789abcdef",
#   "status": "queued",
#   "progress": {"completed_steps": 0, "total_steps": 8, "percent": 0.0},
#   ...
# }

curl http://127.0.0.1:8000/v1/jobs/0123456789abcdef0123456789abcdef
curl -o artifact.mp4 \
  http://127.0.0.1:8000/v1/jobs/0123456789abcdef0123456789abcdef/artifacts
```

The prompt is required; the remaining generation fields have the defaults
shown above. First/last frames are optional base64-encoded images in the JSON
body (20 MiB decoded limit per image). Requests must repeat the server's exact
fixed `model_arch` and ordered optimization list.

Jobs move through `queued -> running -> succeeded|failed|cancelled`.
Progress reports completed denoising transformer forwards, not a wall-clock
estimate: queued jobs are `0/N`, running jobs advance after each forward, and
successful jobs finish at `N/N`. Artifact access returns HTTP 409 until the
job succeeds, then returns an H.264/AAC MP4 with one video stream and one
stereo audio stream. `N` is `num_inference_steps` for `h3-dense` and model
evaluations for `vdn-hybrid`. There is no public cancellation endpoint in
issue #9; `cancelled` records arise when clean shutdown cancels queued work.

Each job is persisted atomically under the configured jobs directory:

```text
jobs/<uuid>/
  job.json
  input-first.png   # only when supplied
  input-last.png    # only when supplied
  output.wav
  output.mp4
```

On process restart, persisted `running` jobs become `failed` with a restart
message; denoising is not resumed. Queued records remain on disk but are not
automatically resubmitted. Terminal records remain pollable directly by their
job ID (the same 32-hex value used as `<uuid>` above). `output.wav` is an
internal stereo sidecar; the public artifacts endpoint returns only the MP4,
whose AAC stream contains the same audio. `type="ref2va"` is schema-valid but
returns HTTP 501 until Task 3 / issue #8 lands. Webhooks remain deferred from
issue #9: the v0.1 contract is polling because callback authentication,
signing, delivery persistence, and retry semantics have not yet been
specified.

## Deferred: shared `moe-kernels` package

> **Deferred — shared `moe-kernels`.** Kernels currently live behind the
> `omni_infinity/kernels/` facade. **Extract to a standalone `moe-kernels`
> package (sibling to moe-store, one-way dep) when the first COMPILED
> CUDA/CUTLASS kernel (e.g. BatchGen's SM120 cutlass GEMM) must be shared
> across ≥2 EfficientMoE runtimes** (per-arch prebuilt wheels are the real
> cost driver; pure-Triton copy-paste is nearly free). Layout: flat
> op-centric API (`moe_kernels.fused_fp8_gemm`, `.mxfp4_gemm`, …); internal
> `_impls/<op>/{triton,cutlass_sm120,…}.py` self-registering into a
> `(op, qtype, backend, arch)` registry with a pure-torch reference as
> lowest-priority fallback; base wheel pure-Python (Triton + reference +
> registry, universal), compiled CUTLASS in optional `[cutlass-cuXYZ]`
> extras (flash-attn/sgl-kernel-style per-(torch,cuda) wheels, multi-arch
> fatbin). Consumers pin ranges (`>=0.x,<0.y`); the reference fallback is
> the forward-compat valve (no lockstep releases). **Extraction = `git mv`
> impls + swap the facade's imports; call sites are already stable.**

## License

Apache-2.0. See [LICENSE](LICENSE).
