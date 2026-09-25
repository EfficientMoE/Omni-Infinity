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

## License

Apache-2.0. See [LICENSE](LICENSE).
