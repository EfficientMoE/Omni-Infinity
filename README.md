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
- [ ] Task 2 — component offload + AdaLN caching + FP8 on a single 24 GB GPU

Reference smoke (diffusers >= 0.40; `--offload` runs components
sequentially when the ~144 GB FL2VA set exceeds one GPU; H3 generates
5-15 s at 24 fps, so `--frames` must be >= 120):

```bash
python examples/fl2va_smoke.py --prompt "a red ball bouncing" \
    --seed 0 --steps 8 --resolution 256p --frames 120 --offload \
    --record-goldens tests/fixtures/goldens
```

## License

Apache-2.0. See [LICENSE](LICENSE).
