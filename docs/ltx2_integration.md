# LTX-2.5 integration plan (issue #54)

> **Status: scaffolding.** This PR adds only the `ltx-2.5` arch name, a
> non-functional `Ltx2Runner` skeleton, and placeholder tests. No weights
> load and no generation runs yet. This document is the phased plan for the
> real integration tracked by
> [#54](https://github.com/EfficientMoE/Omni-Infinity/issues/54).

## Context

[`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) is a 22B
joint **audio-video** DiT with open weights under the **LTX-2.x Community
License** (not Apache-2.0 — see "License" below). v0.1 of this repo only
serves MiniMax-H3 (`h3-dense`) and VDN-Minimax-H3 (`vdn-hybrid`). Pointing
the current runners, store, caches, or job API at an LTX-2.5 checkpoint will
not load it, and several H3 contracts are illegal for LTX-2.5 even if the
weights did load.

The Hub repo is a split, Comfy-aligned pack (one `.safetensors` per
component), not a single diffusers repo id. A Diffusers-friendly repack
lives at
[`Lightricks/LTX-2.5-Diffusers`](https://huggingface.co/Lightricks/LTX-2.5-Diffusers),
but `LTX2ImageToVideoPipeline` / `LTX2LatentUpsamplePipeline` are on
diffusers `main` only — not in a release. The vendor runtime is
[`ltx-pipelines`](https://github.com/Lightricks/LTX-2) (`ModelPaths.from_split`).

## Why it does not fit the current contracts

| Axis | H3 family (today) | LTX-2.5 |
|---|---|---|
| Text encoder | Qwen3-VL-32B | **Gemma-4-12B** + projections |
| Transformer | 33B dense / VDN hybrid | **22B joint A/V DiT**, distilled 8-step (CFG = 1) + full/dev DiT |
| Decoders | 2 VAEs (video, audio) | **2 video VAEs** (DiffVAE + Conv), audio VAE + **vocoder**, **diffusion video decoder** |
| Generation | single-stage denoise | **two-stage**: stage-1 denoise -> ×2 spatial upsample -> stage-2 |
| Checkpoint | single diffusers repo id | **split pack** (one `.safetensors` per component) |
| Diffusers | pinned `0.40.0` | pipelines on **`main` only** |
| License | Apache-2.0 / MiniMax | **LTX-2.x Community License** |

The `ReferenceRunner`/`VdnRunner` store-backed, AdaLN-host-cache,
block-stream recipe assumes the H3 modular pipeline shape and a single repo
id; none of that transfers directly.

## Phased plan

- **P0 — scaffold (this PR).** `ltx-2.5` registry entry + `Ltx2Runner`
  skeleton (`from_pretrained`/`generate` raise `NotImplementedError`) +
  placeholder tests. Establishes the name and surface so later phases are
  additive and reviewable.
- **P1 — environment + weights.** Resolve the diffusers-`main` pin vs the
  repo's golden `0.40.0` pin (likely an optional extra / separate env), the
  `ltx-pipelines` dependency, split-checkpoint assembly
  (`ModelPaths.from_split`), and an accepted local snapshot. Document the
  disk footprint and license-acceptance gate.
- **P2 — text encoder.** Gemma-4-12B with projections; decide streaming /
  offload strategy (the repo's leaf-level group-offload lever vs resident).
- **P3 — decoders.** Wire both video VAEs, the audio VAE + vocoder, and the
  diffusion video decoder; define the latent/output contract feeding the
  existing MP4/AAC mux.
- **P4 — two-stage generation.** Implement stage-1 denoise -> upsample ->
  stage-2 in `Ltx2Runner.generate`, honoring the distilled fixed-8-step /
  CFG = 1 schedule and the `GenerationResult` shape.
- **P5 — serving + optimizations.** Decide which registry optimizations
  apply (fp8 / block-stream / text-encoder-stream) and whether the job and
  stream APIs need new fields. Keep anything accuracy-affecting opt-in.
- **P6 — parity + QA.** Golden recording + a `gpu`/`weights` parity gate
  (`tests/test_ltx2_parity.py`) mirroring `test_vdn_parity.py`, registered
  in the nightly GPU suite.

## Open questions / risks

- **diffusers pin conflict.** The CPU-unit + golden stack pins
  `diffusers==0.40.0`; LTX-2.5 pipelines need `main`. These likely cannot
  co-exist in one environment — P1 must decide the isolation strategy.
- **License.** LTX-2.x Community License is not Apache-2.0. Weights are not
  vendored; loading must be gated on explicit user acceptance, and the code
  license (Apache-2.0) is unchanged.
- **Memory.** 22B DiT + Gemma-4-12B + multiple VAEs + vocoder + diffusion
  decoder is a different budget story than H3; the ≤22 GiB envelope will
  need its own analysis.
- **Audio.** Joint audio-video with a separate vocoder stage changes the
  artifact contract; confirm the existing MP4/AAC mux path suffices.

## References

- Tracking issue: [#54](https://github.com/EfficientMoE/Omni-Infinity/issues/54)
- Model card: <https://huggingface.co/Lightricks/LTX-2.5>
- Diffusers repack: <https://huggingface.co/Lightricks/LTX-2.5-Diffusers>
- Vendor runtime: <https://github.com/Lightricks/LTX-2>
- Scaffold: `omni_infinity/arch/ltx2.py`, `tests/test_ltx2_scaffold.py`
