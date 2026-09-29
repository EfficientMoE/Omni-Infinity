# Opt-in caches (issue #24)

Implementation of the cache survey in
[#24](https://github.com/EfficientMoE/Omni-Infinity/issues/24): every
cache that exists or is missing around the H3 pipeline, which ones this
repo now ships, and which ones stay deferred. Two invariants hold for
everything on this page:

1. **Nothing is on by default.** The parity gates
   (`tests/test_reference_parity.py`, `tests/test_ref2va_parity.py`,
   `tests/test_vdn_parity.py`) run without any of these caches, and the
   registry's documented serving profile does not include them.
2. **Approximate caches never claim `rms_rel=0`.** C1/C3 are exact by
   construction; C5 trades latent error for speed and carries its own
   quality gate.

| # | Cache | Status | Where |
|---|---|---|---|
| C1 | Exact condition cache (cross-request) | **shipped**, opt-in | `omni_infinity/caches/condition.py` |
| C2 | Encoder prefix cache | deferred | — (see below) |
| C3 | Vision-embedding cache | **shipped**, opt-in | `omni_infinity/caches/vision.py` |
| C4 | Shape caches already in VDN | left as-is, documented | `third_party/vdn-minimax-h3` |
| C5 | Denoise-step feature cache | **scaffold shipped**, needs calibration | `omni_infinity/caches/denoise.py` |

Nothing here patches `third_party/vdn-minimax-h3` or
`MiniMaxH3ModularPipeline`: C1/C3 wrap inputs the runners already pass,
and C5 is a wrapper on the transformer `forward`.

## C1 — exact condition cache (`condition-cache`)

The VDN prompt `.pt` file lifted into the runner. Key = SHA-256 over the
model namespace, the **whole prompt string**, then the condition image
bytes **in order** (first/last frame, references), then the canvas
(`height`, `width`, `num_frames`) the keyframes are resized to. Values =
the encode intermediates the denoise block re-declares as inputs:
`prompt_embeds` (layer-50 Qwen3-VL), `text_token_tags`, and the
keyframe/reference VAE `condition_latents` / `audio_condition_latents`.

**A hit requires the entire condition to match.** The three shipped T2VA
examples share only the `[Shot 1]` opener; that shared span is invisible
here by design — no prefix semantics exist anywhere in the key, and no
`[Shot N]` reordering is done to manufacture one (#23).

Why a hit is exact (diffusers 0.40 `MiniMaxH3ModularPipeline`):

- the text encode runs `use_cache=False`, samples nothing, and reads
  `hidden_states[50]`, so its outputs are a pure function of the
  condition;
- the keyframe/reference VAE posterior is sampled under a *fresh*
  generator seeded independently of the request
  (`components.keyframe_encode_seed`), so `condition_latents` are a pure
  function of the image bytes and canvas;
- neither encode block consumes the request `generator`, so the denoise
  RNG stream is byte-identical whether or not the encoders ran.

Mechanically, a hit runs a second *conditioned* pipeline built once via
the modular-diffusers split-blocks pattern
(`SequentialPipelineBlocks.from_blocks_dict`) with `text_encoder` and
`vae_encoder` removed and every loaded component shared; the cached
tensors are passed as pipeline inputs. With `text-encoder-stream`
enabled, a hit therefore skips the entire streamed Qwen3-VL pass —
encode cost and weight traffic both. Anything unexpected (a pipeline
without introspectable blocks, an unhashable reference object) fails
open to the ordinary encode path.

Storage follows vLLM-Omni's AR stage-output cache posture — exact
identity, host-resident — without its block/prefix semantics, which H3's
fixed shot order makes worthless. Entries live on the CPU in an LRU
(`max_entries`, optionally `max_bytes`); an optional disk tier persists
each entry as `<key>.pt` (atomic write, `weights_only=True` load), which
is exactly the VDN prompt-cache file format made runner-managed. The job
server does not keep encoder state between jobs; with
`OMNI_CONDITION_CACHE_DIR` set the disk tier is what survives.

Enable: `--condition-cache [--condition-cache-dir DIR]` on
`examples/fl2va_smoke.py`, `condition_cache=True` on either runner's
`from_pretrained`, or the `condition-cache` optimization in a server
profile (plus `OMNI_CONDITION_CACHE_DIR` for persistence).

## C2 — encoder prefix cache (deferred)

vLLM-style block-hashed prefix reuse inside the Qwen3-VL encode needs
the tower resident and a KV cache (`use_cache=True`), which fights the
~10 GiB story: the streamed encoder is released after every encode, and
there is no resident KV to hash. A complete leading block would also
have to include the `<Picture i>` vision tokens that sit in front of the
text, so Picture 1 matches only when the image bytes match *and* it is
first. Deferred until a resident-encoder deployment exists;
ContextPilot's `.pth` / chat proxy is explicitly not the road there
(#23). C3 was designed to be looked up inside C2 spans once C2 lands.

## C3 — vision-embedding cache (`vision-cache`)

vLLM-Omni's multimodal *encoder cache* rule: content hash → vision-tower
output, lookup independent of any block alignment. The wrapper replaces
the tower's instance `forward` (`visual` / `model.visual`), so with a
streamed encoder a hit also skips the tower's weight onloads. Its value
shows when C1 misses but the images repeat — vLLM's "same image with a
different prompt reuses the common work" case.

Entries are keyed per tower *call* (the H3 encode passes a request's
image patches in one call), not per image inside a call; per-image
splitting inside a partially-cached span is C2's follow-up. Outputs are
stored host-side and replayed onto the caller's device, structure
preserved (Qwen3-VL returns embeds + deepstack features).

Enable: `--vision-cache` on the smoke CLI, `vision_cache=True` on either
runner, or the `vision-cache` optimization in a server profile.

## C4 — VDN shape caches (left alone)

The Flex BlockMask, gather-index and delta-rule backend caches inside
`third_party/vdn-minimax-h3` stay exactly as upstream ships them. Two
operational notes worth knowing, not features to expand here:

- the inference FLASH compile is static per process — **one packed
  `seq_len`** — so a new caption length misses it and falls back (or
  re-errors) rather than recompiling per job;
- the BlockMask/gather-index entries are keyed by packed length, frame
  count, window radius and device. The caption text is not part of the
  key; attention itself always runs.

The same constraint shows up in SGLang's H3 cookbook as CUDA graphs
captured for one fixed text-row bucket (5504 rows in the validated
Ref2VA profile): a new token length is a graph miss. Any future
graph/compile cache keyed by sequence length must keep **one length per
process**, matching both codebases — never a capture per job.

## C5 — denoise-step feature cache (calibration required)

TeaCache's decision rule as a wrapper on the transformer `forward` (the
hook pattern vLLM-Omni and SGLang Diffusion both use), out of
`third_party/`: per step, the relative-L1 distance between consecutive
step inputs is rescaled by a model-specific polynomial and accumulated;
under the threshold the evaluation is skipped and the cached result
replayed; the first and last steps always compute; with CFG-style
multiple calls per step, each intra-step call keeps its own slot
(SGLang's separate positive/negative residual slots).

Two replay modes, because H3's forward contract matters:

- `mode="residual"` is TeaCache-faithful (`output − input` replayed as
  `input + residual`) and requires input-shaped outputs;
- `mode="output"` replays the previous prediction (fixed-decision step
  reuse in the FORA family). This is the only mode applicable to H3
  *from outside the forward*: H3 returns per-modality projections
  (`sample`/`audio_sample`) index-selected from the packed rows, so no
  input-shaped residual exists at this boundary.

**There is no default configuration.** Neither TeaCache nor cache-dit
publishes MiniMax-H3 coefficients; SGLang's own uncalibrated fallback
(Wan2.2) silently no-ops, and a copied foreign polynomial silently
regresses quality. `DenoiseCacheConfig` therefore refuses empty
coefficients, which is also why C5 is *not* a registry optimization or a
job-API value: a static profile row cannot carry a calibration. Enable
it programmatically per generation:

```python
from omni_infinity.caches.denoise import DenoiseCacheConfig

config = DenoiseCacheConfig(
    coefficients=(...),  # from an H3-specific calibration sweep
    threshold=...,
    mode="output",
)
runner.generate(prompt, num_inference_steps=8, denoise_cache=config)
```

Any run with C5 enabled is off the bitwise smoke by definition (the
8-step DMD grid is also at the low end of what the SCM-style literature
even supports) and needs its own quality gate — CLIP/SSIM/PSNR against
reference outputs, the way vLLM-Omni gates its diffusion caches. Where
the `cache-dit` library is installed,
`omni_infinity.caches.denoise.enable_cache_dit` delegates to it
(DBCache/TaylorSeer/SCM) rather than growing a third implementation —
the same integration surface SGLang and vLLM-Omni already use, with the
same calibration caveat.

Never publish H3 text K/V from a denoise step as a cross-request prefix:
in this DiT the packed text rows are in the residual stream and change
with the latents every step. vLLM-Omni's HunyuanImage3 diffusion KV
prefix cache is a different contract (its dynamic target-image KV is
excluded from publication, and its identity includes the VAE random
state); it does not transfer to H3 until a measurement shows text K/V
invariant to the latent.

## Serving semantics

The job server keeps its one-profile contract: requests must repeat the
loaded `model_arch` and ordered optimization list exactly, so requests
with different cache settings cannot even share a server — a stricter
form of SGLang's rule that requests with different cache settings do not
share a batch. Progress reporting is unchanged: skipped C5 steps still
fire the transformer forward hook, and a C1 hit still runs one
transformer forward per step.

## Reuse rules copied, and from where

| Rule | Source |
|---|---|
| Exact host-resident stage-output reuse, no partial blocks | vLLM-Omni AR stage-output prefix cache |
| Content hash → encoder output, alignment-independent | vLLM-Omni multimodal encoder cache |
| Rel-L1 + polynomial + accumulator step skipping, dense boundary steps | TeaCache (arXiv:2411.19108), as shipped by both stacks |
| Uncalibrated model ⇒ refuse rather than no-op | inverted from SGLang's Wan2.2 flag behavior |
| Per-CFG-branch cache slots | SGLang TeaCache (Wan2.1 / Z-Image) |
| Fixed-decision output replay | FORA (arXiv:2407.01425) family |
| Delegate block-level caching to `cache-dit` | both stacks' `cache_backend` integration |
| One compiled shape per process | SGLang H3 cookbook CUDA graphs; VDN static FLASH compile |
| No shot reordering to manufacture a prefix | #23 (ContextPilot investigation) |

NIRVANA-style similarity retrieval of intermediate noise states stays
out of scope per the issue: it is cross-prompt *approximate* latent
reuse, and a time-marked H3 prompt is a timeline, not a bag of nearby
captions.
