# Real-time H3 on RTX PRO 5000 (sm120): theoretical performance model + acceleration research

Tracking: [#10](https://github.com/EfficientMoE/Omni-Infinity/issues/10) (VDN-H3),
[#42](https://github.com/EfficientMoE/Omni-Infinity/issues/42) (SM120 roadmap P1–P7).
Refs: [sm120 gap analysis](../../sm120_gap_analysis.md),
[t2v survey](../../t2v_optimization_survey.md),
[repro_vdn](../../repro_vdn.md), [attribution_vdn](../../attribution_vdn.md),
[baselines_vdn](../../baselines_vdn.md), [streaming_bench](../../streaming_bench.md).

> **Verification caveat.** This is an auto-research plan: it defines what an
> autonomous worker must measure and derive, not pre-decided conclusions.
> Numbers tagged *(SoR)* are the repo's source-of-record measurements (do not
> re-measure). External figures are leads (⚠) until re-confirmed. The Phase-1
> theoretical model is the **gating prerequisite** — no acceleration
> recommendation in Phases 3–5 may be made before the model in Phase 1 exists
> and is validated against the SoR anchors.

> **STATUS 2026-10-08 — SUPERSEDED (do NOT execute from scratch).** The
> `infra4videostreaming` report *MiniMax-H3 在 RTX PRO 5000 上的性能上界与加速手段*
> already delivers Phases 1–5 of this plan, more rigorously, and it has been
> independently validated (see
> `/mnt/raid0nvme0/leyang/Infra4VideoStreaming/papers/VALIDATION_h3_pro5000_bound_2026-10-08.md`
> — cited claims 33/33, derived arithmetic exact, 实测 confirmed to the digit).
> Phase→report mapping and the genuine residual open items are in the new
> **"Status: superseded"** section at the bottom. Keep this plan only as the
> structured question list; start any further work from the report's results,
> not by re-deriving them (user directive: *do not rebuild the wheel*).
>
> One correction the validation forces on this plan: Phase 1's **FP8/FP4 ridge
> roofline is unsound** because NVIDIA does not publish BF16/FP8/dense-FP4 TFLOPS
> for this SKU (the 271/529/1069 figures are third-party). Use the report's
> **bandwidth (byte-bound) roofline** instead — it already shows the 49-step
> byte floor (5.64–12.3 s) exceeds the 5.167 s playback time, which settles RQ2
> without needing tensor-TFLOPS numbers.

## Objective

Answer, with a defensible theoretical performance model as the primary
guidance: **on a single RTX PRO 5000 Blackwell (sm120, 72 GB), can MiniMax-H3
video generation run faster than playback (generation ≥ 24 fps, i.e.
`e2e_rtf ≤ 1.0`), and if so with which combination of techniques?** Classify
every candidate technique on two axes — **lossless vs lossy** (does it change
the reference output?) and **training-free vs training-required** — and map
each onto the bottleneck it attacks in the model. Quantify three gaps
explicitly: the **VDN-H3 sm120 gap** (what the architecture achieves on B200
vs what sm120 permits), the **SGLang-diffusion gap**, and the
**Stable-Diffusion/diffusers gap**.

The hard sequencing constraint (user directive): **the H3 performance bound
and theoretical performance model come first** and gate everything else.

## Research questions

- **RQ1 (model).** What is the analytical per-component (encode / dense or
  window attention / linear branch / FFN / AdaLN / VAE decode) roofline for H3
  on sm120, and which components are compute-bound vs memory-bandwidth-bound at
  BF16 / FP8 / FP4?
- **RQ2 (bound).** What is the *theoretical floor* latency per 24-frame chunk
  and per full clip on one PRO 5000, and therefore the minimum achievable
  `e2e_rtf` / `chunk_rtf`? How far is the floor from 24 fps?
- **RQ3 (taxonomy).** Of the lossless/lossy × train/no-train techniques, which
  are *sufficient and composable* to close the measured real-time gap, and
  which are individually necessary?
- **RQ4 (gaps).** How much of the B200→PRO 5000 VDN-H3 gap is hardware
  (bandwidth, FP8/FP4 TFLOPS, no tcgen05) vs software (kernels, graphs,
  parallelism)? What do SGLang-diffusion and diffusers ship that we do not?
- **RQ5 (prioritization).** Given the model, what is the ranked, accuracy-gated
  sequence of P1–P7 (and beyond) that maximizes fps-per-unit-effort toward
  real-time, separating "deployable now, training-free" from "needs training"?

## Baseline data already in hand — do NOT re-measure

H3 workload / measured performance *(SoR: repo docs & `results/vdn/`)*:

| fact | value | source |
|---|---|---|
| Components | 33B DiT (50 main blocks) + Qwen3-VL-32B encoder + 2 VAEs; ~13B AdaLN branches cacheable | README, ARCHITECTURE |
| Latent tokens vs frames | ~311 tok/frame (38,510@120f · 69,090@222f · 104,766@345f) | attribution `scaling.csv` |
| Dense per-NFE @222f | **28.50 s/NFE** | repro_vdn |
| Hybrid bf16 (V0) / +kernels (V1) / +fp8 (V2) @222f | 19.00 / 16.37 / **13.28** s/NFE (2.15× vs dense) | repro, ablation |
| Dense step scaling exponent | **1.6839** (R²0.9996) in tokens | attribution |
| Hybrid step scaling exponent | **1.0739** (R²1.0) in tokens | attribution |
| Dense-attention share of dense step | **~83%** (29,812.8 / 35,868.2 ms) | attribution `components.csv` |
| Window-softmax / dense-attention ratio | 0.3031 | attribution |
| Per-layer dense→V2 (50 blocks) | 717.36 → 318.60 ms | attribution |
| Peak mem resident / streamed | 86.9 GiB / **19.87 GiB** (bitwise) ; fp8 resident 65.21 GiB | ablation |
| Speedup factorization | 1.50(arch) × 1.16(kernels) × 1.23(fp8) × 6.25(NFE 50→8) = **13.376×** | attribution |
| Cross-step H2D/compute overlap | 0.637 / 0.599 (steps 2/3) | ref2va_step_overlap |
| 72 GB-cap effect on PRO 5000 | 13.29 (capped) vs 13.28 (uncapped) → **capacity-only; PRO 5000 72 GB perf ≈ benched Server GPU for this workload** | `results/vdn/pro5000/` (gitignored; SoR here) |

RTX PRO 5000 Blackwell ceilings *(validated 2026-10-08; see the bound-validation
note in the infra4videostreaming project:
`/mnt/raid0nvme0/leyang/Infra4VideoStreaming/papers/VALIDATION_h3_pro5000_bound_2026-10-08.md`)*:

| anchor | value |
|---|---|
| VRAM / mem BW | 48 or **72 GB** GDDR7 / **1,344 GB/s** (NVIDIA-published) |
| NVIDIA-published compute | **2,064 AI TOPS** (sparse FP4) only; FP32 65 TFLOPS (datasheet PDF) |
| Tensor TFLOPS **(⚠ third-party, NOT NVIDIA)** | BF16 ~271 · FP8 ~529 · FP4 ~1,069 — sm_120 microarch reverse-engineering ([ref](https://zartbot.github.io/micro_arch/nvidia/sm_120/08_workload_analysis.html)); do NOT cite as a ceiling |
| Roofline ridge AI (FLOP/byte) | derived from the third-party TFLOPS above → **use the byte-bound/bandwidth roofline as the sound bound** (NVIDIA omits BF16/FP8/dense-FP4 TFLOPS) |
| Attention caveat | no tcgen05/WGMMA → FA2-class `mma.sync`; **46–61%** efficiency ceiling (third-party) |
| SM / cores / L2 | 110 SM · 14,080 CUDA · 440 5th-gen TC · 96 MB L2 |
| vs PRO 6000 / vs B200 | ~0.5–0.53× compute, 0.75× BW / **0.12× FP8-FP4, 0.17× BW** |

Real-time / playback contract *(SoR: streaming_bench, contract.py)*:
24 fps; 24-frame chunk = 1.0 s; `chunk_rtf = produce_ms/(chunk_s·1000)`;
targets `chunk_rtf_p50 ≤ 1.0`, `sustained_fps ≥ 16` (external tier) / ≥ 24
(true real-time), `stall = 0`, `av_offset ≤ 40 ms`, plus TTFF.

**Motivating gap (from SoR, to be refined by Phase 1):** best current path
(VDN-H3 fp8, 8-step) at 222 f / 768p = 8 × 13.28 ≈ **106 s** for ~9.25 s of
video → `e2e_rtf ≈ 11.5×`. Real-time on one PRO 5000 therefore needs **~11.5×**
beyond today's best. The taxonomy (Phase 3) shows training-free methods cap at
~6–7× — so the model must determine the necessary training-based / streaming /
multi-GPU additions.

---

## Phase 1 (PREREQUISITE) — H3 theoretical performance model & bound

**Deliverable:** `docs/h3_performance_model.md` + `benchmarks/h3_roofline.py`.
Nothing downstream proceeds until this validates against the SoR anchors.

### Task 1.1 — Component FLOP & byte accounting
- [ ] Derive closed-form FLOPs and memory traffic (weights + activations +
      KV-free attention) for each component as a function of latent tokens `L`,
      heads, dim, blocks: dense attention (O(L²)), VDN window-softmax (3.57%
      density ⚠ confirm from `window.py`), linear branch (O(L)), FFN/proj
      (363 Linears *(SoR)*), AdaLN, VAE decode, Qwen3-VL encode.
- [ ] Compute arithmetic intensity (FLOP/byte) per component at BF16/FP8/FP4
      and classify each vs the ridge AIs (220/430/869) → compute- vs
      memory-bound label per component.

### Task 1.2 — Analytical latency / throughput model
- [ ] `t_component = max(FLOPs/peak_compute, Bytes/peak_BW) / efficiency`, with
      an **attention efficiency ceiling of 0.46–0.61** on sm120 (no tcgen05)
      and a separate streaming term for H2D weight traffic (use measured
      overlap 0.637/0.599 as the overlap prior).
- [ ] Assemble step latency → per-NFE → e2e = encode + Σ(NFE·step) + decode.
- [ ] Emit the model as runnable code producing s/NFE, peak GiB, and
      bound-label for any (resolution, frames, NFE, precision, arch).

### Task 1.3 — Validation gate (MUST pass before Phase 2)
- [ ] Model-predicted s/NFE reproduces SoR anchors within **±20%**: dense
      28.50 and V2 13.28 @222f; dense-attention share ≈83%; scaling exponents
      (dense ≈1.68, hybrid ≈1.07) recovered from the FLOP curves.
- [ ] Record every miss >20% with the suspected cause (kernel efficiency,
      launch overhead, VAE, encoder) — the residual *is* a finding.

### Task 1.4 — The real-time bound
- [ ] From the validated model, compute the **theoretical floor** per 24-frame
      chunk and per clip at 256p and 768p, for BF16/FP8/FP4 and NFE ∈
      {50,8,4,2,1}; report minimum achievable `chunk_rtf` / `e2e_rtf` and the
      bottleneck component at the floor.
- [ ] State the real-time feasibility verdict per configuration: reachable on
      1× PRO 5000? only at lower res? only with streaming/AR? only multi-GPU?

**Verification (Phase 1):** `pytest` unit tests on the FLOP/byte formulas
(hand-checked small cases); the ±20% validation table printed with PASS/FAIL
per anchor; `docs/h3_performance_model.md` contains the roofline plot data and
the real-time bound table.

---

## Phase 2 — Real-time feasibility gap (generation vs playback)

### Task 2.1 — Gap quantification
- [ ] Using Phase 1, tabulate the multiplicative gap to `e2e_rtf ≤ 1.0` for each
      (res, NFE, precision, arch) vs the SoR 11.5× motivating point; decompose
      the gap into step-count, attention, precision, and system (launch/overlap)
      factors.

### Task 2.2 — Streaming vs one-shot framing
- [ ] Model chunked streaming (24-frame chunks, pipelined encode/denoise/decode)
      vs one-shot; determine whether TTFF + sustained `chunk_rtf ≤ 1.0` is
      feasible before the full clip, and the queue depth needed to hide tails.

**Verification:** a single gap table keyed to the streaming contract metrics;
every cell traces to a Phase-1 model output, not an external claim.

---

## Phase 3 — Technique taxonomy mapped onto the model (two axes)

Seed (external leads ⚠; classify, then score each against the Phase-1 model —
"model speedup" column is computed by the model, not copied from papers).

| technique | lossless/lossy | train/no-train | reported ⚠ | bottleneck it attacks |
|---|---|---|---|---|
| CUDA graphs + regional `torch.compile` | **lossless** | no-train | 1.5× DiT / 2.7× pipe | launch overhead |
| cuDNN 9.18+ sm120 SDPA / fmha_v2 | **lossless** (near) | no-train | 1.1–1.3× | attention |
| Sequence parallel (Ulysses/Ring/USP), PipeFusion | **lossless** | no-train | 1.3–1.6× | attention scale-out (multi-GPU) |
| VAE tiling / parallel, prefetch-overlap | **lossless** | no-train | — | decode / H2D |
| TeaCache / FBCache / cache-dit (DBCache+TaylorSeer) | lossy | no-train (PTQ-like) | 1.7–4.4× | step compute |
| Sparse VideoGen / VSA / Radial / STA | lossy | no-train (some LoRA) | 1.9–2.3× | attention |
| FP8 weight-only / native FP8 compute | lossy (gated) | no-train (PTQ) | 1.2–2× | bandwidth / compute |
| NVFP4 / SVDQuant / Nunchaku | lossy (gated) | no-train (PTQ+calib) | up to ~3× | bandwidth |
| ViDiT-Q (W8A8/W4A8) | lossy (gated) | no-train (PTQ) | 1.4–1.7× | compute/bandwidth |
| DMD2 / CoDMD / rCM / DFD / PDMD (few-step distill) | lossy | **train** | 12–50× (step cut) | step count |
| VDN / LinGen / LinVideo (linear/hybrid attn) | lossy (recoverable) | **train** | 11–15× | attention O(L²)→O(L) |
| CausVid / Self-Forcing / Diffusion Forcing (AR streaming) | lossy | **train** | 9.4 fps streaming ⚠ | latency-to-first + playback coupling |

### Task 3.1 — Score & compose
- [ ] For each technique, compute the model-predicted speedup on H3@PRO 5000 and
      its composability (which bottleneck; do two techniques fight for the same
      bottleneck?). Produce the **minimal composition that reaches `e2e_rtf ≤ 1`**
      and the **best training-free-only composition** (expected to fall short —
      quantify by how much).
- [ ] Separate "lossless-only" real-time feasibility (if any) from
      "lossy-but-accuracy-gated" (tie each to the repo's `rms_rel`/`allclose`
      gate and C5-style calibration).

**Verification:** every technique row has a model-derived speedup + an
axis-A/axis-B label + a source URL; the two recommended compositions are stated
with their predicted `e2e_rtf` and accuracy tier.

---

## Phase 4 — The three gap analyses

### Task 4.1 — VDN-H3 sm120 gap
- [ ] Decompose the B200→PRO 5000 gap for VDN-H3: SGLang-diffusion reports
      6.9 s denoise / ~9.0 s e2e for 14.4 s video on 8×B200 (MXFP8:
      47.6/25.9/13.1/6.9 s on 1/2/4/8 B200 ⚠). Using Phase-1 ceilings, attribute
      the single-B200 → single-PRO-5000 slowdown to bandwidth (0.17×), FP8/FP4
      TFLOPS (0.12×), and the missing tcgen05/FA4 async path (attention
      efficiency 0.46–0.61) vs software-closable terms (CUDA graphs, native FP8,
      NVFP4, SP).

### Task 4.2 — SGLang-diffusion gap
- [ ] Enumerate what SGLang `multimodal_gen` ships that Omni-Infinity lacks
      (Ulysses+Ring SP, cache-dit, distributed VAE, CFG parallel, CuTe-DSL fused
      norms) and which are portable to sm120 / this diffusers path; map each to a
      P1–P7 item or a new gap.

### Task 4.3 — Stable-Diffusion / diffusers gap
- [ ] Enumerate diffusers fast-paths not yet used (regional `torch.compile`,
      cuDNN SDPA backend switch, `apply_first_block_cache` θ≈0.06–0.08, group
      offload tuning) and the quality/latency each buys on this pipeline.

**Verification:** three gap tables, each row tagged hardware-bound (not
closable) vs software-closable (maps to a task/plan); no number without a source
or a model derivation.

---

## Phase 5 — Synthesis & prioritization

### Task 5.1 — Ranked roadmap keyed to the model
- [ ] Re-rank P1–P7 (and any new items) by model-predicted fps-per-effort toward
      real-time, explicitly splitting **deployable-now / training-free / lossless
      or accuracy-gated** from **requires-training**. State the smallest set that
      crosses `e2e_rtf ≤ 1.0` at a named (res, frames) and its accuracy tier.
- [ ] Write `docs/realtime_feasibility.md` tying Phases 1–4 into the verdict;
      cross-link from `docs/README.md` and issue #42.

**Verification:** the roadmap's top recommendation is traceable to a Phase-1
model output and a Phase-3 composition; docs build and links resolve;
`ruff`/`pytest -m "not gpu and not weights"` clean.

---

## Success criteria

1. `docs/h3_performance_model.md` + `benchmarks/h3_roofline.py` exist and the
   model reproduces every SoR anchor within ±20% (printed PASS table).
2. A stated, defensible **real-time bound**: minimum `e2e_rtf`/`chunk_rtf` on
   1× PRO 5000 at 256p and 768p, with the bottleneck named.
3. Every candidate technique classified on both axes with a model-derived
   (not paper-copied) speedup and a source URL.
4. Three gap tables (VDN-H3 sm120 / SGLang / diffusers) separating
   hardware-bound from software-closable, each mapped to P1–P7 or a new item.
5. A ranked, training-free-vs-training-split roadmap whose top entry is
   traceable to the model.

## Risks

| risk | response |
|---|---|
| Power-law extrapolation below 120 f is unreliable | the model must be FLOP/byte-based; power-laws are validation only, never the bound |
| Exact H3 FLOPs need config introspection | derive from `registry`/diffusers config + attribution anchors; hand-check small cases; mark any assumed dim ⚠ |
| sm120 TFLOPS/BW figures are third-party ⚠ | re-confirm against NVIDIA datasheet; carry a ±range, not a point, into the bound |
| Attention efficiency ceiling (0.46–0.61) dominates the bound | treat as a sensitivity parameter; report the bound as a band across it |
| "Real-time" ambiguity (one-shot vs streaming, res, frames) | always report per (res, frames, mode); default headline = 768p, 24-frame-chunk streaming, `chunk_rtf_p50 ≤ 1.0` |
| External taxonomy speedups don't transfer to H3/sm120 | only model-derived speedups enter the recommendation; paper figures stay in the ⚠ column |

## Status: superseded by the validated infra4videostreaming report (2026-10-08)

This plan's five phases are already answered by the PRO 5000 bound report
(validated: `Infra4VideoStreaming/papers/VALIDATION_h3_pro5000_bound_2026-10-08.md`).
Phase → where it is delivered:

| plan phase | delivered by (report section) | note |
|---|---|---|
| **P1** H3 theoretical model + real-time bound | §0–§3 service inequality `R=T_play/T_gen` + **byte-bound roofline** (49-step floor 5.64–12.3 s > 5.167 s playback) | uses bandwidth 1,344 GB/s only; deliberately leaves tensor-TFLOPS columns blank — **sounder than this plan's FP8/FP4 ridge approach** |
| **P2** real-time feasibility gap | §0, §4, §6 (VDN 1×PRO6000 `R≈0.08`; ~11.5× gap) | one-shot vs buffered-chunk framing in HANDBOOK §workload |
| **P3** technique taxonomy (lossless/lossy × train/no-train) | §9.1–§9.4 | more complete than this plan's seed table |
| **P4** gaps: VDN-H3 sm120 / SGLang / SD | §6 / §7 / §8 | SGLang/cookbook numbers validated 11/11 |
| **P5** synthesis + ranked roadmap | §10 (值得做的) | training-free cannot cross `R>1`; only step-cut + block-causal (training) or more HW |

**Genuine residual open items** (NOT covered by the report; the only things worth
doing next, none of which is "re-derive the bound"):

1. **Dense BF16 H3 single-card sm120 wall-clock** — never measured (report §11);
   today only VDN (8-step hybrid) has a 1×PRO6000 datapoint (160 s + 20.6 s).
2. **345→102 latent-frame mapping** — the validation resolved the measured
   scaling toward ~104k tokens (102 latent), but the model card gives no formula
   (report §11); confirm from H3 VAE config if a definitive answer is needed.
3. **Clean NVFP4 timing on LongLive-2.0-5B** — REPORT_microbenchmark §6.6's
   NVFP4 run (−46.3% / 23.63 GiB anchor) is **LongLive-5B, not H3**; its
   per-step timing was contention-polluted/excluded. An H3 NVFP4 path does not
   exist — that is the separate `2026-10-07-p5-nvfp4` plan.
4. **Training path** (the only lever that moves the bound): block-causal
   H3 + few-step distillation so the service unit becomes a short chunk rather
   than the whole 10⁵-token clip — then re-measure on sm120 (the report is
   explicit that a byte-bound floor ≠ a wall-clock prediction; LongLive shows a
   30–40× efficiency gap above the floor).

**Momus review status:** the Momus model was repointed to `elm/gpt-5.6-sol`
(2026-10-08) and now runs. The four residual items above were turned into an
**executable plan that Momus approved ([OKAY], v3)**:
`/mnt/raid0nvme0/leyang/Omni-Infinity/.sisyphus/plans/2026-10-08-h3-sm120-residual-measurements.md`.
This document stays a superseded record — execute the residual plan, not this one.
