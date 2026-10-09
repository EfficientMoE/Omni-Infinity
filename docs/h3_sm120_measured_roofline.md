# H3 sm120 measured roofline (PRO 6000) — the R<1 verdict, from measurement

Execution record (2026-10-08) of the Momus-approved plan
`.sisyphus/plans/2026-10-08-h3-sm120-microbenchmark-evidence.md`. Every number
here is **measured on a single RTX PRO 6000 Blackwell Server Edition (sm120,
CC 12.0)** on gala2, one idle card pinned per run, or derived from those
measurements. It converts the previously *inferred / extrapolated* parts of the
"single PRO 5000 cannot sustain R = T_play/T_gen ≥ 1 for MiniMax-H3" verdict into
direct measurements. The byte-bound floor stays the *provable* floor; third-party
tensor-TFLOPS are not cited.

## Verdict (measurement-led)

On this silicon, dense MiniMax-H3 denoising runs **46–89× above the byte floor**
and the gap **grows with sequence length** (the fingerprint of O(N²) attention
compute, which the byte floor omits). Measured dense R is **0.08–0.10 at 8 steps
and 0.013–0.016 at 49 steps** at 768p — one to two orders below real-time — with a
**constant 71.9 GiB resident peak** that fits neither the 48 GB (44.7 GiB) nor the
72 GB (67.1 GiB) PRO 5000. A PRO 5000 is a strict slowdown of this proxy
(`R_PRO5000 ≤ R_PRO6000 / 1.71`). No stacked training-free lever closes it.

## M1 — measured sm120 roofline (`results/h3_residual/sm120_roofline.json`)

`benchmarks/sm120_roofline.py`, GPU pinned, CC 12.0.

| quantity | measured (PRO 6000) | NVIDIA spec | third-party claim (NOT used) |
|---|---:|---:|---:|
| HBM bandwidth | **1485 GB/s** (triad; copy 1459) | 1792 GB/s (≈83%) | — |
| BF16 dense GEMM | **411 TFLOPS** (plateau @8192) | not published | ~271 (an *under*estimate) |
| FP8 dense GEMM (`_scaled_mm`) | **721 TFLOPS** (plateau @8192) | not published | ~529 (an *under*estimate) |
| ridge AI (BF16) | **277 FLOP/byte** | — | — |
| ridge AI (FP8) | 485 FLOP/byte | — | — |

The measured BF16/FP8 ceilings *exceed* the refused third-party figures, so those
figures were never a valid ceiling. This gives a measured ridge for M2.

## M2 — which resource binds (measured ridge, 277 FLOP/byte)

Full-attention arithmetic intensity is `AI_attn ≈ N/2` (FlashAttention: `4N²d`
FLOPs over `~8Nd` HBM bytes). Against the measured ridge:

| N (768p) | attn AI ≈ N/2 | vs ridge 277 | bound |
|---:|---:|---:|---|
| 37,296 | 18,648 | 67× over | **compute** |
| 57,456 | 28,728 | 104× over | **compute** |
| 77,616 | 38,808 | 140× over | **compute** |

Attention (≈83% of the dense step per `attribution_vdn.md`) is compute-bound by
two orders of magnitude. The byte-bound floor is therefore a *provable lower
bound*, not the binding constraint — and the binding constraint (O(N²) tensor
compute) cannot be bounded from NVIDIA-published specs, so it is anchored by M3.

## M3 — efficiency gap grows with N (`results/h3_residual/dense_768p_*.json`)

`benchmarks/dense_denoise_timing.py` (dense BF16, store-backed, AdaLN-host-cache,
no block-stream), one harness, one card. Floor = `(2·20e9 + 50·4·N·7168·2)/1.344e12`.

| config | N | measured s/step | byte floor/step | **gap** | peak GiB | fits 48/72? |
|---|---:|---:|---:|---:|---:|---|
| 256p/120f | ~4,300 | 1.802 | 39 ms | **46×** | 71.9 | no/no |
| 768p/124f | 37,296 | 6.887 | 109 ms | **63×** | 71.9 | no/no |
| 768p/192f | 57,456 | 10.257 | 152 ms | **67×** | 71.9 | no/no |
| 768p/260f | 77,616 | 17.296 | 195 ms | **89×** | 71.9 | no/no |

Step time scales as **N^1.23** (super-linear → O(N²) attention); the gap rises
monotonically (`gap ≈ 6.3e-4·N + 36`, R²=0.87). Peak is a constant **71.9 GiB**
(weight-dominated), so the resident dense config fits neither SKU.

Measured dense R (768p, this harness):

| frames | T_play | R @8 steps | R @49 steps |
|---:|---:|---:|---:|
| 124 | 5.17 s | **0.094** | 0.015 |
| 192 | 8.00 s | **0.098** | 0.016 |
| 260 | 10.83 s | **0.078** | 0.013 |

> **Harness note.** This Omni `ReferenceRunner` + FlashAttention path is ~2–2.8×
> *faster* per step than the VDN upstream `flex` path that produced 28.50 s/NFE
> @222f (repro_vdn flags that path as pessimistic: sm100 misdetect + flex static
> block-sparse schedule). So 63–89× is the *optimistic* measured gap; the VDN-path
> gap is 161×. Both are 1–2 orders above the floor, and R ≪ 1 either way.

## M5 — PRO 5000 upper bound, derived from the measured M1 ceilings

PRO 5000 has 110/188 = 0.585× the SMs and 1344/1792 = 0.75× the bandwidth. From
the measured PRO 6000 ceilings:

| | PRO 6000 (measured) | PRO 5000 (derived) |
|---|---:|---:|
| BF16 TFLOPS | 411 | ~241 |
| HBM GB/s | 1485 | ~1113 |
| ridge AI | 277 | ~216 |

Dense H3 is compute-bound (M2), so PRO 5000 is ~**1.71× slower** (derived) →
`R_PRO5000 ≤ R_PRO6000 / 1.71`. **An MPS 58%-SM-cap emulation on this card
measured a 1.7506× slowdown (192f: 10.26 → 17.96 s/step), confirming the 1.71×
derivation; R_pro5000_upper = 0.064 at 192f/8-step**
(`results/h3_residual/pro5000_emulation.json`). Every R stays ≪ 1.

## M6 — stacked lever product is insufficient (`results/h3_residual/lever_deltas.json`)

Already-measured deltas (repro_vdn Table 1 + attribution): arch 1.50× × kernels
1.16× × fp8 1.23× × NFE-50→8 6.25× = **13.376×**; net dense→V2 per-NFE **2.146×**.
VDN 8-step fp8 is still **R≈0.08**. Stacking every training-free + arch lever does
not cross R>1; only block-causal few-step distillation (training) shrinks the
service unit enough (report §10, §13; `results/h3_residual/training_path_scoping.md`).

## Artifacts

`results/h3_residual/`: `sm120_roofline.json` (M1), `dense_768p_{124,192,260}f.json`
+ `m3_sweep.log` (M3), `pro5000_emulation.json` (M5), `lever_deltas.json` (M6).
Harnesses: `benchmarks/sm120_roofline.py` (new), `benchmarks/dense_denoise_timing.py`
(device-pin relaxed to any single idle index). All runs CC 12.0, PRO 6000 Server
Edition, cards left clean.

**Deferred optional confirmations.** M4's 3-segment decode split was not landed —
the `--decode` harness attempt hit a pipeline `output_type` mismatch and was
reverted; end-to-end R is bounded by the M3 denoise R plus the cookbook 20.6 s
decode (decode-only R≈0.7), so the serial R is only worse than the denoise R
above. M2's full per-component `--emit-roofline` sweep was not run — the binding
is the analytical placement in the M2 section (attention AI ≫ measured ridge 277).
The M6 attention-backend bake-off stayed the pre-existing flex/decomposed 0.98×;
a fresh rerun was blocked by a child-env `omegaconf` gap.
