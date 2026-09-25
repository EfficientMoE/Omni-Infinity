# VDN-H3 speedup attribution on sm120

This study decomposes the published VDN-H3 speedup under the repository's
single-GPU protocol: one RTX PRO 6000 Blackwell Server Edition, prompt
`example_0.pt`, eight NFEs, two discarded warmups, and CUDA-event ranges
injected without editing upstream or Omni-Infinity model code. Raw replay
commands and QA output are in
[`attribution_vdn_runlog.md`](attribution_vdn_runlog.md).
[Issue #10 was updated with the numeric verdicts](https://github.com/EfficientMoE/Omni-Infinity/issues/10#issuecomment-5793582234).

The event envelope agrees with device-synchronized wall timing within 0.01%
for every measured NFE. The shim-on/off V0 gate was -0.02%, below the 3%
limit. Component times below are means over the eight recorded NFEs; `other`
is the residual from `step_total`, not an independently timed range.

| rung | step total (ms) | dense | window | linear | gates | other |
|---|---:|---:|---:|---:|---:|---:|
| D | 35,868.2 | 29,812.8 | — | — | — | 6,055.4 |
| V0 | 23,198.9 | — | 9,037.1 | 2,701.1 | 174.1 | 11,286.6 |
| V1 | 19,870.2 | — | 8,374.2 | 1,726.2 | 174.2 | 9,595.6 |
| V2 | 15,930.2 | — | 8,358.7 | 1,723.9 | 174.7 | 5,672.9 |

### H1 — Dense attention exceeds 85% of dense step time

The dense attention modules consume **29,812.8 ms of 35,868.2 ms**, or
**83.12%**. This range uses the `MiniMaxH3Attention` module boundary because
the released dense inference path does not instantiate the planned
`FlexFA4Processor`; it includes 50 main-block and two token-refiner attention
calls. The measured share is dominant but does not cross the paper's strict
greater-than-85% threshold.

**Verdict: REFUTED**

### H2 — The released window realizes 3.57% attention density

The checkpoint records `chunk=5`, `radius=1`, and `anchor_frames="both"`.
Applying `window.py` exactly gives 67 latent frames for requested N=222
(aligned to 226 pixel frames) and 102 latent frames for N=345. The realized
video-frame-pair densities are **26.3088%** and **17.7432%**, respectively.
At N=345 this is +14.1732 percentage points above 3.57%. Dense global tokens
would increase, not reduce, the full packed-sequence density.

At N=222, window/dense CUDA-time ratios are 0.3031 (V0), 0.2809 (V1), and
0.2804 (V2), versus analytic density 0.2631. Their kernel-efficiency factors
are 1.152, 1.068, and 1.066. Thus timing tracks the released geometry, but
the paper's 3.57% evidently uses another denominator or geometry.

**Verdict: REFUTED**

### H3 — Per-layer latency follows the paper's dense→hybrid→optimized shape

Dividing full-step envelopes by the 50 main transformer blocks gives
**717.36, 463.98, 397.40, and 318.60 ms/layer-equivalent** for D, V0, V1,
and V2. The sm120 ratios are D/V0=**1.55×**, V0/V1=**1.17×**, and
V1/V2=**1.25×**; combining the final two gives V0/V2=**1.46×**. The paper's
B200 chain is 332.5→192.1→125.3 ms, or 1.73× then 1.53×. Absolute latency
and the first factor differ, but both platforms show the same two-stage
shape: architecture first, then kernels plus fp8.

**Verdict: CONFIRMED**

### H4 — The linear branch scales approximately linearly and stays secondary

Across the two successful V0 scaling points, the fitted linear-branch
exponent versus packed token count is **1.0181 (R²=1.0000)**. At N=222 it
uses 2,701.1 ms in V0 and 1,723.9 ms in tuned V2, versus 9,037.1 and
8,358.7 ms for window softmax. It is therefore approximately O(L) and
substantially smaller than the window branch. The third requested points,
N=311 and fallback N=294, both OOM before a measured NFE, limiting the fit
to two points.

**Verdict: CONFIRMED**

### H5 — Architecture, kernels, fp8, and distillation close the speedup ledger

The authoritative N=222 baseline ladder supplies the first three factors;
the 50→8 NFE reduction supplies the fourth. Their product reconstructs the
end-to-end denoising ratio to well within the required 5%.

| contribution | factor | cumulative |
|---|---:|---:|
| dense→hybrid architecture | 1.50× | 1.500× |
| inference kernels | 1.16× | 1.740× |
| fp8 | 1.23× | 2.140× |
| 50→8 NFE distillation | 6.25× | **13.376×** |

The corresponding measured endpoint ratio is
`(28.50 s × 50) / (13.28 s × 8)` = **13.413×**, leaving a **0.28%**
residual. On B200, the paper combines architecture at 1.73× and
kernels+fp8 at 1.53×; sm120 yields 1.50× and 1.43× from the baseline ladder.
Task 2 shows why the architecture gain is smaller here: dense attention is
83.12%, below the claimed 85%+, while fp8 removes mostly residual/linear
work rather than reducing the measured window kernel.

Negative controls bound alternative explanations: turbo LoRA alone gives
28.50 s/NFE, exactly the dense baseline; Ulysses contributes at most 1.09×
on PCIe; and the 72 GiB cap changes native-fp8 latency from 13.28 to
13.29 s/NFE. These are not the source of the single-GPU arithmetic gain.

**Verdict: CONFIRMED**

### H6 — Speedup grows with sequence length

Dense step time has fitted exponent **1.6839 (R²=0.9996)** versus packed
token count. V0 has exponent **1.0739 (R²=1.0000)** over its two successful
points. Measured speedup therefore rises from **1.111× at N=120** to
**1.546× at N=222**.

| N | dense s/NFE | V0 s/NFE | speedup | status |
|---:|---:|---:|---:|---|
| 120 | 13.7581 | 12.3849 | 1.111× | measured |
| 222 | 35.8705 | 23.2005 | 1.546× | measured |
| 294 | — | OOM | — | measured capacity bound |
| 311 | — | OOM | — | measured capacity bound |
| 345 | 74.4359 | 35.8184 | 2.078× | **EXTRAPOLATED/OOM-BOUNDED** |

The N=345 V0 value is a linear extrapolation from N=120 and N=222, never a
measured result. Because both planned third hybrid points OOM, the direction
is measured but the three-point hybrid curve success criterion is unmet.

**Verdict: PARTIAL**

## Methodology and artifacts

`benchmarks/vdn_prof_shim/sitecustomize.py` injects CUDA events and NVTX
ranges through `PYTHONPATH`. It records the transformer envelope,
attention branches, gates, and functional-linear call counts, synchronizing
once at step end. `benchmarks/attribution_vdn.py` enforces an idle-GPU check,
launches each run into a dedicated log, polls it to completion, validates
range presence and envelope agreement, and emits the gitignored
`results/vdn/attribution/{components.csv,scaling.csv,speedup_vs_N.md}`.

No file under `third_party/` or `omni_infinity/` was modified. The initial
dense processor hook produced no range, so the documented plan fallback was
applied at the `MiniMaxH3Attention` module boundary and D-prof was rerun.

## Threats to validity

- One sm120 GPU and one prompt were measured; hardware and prompt shapes may
  change component balance.
- V0 overhead is bounded at -0.02%, but the dense module-boundary fallback
  has no separate shim-off gate. Its profiled absolute times exceed the
  authoritative unprofiled baseline, so verdicts use shares/shape rather
  than substituting these absolute times into the baseline ladder.
- CUDA events measure inclusive GPU intervals. The linear range is made
  exclusive of its nested gate before residual accounting.
- H4 and the hybrid exponent use two successful points because N=294 and
  N=311 OOM. N=345 hybrid speedup is explicitly extrapolated.
- The 3.57% comparison audits the released checkpoint's exact frame-pair
  mask. The paper may define density using an unstated denominator.
