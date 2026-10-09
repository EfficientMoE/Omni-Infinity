# VDN-Minimax-H3 Ablation Study (sm120, single GPU, N=222, 8 NFE unless noted)

Grid: `benchmarks/ablation_vdn.py` (16 runs; CSV in
`results/vdn/ablation/results.csv`, gitignored). Companion to
[repro_vdn.md](repro_vdn.md); tracked in
[#10](https://github.com/EfficientMoE/Omni-Infinity/issues/10).
Quality columns: `rms_rel`/`cosine` of saved video latents vs run #2
(`hybrid-8nfe`, the bf16 hybrid reference). fp8 rows are *indicative*
(fp8 legitimately changes the sample). Omni rows report wall-clock
s/eval incl. overhead (no discarded-warmup mechanism), so they compare
within the omni block only.

## Results

| # | run | s/NFE | peak GiB | rms_rel vs #2 | cosine | axis |
|---|---|---:|---:|---:|---:|---|
| 1 | dense-8nfe | 28.51 | 86.9 | 0.728 | 0.744 | arch |
| 2 | hybrid-8nfe | 19.00 | 91.7 | 0 (ref) | 1.000 | arch |
| 3 | dense-turbo-lora-8nfe | 28.50 | 86.7 | 0.678 | 0.768 | arch/lora |
| 4 | hybrid-50ckpt-50nfe | 19.00 | 91.4 | 0.650 | 0.790 | nfe |
| 5 | hybrid-50ckpt-8nfe | 18.99 | 94.4 | 0.694 | 0.765 | nfe/lora |
| 6 | hybrid-8nfe-tuned | 16.37 | 91.7 | 0.318 | 0.950 | kernels |
| 7 | hybrid-8nfe-tuned-fp8 | 13.28 | 91.7 | 0.472* | 0.889* | precision |
| 8 | hybrid-8nfe-tuned-fp8-skip4 | 13.77 | 91.7 | 0.434* | 0.907* | precision |
| 9 | hybrid-8nfe-flex | 16.70 | 94.3 | 0.292 | 0.958 | backend |
| 10 | hybrid-8nfe-decomposed | 16.37 | 91.7 | 0.318 | 0.950 | backend |
| 11 | omni-resident-bf16 | **OOM** | >94.97 | — | — | memory |
| 12 | omni-stream-bf16 | 20.54 | 19.87 | — | — | memory |
| 13 | omni-stream-fp8 | 17.52 | 19.87 | — | — | memory/precision |
| 14 | omni-stream-groups4 | 20.17 | 19.87 | — | — | memory |
| 15 | omni-50step-bf16 | 17.56 | 19.89 | — | — | nfe |
| 16 | omni-notes-textenc | 20.48 | 82.00 | — | — | memory |

\* indicative (different sample by design). Omni rows re-collected
after the goldens-gate harness fix (`43a3f97`).

## Per-axis conclusions

### 1. Model arch (#1 vs #2 vs #3)
Hybrid attention alone buys **1.50×** (28.51 → 19.00 s/NFE). The
control run #3 (dense + community turbo LoRA, no linear branch) runs
at **dense speed** (28.50) — LoRA distillation contributes *zero*
throughput; **all** of VDN's speedup at equal NFE comes from the
hybrid attention structure. (The turbo LoRA's role is enabling 8-NFE
quality, not per-step speed.)

### 2. NFE / turbo adapter (#4, #5 vs #2)
Per-step cost is NFE-independent (18.99–19.00 across 8 and 50 steps).
End-to-end, the 8-step DMD checkpoint is 6.25× cheaper than the
50-step one. Run #5 (50-step ckpt forced to 8 NFE, no turbo adapter)
diverges hard from the 8-step reference (rms_rel 0.694, cosine 0.765)
— the expected quality collapse: the turbo/DMD adapter is what makes
8-NFE trajectories valid; NFE count cannot simply be dialed down.

### 3. Inference kernels (#2 vs #6) and window-softmax backend (#9 vs #10)
Tuned forward-only kernels: **1.16×** (19.00 → 16.37), with a modest
latent shift (rms 0.318, cosine 0.950 — kernel-rounding scale, same
sample). Backend on sm120: `decomposed` (PyTorch varlen/SDPA) beats
`flex` (FlexAttention Triton static schedule) by **2%** (16.37 vs
16.70) and uses ~2.6 GiB less peak — upstream's `auto`=`decomposed`
default is correct on this card too. New data: upstream only reports
H200/B200 for this axis.

### 4. Precision (#6 vs #7 vs #8)
fp8 on all 363 wide Linears: **1.23×** (16.37 → 13.28) at unchanged
~91.7 GiB peak (activation transients dominate at N=222; weight
savings surface in the *diffusers-path* resident peak instead: 85.8 →
65.2 GiB, see repro doc). `skip_end_blocks=4` (#8) costs 3.7% speed
and moves the latents *toward* bf16 (0.434 vs 0.472) — a small
quality/robustness dial. sm120's fp8 step (1.23×) remains the gap vs
upstream's H200/B200 ratios (~1.7×); see repro doc headline analysis.

### 5. Memory optimizations (#11–#16)
`omni-resident-bf16` **does not fit**: hard OOM at ~94 GiB working
set on the 95 GiB card (deterministic across two allocator configs) —
the fully-resident bf16 transformer at 222 frames is exactly the
configuration Omni-Infinity's streaming levers exist to solve. Block
streaming collapses peak to **~19.9 GiB (4.7× less)** at ≈5% s/eval
cost (bf16) and is **bitwise-parity-safe** (goldens gate +
streaming-parity check, `rms_rel=0.0000`). fp8 + streaming (#13) is
the speed/memory sweet spot: 17.52 s/eval @ 19.87 GiB. 50-eval
streaming (#15) confirms per-step cost invariance end-to-end on the
omni path too. #16 isolates text-encoder streaming: keeping the
Qwen3-VL encoder resident costs **+62 GiB peak** (19.87 -> 82.0)
for ~0% speed difference (20.48 vs 20.54 s/eval) - encoder
streaming is pure win. Streaming granularity (#14, 4 blocks/group)
is ~2% faster than 1 block/group at identical peak.

## Threats to validity

- Single GPU model (RTX PRO 6000 Blackwell Server, sm120), single
  prompt (`example_0.pt`), N=222 frames, seed 0/42 — relative
  orderings are robust (<0.3% within-run step variance) but absolute
  numbers are card-specific; no FA4 on this silicon.
- Upstream vs omni s/eval are not directly comparable (different
  warmup accounting).
- fp8 quality metrics are latent-level proxies vs a bf16 reference
  that is itself a different sample; they bound divergence, they do
  not measure perceptual quality.

## Harness incidents (documented for reproducibility)

1. Launch-shell `HF_HOME` defeated the harness's `env.setdefault` for
   omni rows → first omni attempt hit the unwritable shared cache.
   Run with `HF_HOME` unset (or the private cache) — see repro doc.
2. Grid omni rows originally passed the 120-frame goldens fixture to
   222-frame runs (temporal 37 vs 67 → hard shape error after
   successful generation). Fixed in `43a3f97`: grid runs don't take
   the goldens gate; parity is owned by `tests/test_vdn_parity.py`.

## P2 resident optimization ablation

This separate, filterable P2 suite uses `--suite p2` in the same harness. It
tests the `h3-dense` portable stack at 256p, 120 requested/124 effective
frames, seed 0, `tests/fixtures/ref.png`, and prompt “a red ball bouncing.”
Timing columns come from physical GPU 4, an RTX PRO 6000 Blackwell Max-Q
Workstation Edition, using the median CUDA step window after the first forward
for eager rows and steady replay windows for graph rows. Parity columns come
from separate NFE-8 runs on physical GPU 0, the RTX PRO 6000 Blackwell Server
Edition matching the golden provenance. Both used driver 590.48.01, PyTorch
2.12.0+cu130, and diffusers 0.40.0.

| config | NFE | s_per_nfe (ms) | peak_gib | rms_rel | cosine | parity tier | notes |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 8 | 1483.38 | 71.89 | 0 | 1.000004 | bitwise | canonical Phase-0 timing; reference |
| baseline | 16 | 1492.56 | 71.89 | — | — | no golden | fresh timing; no golden at this NFE |
| compile | 8 | 1579.38 | 71.89 | 0.06644 | 0.997807 | **FAIL 2e-2** | canonical Phase-1 timing; 0.939x, slower |
| compile | 16 | 1446.59 | 71.89 | — | — | no golden | fresh timing; no golden at this NFE |
| graph | 8 | **966.42** | 71.89 | 0 | 1.000004 | bitwise | 1.535x; 24.23 GiB host pinned; ≈2.5 s capture |
| graph | 16 | **965.41** | 71.89 | — | — | no golden | 11 steady replays; no golden at this NFE |
| compile+graph | 8 | 924.82 | 71.89 | 0.04114 | 0.999162 | **FAIL 2e-2** | capture succeeded; 2.67 s capture |
| compile+graph | 16 | 924.90 | 71.89 | — | — | no golden | capture succeeded; no golden at this NFE |

The NFE-8 baseline, compile, and graph timing cells deliberately report the
canonical Phase 0/1/2 records so the headline comparison retains identical
provenance. Fresh same-GPU rechecks are also preserved in their normalized
JSON artifacts: baseline was 1492.84 ms, compile was 1238.67 ms, and graph was
964.76 ms. The pageable AdaLN H2D portion of the eager rows varied materially
with host-memory state, while pinned graph replay remained stable; replacing
the canonical eager numbers with the favorable compile recheck would therefore
be misleading. The NFE-16 rows are fresh measurements and should be read with
that host-copy variability in mind. Exact records and the aggregate CSV are in
`results/p2_phase2_ablation/`.

**Verdict.** Graph-only is the single useful configuration: it preserves
bitwise golden parity and delivers the canonical 1.535x resident-profile win.
Compile-only remains a measured NFE-8 regression and fails the `2e-2` parity
gate. Compile+graph did capture on torch 2.12 and replayed about 4% faster than
graph-only, but it inherited compile's numerical failure, so that speed is not
usable under the golden contract. Readers with enough memory for the
~71.89 GiB resident profile and an additional 24.23 GiB of pinned host memory
should choose resident graph-only; none of these rows is a 22 GiB profile.
