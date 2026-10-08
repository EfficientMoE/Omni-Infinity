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

## Window-softmax backend bake-off (sm120, issue #42 P3)

`benchmarks/attn_bakeoff.py` times the window branch in isolation at H3
geometry (56 heads x 128 head-dim, 1008 tokens/frame + 512 global rows,
chunk=5 radius=1 anchors=both), CUDA-event medians over 20 iters on one
RTX PRO 6000 Blackwell (CC 12.0), bf16, errors vs the fp32 eager
reference. `cudnn` and `fmha-v2` run the same union-of-dense
decomposition with the kernel swapped under the legs
(`omni_infinity/arch/vdn_attention.py`); `fa4` has no sm120 install
(flash-attn ships no consumer-Blackwell cute kernels) — recorded as
unavailable rather than benched.

| backend | window 8.6k (ms) | window 20.7k | window 38.8k | rms_rel class |
|---|---:|---:|---:|---|
| decomposed (auto today) | 6.98 | 27.81 | 60.52 | 2.3e-03 (bf16) |
| flex | 7.01 | 30.55 | 66.78 | 2.3e-03 (bf16) |
| **cudnn** | **6.53** | **27.02** | **59.24** | 2.3e-03 (bf16) |
| fmha-v2 (FlashInfer 0.6.18) | 6.68 | 27.47 | 59.46 | 2.3e-03 (bf16) |
| sage (INT8 QK) | 7.72 | 29.60 | 62.85 | 1.3e-02 (INT8 tier) |

Full matrix (dense-mask rows included): `results/attn_bakeoff/results.csv`.
`cudnn` leads at every size (2.1-6.4 % over `decomposed` on the window
branch), `fmha-v2` second; both sit in the same bf16 reduction-order
parity class as `decomposed` itself (2.3e-03 vs fp32). `sage` is slower
AND INT8-tier — a negative result; it stays the accuracy-gated
`sage-attn` registry opt-in. The gains are branch-local: scale by the
window share of step time from the attribution study before expecting
end-to-end movement.

### Integration rung (vdn_smoke, 8 NFE, 120 frames, offload + text-encoder stream, GPU 4)

| backend | wall s | s/eval incl. overhead | peak GiB | rms_rel vs goldens |
|---|---:|---:|---:|---:|
| main @83d1950 (implicit = spec flex) | 133.6 | 16.70 | 83.4 | 0.288* |
| decomposed (pinned) | 127.8 | 15.97 | 81.8 | 0.306* |
| cudnn | 168.5 | 21.06 | 81.8 | 0.409* |
| fmha-v2 | 120.5 | 15.06 | 82.0 | 0.422* |

\* the unmodified main baseline fails these goldens by the same class
(0.288), so the committed `vdn_goldens.pt` does not match this
offload + text-encoder-stream environment — the column compares
backends relatively, not against a valid bitwise reference. The main
row also ran a different kernel: the hub spec pins
`softmax_backend: flex`, so the pre-P3 implicit default was flex (its
slower wall and distinct rms_rel are consistent with the microbench
flex rows). Decision:
**`auto` on CC 12.x stays pinned to `decomposed`.** `cudnn`'s 2-6 %
microbench win inverts at model scale (the per-chunk dense loop pays
~50 layers x ~num-chunk launches per eval where `decomposed` makes two
batched calls), a negative result the plan anticipated. `fmha-v2` is
the runner-up (-0.9 s/eval wall, within config noise) and the adapter
stays available behind `--softmax-backend fmha-v2` for a future
re-measure once the goldens are re-recorded for this config.
