# VDN-Minimax-H3 Reproduction on RTX PRO 6000 Blackwell (sm120)

Reproduction of the [OpenVDN/vdn-minimax-h3](https://github.com/OpenVDN/vdn-minimax-h3)
(commit `e9204ce`, pinned as `third_party/vdn-minimax-h3`) single-GPU
results, per the plan tracked in
[#10](https://github.com/EfficientMoE/Omni-Infinity/issues/10).
Raw artifacts: `results/vdn/repro/` (gitignored; per-run
`*.inference.json`, `*.gpu_mem.log`, `*.render.log`).

## Environment

| | |
|---|---|
| GPU (bench) | NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB, **CC 12.0 (sm120)** |
| Kernel dispatch class | "consumer Blackwell" per upstream: FlexAttention Triton / PyTorch varlen+SDPA; **no FlashAttention-4** (`flash_available: null` confirmed in run records) |
| Env | python 3.12.3, torch 2.13.0+cu129, torchvision 0.28.0+cu129, patched diffusers `0f58fbb2c` (base `3a2f35d4e` + AdaLN-SiLU-fp32 pin), torchao 0.18.0, no flash-attn |
| Weights | HF `OpenVDN/vdn-minimax-h3` (82 GB local; MiniMax H3 Community License) |
| Prompt | `prompts/example_0.pt` (upstream-shipped T2VA cache), seed 42 (config default), `render.warmup_steps=2` |

## Deviations from upstream conditions

1. **Frame count N=222, not 345.** The plan assumed dense is the
   memory-binding rung; empirically the **untuned bf16 hybrid (V0) is
   the ceiling**: it OOMs at 345f (`linear_attention/scan.py::
   _frame_stats_prep_body`, needed 1.35 GiB with 714 MiB free) because
   it carries the full backbone *plus* linear-branch scan transients,
   while dense fits at 345f (peak 93,567 MiB). For cross-rung
   comparability all five rungs ran at N=222. A dense 345f run is
   preserved (`dense_bf16_345f.*`): **58.56 s/NFE** — dense scales
   super-linearly with sequence (28.50 → 58.56 for 222 → 345 frames).
2. **s/NFE is not comparable in absolute terms** to upstream's
   H200/B200 tables (different silicon, no FA4, different frame
   count). The reproduction criterion is the **speedup ratio**.
3. Rungs D/V0 pin `kernels.softmax_backend=flex`; V1/V2/S50 use the
   config default `auto` → resolves to `decomposed` (upstream's
   shipped default), confirmed in the run records.

## Table 1 — upstream native stack, single GPU, 8 NFE, N=222

| rung | config / checkpoint | s/NFE | speedup vs dense | peak MiB | notes |
|---|---|---:|---:|---:|---|
| D — dense MiniMax-H3 | `8nfe.yaml`, `checkpoint=null` | 28.50 | 1.00× | 88,999 | full softmax; `fp8_linears=0` |
| V0 — hybrid bf16 | `8nfe.yaml`, dmd-250 | 18.99 | 1.50× | 92,307 | flex; 571 LoRA pairs merged |
| V1 — + tuned kernels | `8nfe_tuned.yaml`, dmd-250 | 16.38 | 1.74× | 93,903 | decomposed |
| V2 — + fp8 | `8nfe_tuned_fp8.yaml`, dmd-250 | 13.28 | **2.15×** | 93,891 | 363 fp8 Linears |
| S50 — 50-NFE fp8 | `50nfe_tuned_fp8.yaml`, b-2000 | 13.29 | 2.14× (per-NFE) | 94,903 | 208 LoRA pairs; 50 steps |

Upstream single-GPU reference (345f, 768p): H200 dense 32.7 → VDN fp8
11.2 (**2.92×**); B200 dense 16.74 → 6.41 (**2.61×**).

### Headline result

**dense ÷ (hybrid+kernels+fp8) = 28.50 / 13.28 = 2.15×** — *below* the
2.3–3.2× band bracketing upstream's ratios. Decomposition of the gap:

- hybrid attention alone: 1.50× (D→V0)
- - tuned inference kernels: 1.16× (V0→V1)
- - fp8: 1.23× (V1→V2)

The fp8 step is the compressed one (upstream implies ~1.7–1.9× from
their kernels+fp8 tiers). Plausible sm120 causes: fp8 GEMM scale
granularity/kernel selection differs from H200/B200 fast paths, and
the sm120 dense baseline is comparatively cheap (no FA4 on either
side of the ratio). Per-NFE timing variance was <0.3% within every
run; the 2.15× is a stable measurement, not noise. Follow-up signal
lives in the ablation grid (backend + precision axes).

Verification also confirmed: 50-NFE and 8-NFE fp8 rungs run at
identical per-step cost (13.28 vs 13.29 s/NFE) — step count does not
change per-step cost, as expected.

## Table 2 — diffusers/Omni-Infinity path (`VdnRunner` via
`examples/vdn_smoke.py`), 8 evals, N=222, GPU 1

Wall-clock here **includes** per-run overhead (component load already
excluded; first-eval compile amortized differently than the native
stack's discarded warmup), so rows compare against each other, not
against Table 1.

| row | flags | s/eval (incl. overhead) | peak GiB | parity |
|---|---|---:|---:|---|
| resident bf16 | `--offload --stream-text-encoder` | 20.69 | 85.80 | reference |
| resident fp8 | + `--fp8` | 18.26 | 65.21 | n/a (fp8 changes the sample) |
| streamed bf16 | + `--block-stream 1` | 21.54 | 19.87 | bitwise vs goldens (B5) |
| streamed fp8 | + `--block-stream 1 --fp8` | 19.05 | 19.87 | n/a |

Golden-latent gate: `tests/test_vdn_parity.py` (bf16,
streamed-encoder recipe, 120f/8 evals) — recorded and replayed
bitwise on the Server-Edition card (`rms_rel=0.0000`), plus B5
re-verification with transformer block streaming enabled
(`rms_rel=0.0000` — device-only bf16 moves are bitwise-safe).
fp8 halves the resident peak (85.8 -> 65.2 GiB); block streaming
collapses it to ~20 GiB at a ~4% s/eval cost (bf16), matching the
upstream H200 table's shape.

Upstream diffusers-path reference (H200, 345f): resident bf16
14.3 s/eval; fp8 12.1; streamed bf16 16.2 @ 22 GB; streamed fp8
13.1 @ 20 GB.

## Environment gotcha worth keeping

The shared machine cache `HF_HOME=/mnt/raid0nvme0/public/huggingface`
is not writable by this user; the VDN modular index resolves
sub-components by **Hub repo id** (`MiniMaxAI/MiniMax-H3` for
encoder/VAEs/schedulers), so `load_components` requires a writable
HF cache. All VdnRunner invocations must export
`HF_HOME=/mnt/raid0nvme0/leyang/.cache/huggingface` (pre-populated;
the H3 `transformer/` subtree is excluded — the VDN path never reads
it). The same root cause affects the pre-existing H3 parity test on
this machine.

## Second environment gotcha: cross-repo `trust_remote_code`

The vendored diffusers (`modular_pipeline.py:2456`) strips
`trust_remote_code` from any component whose repo differs from the
pipeline's. The VDN `transformer` is remote-code and declares repo
`OpenVDN/vdn-minimax-h3`, so loading the pipeline from a **local
directory** always fails (`trust_remote_code ... not forwarded ...
different repository`). Consequence: `VdnRunner` checkpoints must be
the Hub repo-id `OpenVDN/vdn-minimax-h3` (resolved offline from the
pre-populated `HF_HOME` cache) — never a local path. Verified
weight-identical (same snapshot; goldens replay bitwise).

Note: the upstream component log prints an `sm100 INFERENCE ... flex
is running the STATIC block-sparse schedule` warning on this card
(physical CC is 12.0); absolute s/eval on the diffusers path is
therefore pessimistic vs an FA-enabled build. Row ordering unaffected.
