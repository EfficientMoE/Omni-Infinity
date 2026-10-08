# P3 — Attention backend bake-off for the window-softmax branch (SM120)

Tracking: [#42](https://github.com/EfficientMoE/Omni-Infinity/issues/42) (P3).
Refs: [t2v survey](../../t2v_optimization_survey.md) §2, [repro_vdn](../../repro_vdn.md).

## Objective

Pick the fastest parity-passing backend for VDN's window-softmax branch on
sm120, replacing the current `auto→decomposed` default, and fix the sm100
misdetection logged by the upstream component (physical CC is 12.0).

## Candidates

| Backend | Why | Integration |
|---|---|---|
| cuDNN 9.18+ SDPA | native sm120 paths, d>128 via 2-CTA MMA; ~1.1–1.3× | `sdpa_kernel([CUDNN])` ctx around the varlen path |
| FA4-fwd CuTe-DSL | merged for sm120; diffusion needs no paged/split-KV | `flash_attn.cute.flash_attn_func(causal=False)` |
| FlashInfer fmha_v2 | 10–31 % over FA2 at S≥512 | flashinfer wrapper, non-causal |
| SageAttention2 (INT8 QK) | 1.2–1.5× training-free; Triton on sm120 | SDPA monkey-patch; accuracy-gated |
| decomposed (current) | baseline | — |
| flex (Triton) | existing alt | — |
| Dispatch-pattern donor | BatchGen `v4_flashmla_adapter.py` env-gated backend selection + probe/dump | copy pattern, not kernel |

## Approach

1. **Fix detection first.** Pin `softmax_backend` dispatch on
   `torch.cuda.get_device_capability()` (12,x) explicitly in the VDN runner
   config plumbing (`omni_infinity/arch/vdn.py` → third_party dispatch);
   log the resolved backend per run. The "sm100 INFERENCE… STATIC
   block-sparse" warning must disappear or be explained in repro docs.
2. **Microbench matrix** (standalone script in `benchmarks/`): backends ×
   seqlen {8k, 20k, 37k} × head-dim (H3 geometry) × dense/window mask,
   CUDA-event timed, plus max-abs/rms error vs fp32 reference.
3. **Integration bench**: top 2 backends wired as `softmax_backend` values;
   full VDN ladder rerun (per-NFE s, goldens). SageAttention2 runs as an
   additional opt-in (`sage-attn`) with its own accuracy tier — it will NOT
   be bitwise; document rms_rel like the FP8 gate does.
4. **Decision**: new `auto` resolution order for CC 12.x recorded in
   repro_vdn.md + registry default updated.

## Tasks

- [ ] Capability-pinned dispatch + backend logging
- [ ] `benchmarks/attn_bakeoff.py` matrix runner (bounded runtime)
- [ ] cuDNN / FA4 / fmha_v2 adapters behind existing backend enum
- [ ] SageAttention2 opt-in + accuracy tier doc
- [ ] VDN ladder rerun; update ablation/attribution docs + issue #42

## Verification

Parity: window branch output allclose vs decomposed reference (document
tier); end-to-end goldens for the chosen default must keep current parity
class. Perf: per-NFE table appended to `docs/ablation_vdn.md`.

## Risks

- FA4 on sm120 ≈ FA2-class (`mma.sync`) — may not beat cuDNN; that is an
  acceptable negative result, record it.
- Window mask may not map to fmha_v2/cuDNN primitives → fall back to the
  decomposed-mask formulation per backend (same union-of-dense trick).
- SageAttention quality on H3 unknown → strictly opt-in, calibrated like C5.
