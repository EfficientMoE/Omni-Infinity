# VDN-H3 baseline tracking

Research record (2026-10-08 snapshot). Tracks **VDN-Minimax-H3** as the
strongest baseline for this stack, records the MiniMax-H3 reference
numbers it inherits, and surveys every model that currently *claims* a
stronger result than MiniMax H3 on video-generation performance or
accuracy (GitHub, arXiv, and arena leaderboards). Companion docs:
[reproduction](repro_vdn.md), [ablation](ablation_vdn.md),
[attribution](attribution_vdn.md). Tracking issue:
[#10](https://github.com/EfficientMoE/Omni-Infinity/issues/10).

> **Verification caveat.** Collected by automated research passes.
> Arena Elo values are snapshot-dependent and moved between the
> September and October 2026 snapshots; benchmark figures are leads,
> not gospel — verify against the live leaderboards before citing.
> Items marked ⚠ are single-sourced.

## 1. Strongest tracked baseline: VDN-Minimax-H3

VDN-H3 ("Video DeltaNet on MiniMax H3") is the strongest baseline we
track: the fastest open-weights model on the H3 backbone with
near-lossless quality versus dense H3, and the only challenger that is
locally runnable under this repo's parity gates (registry arch
`vdn-hybrid`, pinned at `third_party/vdn-minimax-h3`).

| Field | Value |
|---|---|
| Full name | Video DeltaNet on MiniMax H3 (VDN-H3) |
| Organization | OpenVDN (UC Berkeley, Impossible Inc., UT Austin) |
| Released | 2026-09-06 (code/weights), 2026-09-17 (paper) |
| Paper | [arXiv:2609.20744](https://arxiv.org/abs/2609.20744) |
| Code | [OpenVDN/vdn-minimax-h3](https://github.com/OpenVDN/vdn-minimax-h3) |
| Weights | [HF: OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3) |
| License | MiniMax H3 Community License (weights), Apache-2.0 (code) |
| Workflows | T2VA, I2VA, L2VA, FL2VA |

Published performance (768p, 14.4 s generation):

| Configuration | Denoise time | Speedup vs dense H3 |
|---|---|---|
| 8-step, 8×B200 | 6.9 s | 14.5× |
| 8-step, 8×H200 | 12.5 s | 10.7× |
| 50-step, 1×B200 | 307.9 s | 2.6× (dense: 799.6 s) |

Published quality (8-step VDN-H3 vs 50-step dense H3, from the paper):
FIRM-Video instruction following 2.25 vs 2.25, perceptual quality 4.45
vs 4.40, world coherence 1.77 vs 1.84; RAFT motion magnitude 11.71 px
vs 11.55 px — i.e. parity within noise across the published metrics.

Local tracking in this repo (RTX PRO 6000 Blackwell, sm120):
dense→hybrid+fp8 **2.15×** per-NFE with bitwise golden parity
(`tests/test_vdn_parity.py`), block-streamed to ~20 GiB peak —
see [repro_vdn.md](repro_vdn.md) and the 16-run grid in
[ablation_vdn.md](ablation_vdn.md).

## 2. Reference point: MiniMax H3

The dense backbone (`h3-dense`, `MiniMaxAI/MiniMax-H3`) that both
VDN-H3 and the challengers below measure against. Released 2026-07-31
([announcement](https://www.minimax.io/blog/minimax-h3),
[open-source release](https://www.minimax.io/news/minimax-h3-open-source)).
Native 2K at 24 fps, 4–15 s, stereo audio, omni-reference input
(up to 9 images + 3 videos + 3 audio clips).

Artificial Analysis Video Arena position (Sep–Oct 2026 snapshots):

| Arena | Elo | Rank |
|---|---|---|
| Video editing (v1.1) | 1302 | #1 |
| T2V silent (v2.0) | 1301 | #2 (behind Gemini Omni Flash, 1324) |
| T2V with audio (v2.0, 768p) | 1137 | #4 |
| I2V (v1.0) | 1181–1195 ⚠ | top tier |

H3 post-dates VBench-2.0 (2025-03) and is not on that leaderboard; its
predecessor Hailuo 02 scored ~83.4 on original VBench ⚠ and held the
WorldModelBench "physics champion" title that the Hailuo 2.3 → H3 line
inherits.

## 3. Claimed-stronger challengers (search results, 2026-10)

Models claiming a stronger result than MiniMax H3 on video-generation
performance or accuracy, ranked by evidence strength. "Third-party" =
arena/leaderboard or independent paper; "vendor" = self-published.

### 3.1 Third-party-verified claims

| Model | Claim vs H3 | Numbers | Evidence | Source |
|---|---|---|---|---|
| **Wan 3.0** (Alibaba) | #1 T2V with audio | Elo 1156–1157 vs H3's 1137 | third-party | [AA T2V leaderboard](https://artificialanalysis.ai/video/leaderboard/text-to-video) |
| **HappyHorse-1.0** (Alibaba-ATH) ⚠ | #1 T2V and I2V overall | Elo 1361 (T2V), 1398 (I2V); limited public access | third-party | [leaderboard mirror](https://awesomeagents.ai/leaderboards/video-generation-benchmarks-leaderboard/) |
| **Gemini Omni Flash** (Google) | #1 T2V silent | Elo 1324 vs H3's 1301 (CI non-overlapping) | third-party | [AA snapshot](https://www.edgechat.ai/artificial-analysis-text-to-video-leaderboard) |
| **Seedance 2.5** (ByteDance) | above H3 on T2V+audio, strong I2V | Elo 1143 (T2V), 1347 (I2V) | third-party | [AA T2V leaderboard](https://artificialanalysis.ai/video/leaderboard/text-to-video) |
| **Kling 3.0** (Kuaishou) | top-tier T2V, native 4K | Elo ~1100–1247 ⚠ (snapshot-dependent) | third-party | [AA T2V leaderboard](https://artificialanalysis.ai/video/leaderboard/text-to-video) |
| **Veo 3.1** (Google DeepMind) | stronger physics, native audio | Elo ~1100–1217 ⚠ (snapshot-dependent) | third-party | [AA T2V leaderboard](https://artificialanalysis.ai/video/leaderboard/text-to-video) |

### 3.2 Open-source challengers (GitHub / arXiv)

| Model | Claim | Numbers | Source |
|---|---|---|---|
| **Alice v1** ⚠ | surpasses Veo 3 (~90) and Sora 2 (~88) on VBench | VBench 91.2 | [arXiv:2605.08115](https://arxiv.org/html/2605.08115v1) |
| **Wan 2.2** (Alibaba) | highest open-source VBench | VBench 84.7 | [leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) |
| **HunyuanVideo 1.5** (Tencent) | SOTA among open-source at 8.3B | VBench 83.43 (≈ H3-class ⚠) | [arXiv:2511.18870](https://arxiv.org/html/2511.18870) |
| **LTX-2.3** (Lightricks) ⚠ | fastest open-source with native audio | no quantitative H3 comparison | vendor |

### 3.3 Efficiency-method papers (VDN-H3's own competitor class)

Few-step / causal methods claiming SOTA on the same quality-per-NFE
axis VDN-H3 occupies — candidates for a future stronger baseline on
this stack:

| Method | Claim | Numbers | Source |
|---|---|---|---|
| **ViRDM** | beats best few-step causal baseline by +0.36 | VBench 84.87 vs 84.51 | [arXiv:2609.28923](https://arxiv.org/abs/2609.28923) |
| **Causal Forcing++** | 2-step surpasses 4-step Causal Forcing | VBench 84.14 vs 84.04 | [arXiv:2605.15141](https://arxiv.org/html/2605.15141v3) |
| **HiAR** | best overall + lowest temporal drift at 20 s | VBench 0.821 vs 0.810 | [arXiv:2603.08703](https://arxiv.org/html/2603.08703v1) |
| **VideoAR** | SOTA among autoregressive models | VBench 81.74 | [arXiv:2601.05966](https://arxiv.org/html/2601.05966) |

### 3.4 Vendor claims without decisive third-party numbers

- **Runway Gen-4.5** — "world consistency" + physics realism; peaked at
  Elo ~1247 (2025-12) but slid to ~#9 by 2026-04 ⚠
  ([vendor](https://runway.com/research/introducing-runway-gen-4)).
- **Luma Ray3** — "state-of-the-art motion realism", HDR; vendor-funded
  eval report only, no arena Elo
  ([report](https://static.cdn-luma.com/files/495b217627858402/Ray3_Evaluation_Report___State_of_the_Art_Performance_for_pro_Video_Generation.pdf)).
- **Sora 2** (OpenAI) — "improved physics simulation"; VBench ~88
  (estimate from the Alice v1 paper ⚠); API discontinued 2026-09.

## 4. Verdict for this repo

- **VDN-H3 remains the strongest *tracked* baseline.** Every
  challenger above H3 on the arenas is closed/API-only (Wan 3.0,
  HappyHorse, Gemini Omni Flash, Seedance, Kling, Veo) or on a
  different backbone with no local-parity story. None are runnable
  under this repo's bitwise parity gates or the ≤22 GiB envelope.
- **H3 itself is no longer the undisputed arena leader** (Wan 3.0 and
  HappyHorse-1.0 lead the 2026-10 T2V snapshots), but it still holds
  #1 video editing and #2 silent T2V — a strong accuracy reference.
- **Watch list for a stronger tracked baseline:** an open-weights
  release of Wan 3.0; ViRDM or Causal Forcing++ checkpoints on an
  H3-class backbone; the pending MiniMax H3 technical report and any
  H3.5/H4 open release.

Re-run the challenger search when refreshing this record: query
GitHub/arXiv for claims against "Hailuo"/"MiniMax H3" and diff the
[AA leaderboards](https://artificialanalysis.ai/video/leaderboard/text-to-video)
and the
[VBench leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)
against the tables above.
