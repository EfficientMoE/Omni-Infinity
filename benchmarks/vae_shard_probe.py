# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""P7 Phase 2 probe: VAE decoder spatial-tiling seam-exactness.

The MiniMax-H3 video decoder is a global-attention ViT
(`MiniMaxH3VideoViTDecoder3d`), so a halo-exchange spatial shard cannot
be seam-exact: every output token attends to every input token. The
decoder's own memory lever is instead overlap-tiling with *blended*
seams. This probe quantifies that blend as a seam-exactness test:
decode the committed golden video latents single-shot (tiling disabled)
versus with spatial tiling forced on (the tile threshold is lowered so
the 256-px frame splits into blended tiles), and report the pixel
rms_rel / max-abs between the two. A nonzero gap is the seam the blend
leaves behind — i.e. the native tiling is a memory tool, not an exact
spatial shard. (At the default threshold a 256-px decode is a single
tile, which is why the committed goldens match bitwise.)

Run:
  CUDA_VISIBLE_DEVICES=<idle> HF_HOME=... HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 OMNI_H3_CHECKPOINT=<snap> OMNI_H3_STORE=<store> \
  PYTHONPATH=. python benchmarks/vae_shard_probe.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

GOLDENS = Path("tests/fixtures/goldens/fl2va_goldens.pt")


def _decode(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return vae.decode(latents, return_dict=False)[0].float()


def main() -> int:
    from omni_infinity.runner import ReferenceRunner

    payload = torch.load(GOLDENS, weights_only=False)
    runner = ReferenceRunner.from_pretrained(
        os.environ["OMNI_H3_CHECKPOINT"],
        device="cuda:0",
        offload=False,
        components=("vae",),
        store_dir=os.environ.get("OMNI_H3_STORE"),
        store_components=("vae",),
    )
    vae = runner.pipeline.vae
    mean = torch.tensor(vae.config.latents_mean, device="cuda:0").view(
        1, -1, 1, 1, 1
    )
    std = torch.tensor(vae.config.latents_std, device="cuda:0").view(
        1, -1, 1, 1, 1
    )
    latents = payload["latents"].to("cuda:0") * std + mean

    vae.disable_tiling()
    reference = _decode(vae, latents)
    vae.enable_tiling(
        tile_sample_min_height=128,
        tile_sample_min_width=128,
        tile_sample_min_overlap_height=32,
        tile_sample_min_overlap_width=32,
    )
    tiled = _decode(vae, latents)

    diff = (tiled - reference).flatten()
    denom = torch.linalg.vector_norm(reference.flatten()).clamp_min(1e-12)
    rms_rel = (torch.linalg.vector_norm(diff) / denom).item()
    max_abs = diff.abs().max().item()
    print(f"  decode output shape: {tuple(reference.shape)}")
    print(
        f"  tiled vs single-shot: rms_rel={rms_rel:.3e} "
        f"max_abs={max_abs:.3e} bitwise={torch.equal(tiled, reference)}"
    )
    print(
        "  nonzero => native overlap-tiling blends (NOT seam-exact); a "
        "halo-exchange shard of this global-attention ViT cannot be exact."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
