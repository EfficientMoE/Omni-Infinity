# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Lightricks LTX-2.5 runner scaffold (model-arch: ltx-2.5).

**Status: WIP scaffold for issue #54 -- not yet runnable.** This module
wires an ``ltx-2.5`` arch name into the registry and pins down the runner
surface (``from_pretrained`` / ``generate``) so the integration can land
incrementally. Every loading/generation entry point raises
``NotImplementedError``; nothing here loads weights or runs inference yet.

LTX-2.5 (`Lightricks/LTX-2.5`) is a 22B joint audio-video DiT with open
weights under the LTX-2.x Community License (not Apache-2.0). It differs
from the H3 family this repo currently serves in ways that make the
existing ``ReferenceRunner`` / ``VdnRunner`` contracts insufficient:

- **Distilled 22B DiT** (fixed 8-step, CFG = 1) plus a full/dev DiT, and a
  distilled LoRA for dev-transformer workflows.
- **Gemma-4-12B text encoder** with projections -- not Qwen3-VL.
- **Two video VAEs** (DiffVAE + a lighter Conv VAE), an **audio VAE +
  vocoder**, and a **diffusion video decoder** that replaces the VAE
  reconstruction stage.
- **Two-stage distilled generation** (stage-1 denoise -> upsample ->
  stage-2), a Comfy-aligned split checkpoint (one ``.safetensors`` per
  component), not a single diffusers repo id.
- Diffusers support (``LTX2ImageToVideoPipeline`` /
  ``LTX2LatentUpsamplePipeline``) lives on diffusers ``main`` only; a
  Diffusers-friendly repack is at ``Lightricks/LTX-2.5-Diffusers`` and the
  vendor runtime is ``ltx-pipelines`` (``ModelPaths.from_split``).

The phased integration plan lives in
[docs/ltx2_integration.md](../../docs/ltx2_integration.md); the tracking
issue is #54.
"""

from __future__ import annotations

from typing import Any

import torch

from omni_infinity.runner import GenerationResult

_ISSUE = "https://github.com/EfficientMoE/Omni-Infinity/issues/54"
_DEFAULT_STEPS = 8


def _not_implemented(what: str) -> NotImplementedError:
    return NotImplementedError(
        f"LTX-2.5 {what} is not implemented yet; this is a WIP scaffold "
        f"for the arch registration only. Track and contribute at {_ISSUE} "
        f"(plan: docs/ltx2_integration.md)."
    )


class Ltx2Runner:
    """Scaffold runner for Lightricks LTX-2.5 (arch ``ltx-2.5``).

    Mirrors the ``from_pretrained`` / ``generate`` surface of
    :class:`omni_infinity.runner.ReferenceRunner` and
    :class:`omni_infinity.arch.vdn.VdnRunner` so call sites and the
    registry can reference it today. Both entry points raise
    ``NotImplementedError`` until the integration (issue #54) lands.
    """

    def __init__(
        self,
        pipeline: Any = None,
        *,
        default_steps: int = _DEFAULT_STEPS,
    ):
        self.pipeline = pipeline
        self.default_steps = default_steps

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "Lightricks/LTX-2.5",
        *,
        variant: str = "distilled",
        workflow: str = "t2va",
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        offload: bool = False,
        **kwargs: Any,
    ) -> "Ltx2Runner":
        """Load LTX-2.5 (not implemented -- scaffold only).

        The real implementation must assemble the split checkpoint
        (``ModelPaths.from_split`` / ``Lightricks/LTX-2.5-Diffusers``),
        the Gemma-4-12B text encoder, both video VAEs, the audio VAE +
        vocoder, and the diffusion video decoder. See issue #54.
        """
        raise _not_implemented("checkpoint loading")

    def generate(
        self,
        prompt: str,
        *,
        seed: int = 0,
        num_inference_steps: int | None = None,
        num_frames: int | None = None,
        output_type: str = "np",
        image: Any = None,
        last_image: Any = None,
        step_callback: Any = None,
        denoise_cache: Any = None,
    ) -> GenerationResult:
        """Run two-stage LTX-2.5 generation (not implemented -- scaffold)."""
        raise _not_implemented("generation")
