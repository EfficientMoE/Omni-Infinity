# Omni-Infinity VDN-Minimax-H3 image

This image reproduces the Python 3.12 / CUDA 12.9 environment used for the
VDN-Minimax-H3 results. It pins OpenVDN's source, the patched Diffusers tree,
PyTorch and torchvision from the same cu129 index, torchao, and moe-store.
The CUDA *devel* base includes the headers and compiler required by JIT
kernels.

## Build

From the Omni-Infinity repository root:

```bash
docker build -t omni-infinity-vdn:latest -f docker/Dockerfile .
```

FlashAttention 4 is intentionally skipped for Ampere, Ada, and consumer
Blackwell (including sm120). Hopper and data-center Blackwell users can opt
in; this adds a long architecture-specific build layer:

```bash
docker build --build-arg INSTALL_FLASH_ATTN=1 \
  -t omni-infinity-vdn:fa4 -f docker/Dockerfile .
```

`VDN_REF` and `MOE_STORE_REF` are build arguments pinned to verified commits.
Override them only when deliberately testing a different revision.

## Unit tests

No model weights are needed:

```bash
docker run --rm omni-infinity-vdn:latest \
  python -m pytest tests/ -k "not parity"
```

## Model cache and license

The approximately 82 GB VDN checkpoint is **not** included in the image. It
is distributed under the MiniMax H3 Community License, not this repository's
Apache-2.0 code license. Review and accept that license before downloading or
using the weights.

The image sets `HF_HOME=/cache/huggingface`. Mount a writable host cache for
every Hub-backed run:

```bash
-v /path/to/host/huggingface:/cache/huggingface
```

For `VdnRunner`, the checkpoint must remain the Hub repo ID
`OpenVDN/vdn-minimax-h3`; a local checkpoint path fails the patched
Diffusers cross-repository `trust_remote_code` guard.

## Parity gate

Use an idle GPU (this example exposes host GPU 2) and a cache that already
contains the checkpoint:

```bash
docker run --rm --gpus '"device=2"' \
  -v /path/to/host/huggingface:/cache/huggingface \
  -e VDN_CHECKPOINT=OpenVDN/vdn-minimax-h3 \
  omni-infinity-vdn:latest \
  python -m pytest tests/test_vdn_parity.py -q
```

The committed goldens are GPU-model and PyTorch-version specific; the test
skips rather than comparing incompatible goldens.

## Streamed-bf16 smoke run

This recipe streams the text encoder and one transformer block at a time:

```bash
docker run --rm --gpus '"device=2"' \
  -v /path/to/host/huggingface:/cache/huggingface \
  omni-infinity-vdn:latest \
  python examples/vdn_smoke.py \
    --checkpoint OpenVDN/vdn-minimax-h3 \
    --prompt "a red ball bouncing" --seed 0 --evals 8 --frames 120 \
    --offload --stream-text-encoder --block-stream 1
```

## Upstream native ladder

The pinned upstream repository is `/opt/vdn-minimax-h3`. Native configs use
a local `ckpts/` tree rather than the Hugging Face cache, so mount an accepted
and downloaded VDN checkpoint layout there. For example:

```bash
docker run --rm --gpus '"device=2"' \
  -v /path/to/vdn-weights:/opt/vdn-minimax-h3/ckpts:ro \
  -v "$PWD/results/vdn:/output" \
  -w /opt/vdn-minimax-h3 omni-infinity-vdn:latest \
  python src/inference/infer.py \
    --config configs/inference/8nfe_tuned_fp8.yaml \
    checkpoint=ckpts/stage-dmd-step-250 \
    render.prompt_file=prompts/example_0.pt \
    render.out=/output/native-vdn.mp4 render.record=true
```

Change the config to `8nfe.yaml`, `8nfe_tuned.yaml`, or
`50nfe_tuned_fp8.yaml` (with the matching checkpoint directory) to run the
upstream native ladder. The first invocation can spend several minutes
compiling kernels; preserving a host cache for Triton/PyTorch compilation is
recommended for repeated runs.
