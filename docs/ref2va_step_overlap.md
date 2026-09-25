# Ref2VA store path and cross-step prefetch

Task 3 extends the Task-2 memory path to MiniMax-H3 Ref2VA and closes the
transformer weight-prefetch chain across denoising-step boundaries.

## FL2VA versus Ref2VA

FL2VA runs the `transformer` component and accepts optional first/last
keyframes. Ref2VA runs `transformer_ref` and accepts an ordered `references`
list of `MiniMaxH3ImageReference` objects. Diffusers normalizes each reference,
encodes it with Qwen3-VL and the visual VAE, and packs reference rows into the
single sequence `[text | reference blocks | target audio | target video]`.

The authoritative Diffusers loop remains unchanged. In
`minimax_h3/denoise.py`, lines 240–268 iterate scheduler timesteps, lines
134–139 select and invoke `transformer_ref`, and lines 220–237 apply the
scheduler update. Omni-Infinity changes weight placement and transfer timing,
not the arithmetic order.

## Cross-step overlap

Diffusers' native block-level group offload prefetches group N+1 while group N
computes within one transformer forward. Omni-Infinity preserves that chain
and adds only its missing wraparound edge: the final group in step N launches
the first-group H2D copy for step N+1.

Step 1 is warmup. Diffusers discovers the lazy hook order during that first
forward, and the first group may synchronously self-load once. The CUDA-event
gate therefore starts at destination step 2. It separately verifies that the
first-group copy originated in the preceding step and overlapped its final
group's compute, preventing native within-forward prefetch from producing a
false pass.

The adapter in `omni_infinity/step_overlap.py` depends on private Diffusers
hook APIs: `HookRegistry`, `_GROUP_OFFLOADING`, `GroupOffloadingHook`, and
`ModuleGroup`. It fails closed if the expected topology changes. The validated
stack uses **diffusers 0.40.0**.

## Golden provenance

`tests/fixtures/goldens/ref2va_goldens.pt` was recorded before optimized
execution with a full-resident bf16 `transformer_ref` and component-level CPU
offload. The run used torch **2.12.0+cu130**, diffusers **0.40.0**, an NVIDIA
RTX PRO 6000 Blackwell Server Edition, seed 0, 8 requested steps, 256p, and 120
requested frames (the H3 VAE rounds this to 124 internally). The committed
reference fixture SHA-256 is
`04f07c1bc50128a772dd14b95275e566a2d34635fe5856a7569383ad06a569f9`.

The planned `HF_HOME=/mnt/raid0nvme0/leyang/hf-home` snapshot was partial: it
did not contain `transformer_ref`. The Task-3 QA store was consequently rebuilt
from merged snapshot roots and is `/mnt/raid0nvme0/leyang/h3-store-v2`, not the
stale `/mnt/raid0nvme0/leyang/h3-store`. The v2 store contains 122 groups and
50 AdaLN groups for each of `transformer` and `transformer_ref`. Offline golden
recording used a local merged checkpoint root whose component specs point to
the complete snapshot; no site-packages or third-party source was modified.

## QA commands and measurements

Inspect the corrected store:

```bash
HF_HOME=/mnt/raid0nvme0/leyang/hf-home HF_HUB_OFFLINE=1 \
/home/leyang/anaconda3/bin/python - <<'PY'
from omni_infinity.store import StoreComponentSource
s = StoreComponentSource('/mnt/raid0nvme0/leyang/h3-store-v2')
for name in ('transformer', 'transformer_ref'):
    groups = s.groups_for(name)
    print(name, 'groups=', len(groups), 'adaln=', len(s.adaln_groups(name)))
    print([s.stage_name(group) for group in groups[:12]])
PY
```

Run optimized bitwise parity. `OMNI_H3_CHECKPOINT` must name a complete local
snapshot root in offline environments:

```bash
CUDA_VISIBLE_DEVICES="$OMNI_TASK3_GPU" \
OMNI_H3_CHECKPOINT="$OMNI_H3_CHECKPOINT" \
OMNI_H3_STORE=/mnt/raid0nvme0/leyang/h3-store-v2 \
OMNI_H3_STORE_COMPONENTS=transformer_ref,vae,audio_vae \
OMNI_H3_ADALN_CACHE=1 OMNI_H3_OFFLOAD=1 OMNI_H3_BLOCK_STREAM=1 \
OMNI_H3_STREAM_TEXT_ENCODER=1 OMNI_H3_STEP_OVERLAP=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/leyang/anaconda3/bin/python -m pytest tests/test_ref2va_parity.py -q -s -o addopts=
```

Result: **PASS**; both video and audio latents satisfy `torch.equal` against
the full-resident golden.

Run the issue's required smoke command verbatim:

```bash
python examples/ref2va_smoke.py --ref tests/fixtures/ref.png --seed 0 --steps 8 --resolution 256p --frames 120 --max-vram 22GiB
```

Result: exit 0, bitwise parity, **77.0 s** generation wall-clock, and **10.72
GiB** full-pipeline `max_memory_allocated` against the 22.00 GiB budget.

Run the required overlap command:

```bash
CUDA_VISIBLE_DEVICES="$OMNI_TASK3_GPU" pytest tests/test_step_overlap.py
```

Result: PASS. The measurement run reported:

| Destination step | Overlapped H2D | Total H2D | Ratio |
|---:|---:|---:|---:|
| 2 | 14.611 ms | 22.947 ms | **0.637** |
| 3 | 13.473 ms | 22.488 ms | **0.599** |

Both ratios exceed 0.50, and each first-group copy began during the preceding
step's final-group compute.
