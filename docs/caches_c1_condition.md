# C1 exact condition cache

The C1 cache is opt-in through the `condition-cache` registry optimization.
It reuses only a complete condition: checkpoint namespace, full prompt,
ordered image bytes (including empty slots), height, width, and frame count.
A shared [Shot 1] opener is not a hit. Shot prefixes are never matched or
reordered.

Misses run the original pipeline unchanged. Hits build an encoder-less view
and inject only values declared by that view. H3 condition encoding runs with
`use_cache=False`; the cache does not alter that setting. The
`keyframe_encode_seed` remains part of the normal request behavior rather
than becoming hidden mutable cache state.
During preparation, the request generator is not consumed.

Unexpected hash values, unreadable disk entries, or pipelines that cannot be
reduced fail open to normal condition encoding. Disk entries are loaded with
PyTorch's `weights_only=True` mode.

## Warm-hit smoke gate

Status: passed on RTX PRO 6000 Blackwell (sm120), 2026-09-30.

Measured with the commands below (a local H3 snapshot and moe-store on
NVMe, `--frames 120 --steps 8 --resolution 256p`):

| phase | wall-clock | rms_rel vs goldens |
|---|---|---|
| cold (miss) | 67.7 s | 0.0000 |
| warm (hit) | 31.0 s | 0.0000 |

The warm hit reproduces the goldens bitwise. The warm run is strictly
faster than the cold run because the text and VAE encoders are skipped;
the stored `<key>.pt` entry was present in the cache directory between
runs.

Cold run (miss path):

```bash
rm -rf /tmp/omni-c1-gate
python examples/fl2va_smoke.py --prompt "a red ball bouncing" \
    --seed 0 --steps 8 --resolution 256p --frames 120 \
    --checkpoint <local-H3-snapshot-dir> --offload \
    --store-dir <moe-store> --store-components transformer,vae,audio_vae \
    --adaln-host-cache --block-stream-blocks-per-group 1 \
    --condition-cache --condition-cache-dir /tmp/omni-c1-gate \
    --goldens tests/fixtures/goldens/fl2va_goldens.pt
```

Warm run (hit path; use the same cache directory without removing it):

```bash
python examples/fl2va_smoke.py --prompt "a red ball bouncing" \
    --seed 0 --steps 8 --resolution 256p --frames 120 \
    --checkpoint <local-H3-snapshot-dir> --offload \
    --store-dir <moe-store> --store-components transformer,vae,audio_vae \
    --adaln-host-cache --block-stream-blocks-per-group 1 \
    --condition-cache --condition-cache-dir /tmp/omni-c1-gate \
    --goldens tests/fixtures/goldens/fl2va_goldens.pt
```
