# C2 encoder prefix cache

Deferred until the Qwen3-VL tower stays resident. The current encoder path
releases the tower, so a prefix cache would not provide reusable residency.

A hit is a complete leading block only. Vision tokens in front of the text are part of the prefix. Picture 1 matches only when the image bytes match and it is first. Do not reorder [Shot N] blocks. vLLM's block hash is the reuse rule
being copied, and partial blocks are not stored.

The future cache-key API must include a hash of each image's content and that
image's token-span position. `token_ids` alone cannot establish byte identity:
the pinned Qwen3-VL path repeats `<|image_pad|>` IDs while the image content
is carried separately in `pixel_values`.

Do not install ContextPilot. This plan does not implement prefix_blocks.

```python
def prefix_blocks(token_ids: tuple[int, ...], block_size: int) -> list[bytes]:
    """Hash complete leading blocks of token_ids and image metadata.

    The cache key also includes image content hashes and token-span positions.
    Drop the tail if it is shorter than block_size. Do not reorder shots.
    """
```

`prefix_blocks` only hashes inputs. It has no tower-residency state and must
not claim to determine residency or return an empty result for a non-resident
tower. Its caller must invoke it only while the Qwen3-VL tower is resident.

A later implementation may call C3's `VisionEmbedCache` for images inside a
block that still has to run. This PR does not import `caches.vision`.
