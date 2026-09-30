# C2 encoder prefix cache

Deferred until the Qwen3-VL tower stays resident. The current encoder path
releases the tower, so a prefix cache would not provide reusable residency.

A hit is a complete leading block only. Vision tokens in front of the text are part of the prefix. Picture 1 matches only when the image bytes match and it is first. Do not reorder [Shot N] blocks. vLLM's block hash is the reuse rule
being copied, and partial blocks are not stored.

Do not install ContextPilot. This plan does not implement prefix_blocks.

```python
def prefix_blocks(token_ids: tuple[int, ...], block_size: int) -> list[bytes]:
    """Hash complete leading blocks of token_ids.

    Drop the tail if it is shorter than block_size. Picture tokens are
    already inside token_ids. Return an empty list when the tower is not
    resident. Do not reorder shots.
    """
```

A later implementation may call C3's `VisionEmbedCache` for images inside a
block that still has to run. This PR does not import `caches.vision`.
