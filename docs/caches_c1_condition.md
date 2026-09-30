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
