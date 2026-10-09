# C2 exact encoder-prefix cache

C2 v1 reuses only an exact tokenized presentation. Enable it for the dense H3
runner with the `encoder-cache` registry optimization or `--encoder-cache` in
the FL2VA smoke CLI. It wraps Qwen3-VL's model forward and stores the complete
structured encoder output on CPU; a hit materializes that output on the input
device without running the encoder.

The SHA256 key is versioned with `omni-encoder-v1` and covers the model
namespace, the full token-ID tensor, and ordered byte-stable image/video input
tensors. A changed token, image, image order, dtype, or shape is a miss, so a
hit is bitwise by construction.

The host LRU uses a lock and promotes hits.
Both the entry cap and byte budget are enforced. Admission follows
MoE-Infinity's policy shape: pressure evicts the oldest resident entry; an
output too large to fit after all possible evictions is transient, so it is
returned but not stored. The single-worker serving model needs no lease or
refcount machinery.

Partial-prefix splicing is not implemented. C2 v1 does not hash token blocks,
reuse a leading subset, or merge cached and newly encoded hidden states.

C2's contribution is measured as encoder time saved on exact repeats in the
seeded VidProM serving trace and on repeated encoder presentations in the
multi-prompt demo. The one-factor `c2` ablation row records cold/warm latency,
bitwise `rms_rel`, and encoder-cache hit/miss counters; measured numbers remain
weights-gated in [cache_benchmarks.md](cache_benchmarks.md).
