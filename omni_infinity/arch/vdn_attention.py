# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Capability-pinned window-softmax dispatch + extended backends (#42 P3).

The upstream VDN component resolves ``kernels.softmax_backend=auto`` by CUDA
capability with a ``>= 10`` test, so consumer Blackwell (CC 12.x, sm120) is
treated as data-center sm100 and the "sm100 INFERENCE ..." warning fires on
this card. Omni never forwards an implicit backend: :func:`resolve_backend`
pins the choice on ``torch.cuda.get_device_capability()`` explicitly and the
runner logs what resolved, so every run records the kernel it actually used.

Beyond the upstream enum (``auto | flex | decomposed | ref``) this module adds
sm120 bake-off candidates behind the same knob. Each one reuses upstream's
union-of-dense decomposition (the c1 window mask is a few dense rectangles:
one dense-q leg over the full kv, plus per-chunk window groups), swapping the
kernel under the legs:

- ``cudnn``   -- every leg on torch SDPA pinned to cuDNN (9.18+ has native
  sm120 paths); window groups run as a loop of dense calls.
- ``fa4``     -- FA4's CuTe-DSL kernels: varlen for the window leg, dense for
  the dense-q leg (needs a flash-attn build with sm120 cute kernels).
- ``fmha-v2`` -- FlashInfer: ragged batch prefill for the window leg, single
  prefill for the dense-q leg (both NHD, non-causal, explicit scale).
- ``sage``    -- SageAttention (INT8 QK) under the cudnn-style loop;
  accuracy-gated opt-in, never a default.

The adapters are installed by wrapping each hybrid layer's
``_window_softmax`` (duck-typed -- at runtime the transformer is diffusers
remote code, not ``third_party/``), with ``softmax_impl`` left at
``decomposed`` so the upstream forward never takes the flex path.
"""

from __future__ import annotations

import logging

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

logger = logging.getLogger(__name__)

UPSTREAM_BACKENDS = ("auto", "flex", "decomposed", "ref")
EXTENDED_BACKENDS = ("cudnn", "fa4", "fmha-v2", "sage")
ALL_BACKENDS = UPSTREAM_BACKENDS + EXTENDED_BACKENDS

_PLAN_CACHE: dict = {}
MAX_CACHED_PLANS = 4
_WORKSPACE: dict = {}


def resolve_backend(
    requested: str | None, device: int | str | None = 0
) -> tuple[str, str]:
    """Config value -> (explicit backend, reason). ``None``/``auto`` pins on
    the physical capability: CC 12.x is sm120 (consumer Blackwell), where the
    bake-off baseline stays ``decomposed``; other CUDA keeps upstream auto
    semantics (also ``decomposed``); CPU falls back to ``flex`` like
    upstream. Explicit names pass through; unknown names raise."""
    if requested in (None, "auto"):
        if not torch.cuda.is_available():
            return "flex", "cpu: flex (training kernel)"
        if device is None:
            probe = torch.device("cuda", 0)
        elif isinstance(device, int):
            probe = torch.device("cuda", device)
        else:
            probe = torch.device(device)
        if probe.type != "cuda":
            probe = torch.device("cuda", 0)
        cc = torch.cuda.get_device_capability(probe)
        if cc[0] == 12:
            return (
                "decomposed",
                f"cc {cc}: sm120 pinned default (P3 bake-off baseline)",
            )
        return "decomposed", f"cc {cc}: upstream auto semantics"
    if requested not in ALL_BACKENDS:
        raise ValueError(
            f"softmax_backend={requested!r}; expected one of {ALL_BACKENDS}"
        )
    return requested, "explicit"


def backend_available(backend: str) -> tuple[bool, str]:
    """(usable, detail) without raising -- the bake-off records unavailable
    backends as rows, it does not crash on them."""
    if backend in UPSTREAM_BACKENDS:
        return True, "upstream"
    if backend == "cudnn":
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        return True, f"torch SDPA cuDNN {torch.backends.cudnn.version()}"
    if backend == "fa4":
        try:
            import flash_attn.cute.interface  # noqa: F401
        except ImportError as exc:
            return False, f"flash-attn cute missing: {exc}"
        return True, "flash_attn.cute"
    if backend == "fmha-v2":
        try:
            import flashinfer  # noqa: F401
        except ImportError as exc:
            return False, f"flashinfer missing: {exc}"
        return True, f"flashinfer {getattr(flashinfer, '__version__', '?')}"
    if backend == "sage":
        try:
            import sageattention  # noqa: F401
        except ImportError as exc:
            return False, f"sageattention missing: {exc}"
        return True, "sageattention"
    return False, f"unknown backend {backend!r}"


class _Plan:
    """Vendored copy of upstream's decomposition plan (decomposed.py): query
    rows split into one dense-q leg (globals + anchor-row frames, full kv)
    and window groups (chunks) with per-group gathered kv. Kept in lockstep
    with the pinned third_party revision; same cache-key discipline."""

    __slots__ = (
        "dense_q",
        "win_q",
        "kv_gather",
        "cu_q",
        "cu_k",
        "max_q",
        "max_k",
        "has_windows",
        "groups",
    )

    def __init__(self, layout, bounds, anchor_frames, device):
        S = layout.seq_len
        F, TPF = layout.num_frames, layout.tokens_per_frame
        vs, ve = layout.video_start, layout.video_end
        anchor_set = (
            {0, F - 1}
            if anchor_frames in ("columns", "rows", "both")
            else set()
        )
        dense_row_frames = (
            anchor_set if anchor_frames in ("rows", "both") else set()
        )
        dense_col_frames = (
            anchor_set if anchor_frames in ("columns", "both") else set()
        )

        def frame_rows(f):
            return (vs + f * TPF, vs + (f + 1) * TPF)

        global_ranges = [r for r in ((0, vs), (ve, S)) if r[0] < r[1]]

        def merge(ranges):
            out = []
            for a, b in sorted(ranges):
                if out and out[-1][1] >= a:
                    out[-1] = (out[-1][0], max(out[-1][1], b))
                else:
                    out.append((a, b))
            return out

        def cat_ranges(ranges):
            return torch.cat(
                [torch.arange(a, b, device=device) for a, b in ranges]
            )

        dense_ranges = merge(
            global_ranges + [frame_rows(f) for f in sorted(dense_row_frames)]
        )
        self.dense_q = (
            cat_ranges(dense_ranges)
            if dense_ranges
            else torch.empty(0, dtype=torch.long, device=device)
        )

        groups = []
        for f in range(F):
            if f in dense_row_frames:
                continue
            if (
                groups
                and bounds[groups[-1][-1]] == bounds[f]
                and groups[-1][-1] == f - 1
            ):
                groups[-1].append(f)
            else:
                groups.append([f])

        q_idx, kv_idx, q_lens, k_lens = [], [], [], []
        for frames in groups:
            lo, hi = bounds[frames[0]]
            kv_frames = sorted(
                set(range(max(lo, 0), min(hi + 1, F))) | dense_col_frames
            )
            q_r = merge([frame_rows(f) for f in frames])
            kv_r = merge(global_ranges + [frame_rows(f) for f in kv_frames])
            qi, ki = cat_ranges(q_r), cat_ranges(kv_r)
            q_idx.append(qi)
            kv_idx.append(ki)
            q_lens.append(len(qi))
            k_lens.append(len(ki))

        self.has_windows = bool(groups)
        self.groups = None
        if self.has_windows:
            self.win_q = torch.cat(q_idx)
            self.kv_gather = torch.cat(kv_idx)
            zero = torch.zeros(1, dtype=torch.long)
            self.cu_q = torch.cat([zero, torch.tensor(q_lens).cumsum(0)]).to(
                device, torch.int32
            )
            self.cu_k = torch.cat([zero, torch.tensor(k_lens).cumsum(0)]).to(
                device, torch.int32
            )
            self.max_q, self.max_k = max(q_lens), max(k_lens)
            self.groups = list(zip(q_lens, k_lens))
        else:
            self.win_q = torch.empty(0, dtype=torch.long, device=device)

        order = torch.cat([self.dense_q, self.win_q])
        if len(order) != S:
            raise ValueError(f"decomposition covers {len(order)} of {S} rows")


def _plan(layout, bounds, anchor_frames, device):
    key = (
        layout.seq_len,
        layout.video_start,
        layout.num_frames,
        layout.tokens_per_frame,
        tuple(bounds),
        anchor_frames,
        str(device),
    )
    if key not in _PLAN_CACHE:
        _PLAN_CACHE[key] = _Plan(layout, bounds, anchor_frames, device)
        while len(_PLAN_CACHE) > MAX_CACHED_PLANS:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
    return _PLAN_CACHE[key]


def _sdpa_dense(q, k, v, scale, backends):
    # [rows, H, d] -> [rows, H, d] through SDPA's [1, H, S, d] layout.
    with sdpa_kernel(backends):
        out = scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            scale=scale,
        )
    return out[0].transpose(0, 1)


def _loop_window(plan, query, key, value, scale, dense_fn):
    """cudnn/sage leg: each window group is one dense rectangle, run as a
    plain dense attention over its gathered rows."""
    out = torch.empty_like(query)
    if len(plan.dense_q):
        out[plan.dense_q] = dense_fn(query[plan.dense_q], key, value, scale)
    if plan.has_windows:
        qw = query[plan.win_q]
        kw = key[plan.kv_gather]
        vw = value[plan.kv_gather]
        ow = torch.empty_like(qw)
        q0 = k0 = 0
        for qlen, klen in plan.groups:
            ow[q0 : q0 + qlen] = dense_fn(
                qw[q0 : q0 + qlen],
                kw[k0 : k0 + klen],
                vw[k0 : k0 + klen],
                scale,
            )
            q0 += qlen
            k0 += klen
        out[plan.win_q] = ow
    return out


def _cudnn_dense(q, k, v, scale):
    return _sdpa_dense(q, k, v, scale, [SDPBackend.CUDNN_ATTENTION])


def _sage_dense(q, k, v, scale):
    from sageattention import sageattn

    # sageattn wants a batch dim; NHD = [b, s, h, d].
    out = sageattn(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        tensor_layout="NHD",
        is_causal=False,
        sm_scale=scale,
    )[0]
    return out.to(q.dtype)


def _fa4_window(plan, query, key, value, scale):
    from flash_attn.cute.interface import (
        flash_attn_func,
        flash_attn_varlen_func,
    )

    out = torch.empty_like(query)
    if len(plan.dense_q):
        od = flash_attn_func(
            query[plan.dense_q].unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            softmax_scale=scale,
            causal=False,
        )
        od = od[0] if isinstance(od, tuple) else od
        out[plan.dense_q] = od[0]
    if plan.has_windows:
        ow = flash_attn_varlen_func(
            query[plan.win_q],
            key[plan.kv_gather],
            value[plan.kv_gather],
            cu_seqlens_q=plan.cu_q,
            cu_seqlens_k=plan.cu_k,
            max_seqlen_q=plan.max_q,
            max_seqlen_k=plan.max_k,
            softmax_scale=scale,
        )
        out[plan.win_q] = ow[0] if isinstance(ow, tuple) else ow
    return out


def _fmha_v2_window(plan, query, key, value, scale):
    import flashinfer

    heads, head_dim = query.shape[1], query.shape[2]
    out = torch.empty_like(query)
    if len(plan.dense_q):
        out[plan.dense_q] = flashinfer.single_prefill_with_kv_cache(
            query[plan.dense_q], key, value, causal=False, sm_scale=scale
        )
    if plan.has_windows:
        buf = _WORKSPACE.get(query.device)
        if buf is None:
            buf = _WORKSPACE[query.device] = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=query.device
            )
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(buf, "NHD")
        wrapper.plan(
            plan.cu_q,
            plan.cu_k,
            heads,
            heads,
            head_dim,
            causal=False,
            sm_scale=scale,
            q_data_type=query.dtype,
        )
        out[plan.win_q] = wrapper.run(
            query[plan.win_q], key[plan.kv_gather], value[plan.kv_gather]
        )
    return out


def make_window_softmax(backend: str):
    """Extended-backend window softmax with upstream's exact signature:
    ``f(query, key, value, layout, bounds, scale, anchor_frames)`` over
    [T, H, d] packed rows. Raises RuntimeError naming the install when the
    backend's kernels are missing."""
    if backend not in EXTENDED_BACKENDS:
        raise ValueError(
            f"extended backend {backend!r}; expected one of {EXTENDED_BACKENDS}"
        )
    ok, detail = backend_available(backend)
    if not ok:
        raise RuntimeError(f"softmax_backend={backend!r} unavailable: {detail}")

    def window_softmax(
        query, key, value, layout, bounds, scale, anchor_frames="none"
    ):
        plan = _plan(
            layout, tuple(map(tuple, bounds)), anchor_frames, query.device
        )
        if not key.is_contiguous():
            key = key.contiguous()
        if not value.is_contiguous():
            value = value.contiguous()
        if backend == "cudnn":
            return _loop_window(plan, query, key, value, scale, _cudnn_dense)
        if backend == "sage":
            return _loop_window(plan, query, key, value, scale, _sage_dense)
        if backend == "fa4":
            return _fa4_window(plan, query, key, value, scale)
        return _fmha_v2_window(plan, query, key, value, scale)

    window_softmax.backend = backend
    return window_softmax


def _iter_hybrids(transformer):
    for block in transformer.transformer_blocks:
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "softmax_impl"):
            yield attn


def install_backend(
    transformer, backend: str, device: int | str | None = 0
) -> str:
    """Resolve + write the backend onto every hybrid layer. Upstream names
    mirror upstream ``set_softmax_backend`` (the string is the dispatch);
    extended names wrap ``_window_softmax`` and leave ``softmax_impl`` at
    ``decomposed`` so the upstream forward never prebuilds a flex mask."""
    resolved, _ = resolve_backend(backend, device)
    if resolved in EXTENDED_BACKENDS:
        fn = make_window_softmax(resolved)
        for attn in _iter_hybrids(transformer):
            if not hasattr(attn, "anchor_frames"):
                raise RuntimeError(
                    f"{type(attn).__name__} has softmax_impl but no "
                    "anchor_frames; the remote-code layer shape changed -- "
                    "refusing to install an extended backend mid-render"
                )

            def wrapped(
                query,
                key,
                value,
                layout,
                bounds,
                scale,
                inference,
                _fn=fn,
                _attn=attn,
            ):
                return _fn(
                    query,
                    key,
                    value,
                    layout,
                    bounds,
                    scale,
                    anchor_frames=_attn.anchor_frames,
                )

            attn._window_softmax = wrapped
            attn.softmax_impl = "decomposed"
    else:
        for attn in _iter_hybrids(transformer):
            attn.softmax_impl = resolved
    return resolved
