# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the capability-pinned window-softmax dispatch (#42 P3)."""

import math
from dataclasses import dataclass

import pytest
import torch

from omni_infinity.arch import vdn_attention as va


@dataclass(frozen=True)
class _Layout:
    seq_len: int
    video_start: int
    num_frames: int
    tokens_per_frame: int

    @property
    def video_end(self):
        return self.video_start + self.num_frames * self.tokens_per_frame


def _fake_cuda(monkeypatch, cc):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=0: cc
    )


def test_resolve_pins_sm120(monkeypatch):
    _fake_cuda(monkeypatch, (12, 0))
    resolved, reason = va.resolve_backend(None)
    assert resolved == "decomposed"
    assert "12" in reason and "sm120" in reason


def test_resolve_auto_other_cuda(monkeypatch):
    _fake_cuda(monkeypatch, (9, 0))
    resolved, reason = va.resolve_backend("auto")
    assert resolved == "decomposed"
    assert "(9, 0)" in reason


def test_resolve_cpu_falls_back_to_flex(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert va.resolve_backend(None) == ("flex", "cpu: flex (training kernel)")


def test_resolve_explicit_passthrough():
    for name in va.ALL_BACKENDS:
        if name == "auto":
            continue
        assert va.resolve_backend(name) == (name, "explicit")


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="fmha-v2"):
        va.resolve_backend("nope")


def test_backend_available_never_raises():
    for name in va.ALL_BACKENDS:
        ok, detail = va.backend_available(name)
        assert isinstance(ok, bool) and isinstance(detail, str)
    for name in va.UPSTREAM_BACKENDS:
        assert va.backend_available(name) == (True, "upstream")


def test_make_window_softmax_rejects_upstream_names():
    with pytest.raises(ValueError):
        va.make_window_softmax("decomposed")


def test_make_window_softmax_unavailable_raises(monkeypatch):
    monkeypatch.setattr(va, "backend_available", lambda b: (False, "not here"))
    with pytest.raises(RuntimeError, match="not here"):
        va.make_window_softmax("fa4")


def _tiny_layout():
    # 4 global rows then 4 frames x 2 tokens: seq_len 12.
    return _Layout(seq_len=12, video_start=4, num_frames=4, tokens_per_frame=2)


def test_plan_matches_hand_computed():
    layout = _tiny_layout()
    bounds = [(0, 1), (0, 1), (2, 3), (2, 3)]  # chunk=2, radius=0 style
    plan = va._Plan(layout, bounds, "both", torch.device("cpu"))
    # anchor rows 0 and 3 are dense-q; globals are rows 0..3.
    assert plan.dense_q.tolist() == [0, 1, 2, 3, 4, 5, 10, 11]
    # remaining frames 1, 2 have distinct bounds -> two groups.
    assert plan.win_q.tolist() == [6, 7, 8, 9]
    assert plan.cu_q.tolist() == [0, 2, 4]
    # frame 1 window {0,1} + anchor col {3} + globals -> rows 0-7, 10-11.
    assert plan.kv_gather.tolist()[: plan.cu_k[1]] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        10,
        11,
    ]
    # frame 2 window {2,3} + anchor col {0} + globals.
    assert plan.kv_gather.tolist()[plan.cu_k[1] :] == [
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        10,
        11,
    ]
    assert plan.max_q == 2 and plan.max_k == 10


def _eager_oracle(q, k, v, layout, bounds, scale, anchor_frames):
    S = layout.seq_len
    F, TPF = layout.num_frames, layout.tokens_per_frame
    vs = layout.video_start
    keep = torch.zeros(S, S, dtype=torch.bool)
    is_global = torch.ones(S, dtype=torch.bool)
    is_global[vs : layout.video_end] = False
    keep[is_global] = True
    keep[:, is_global] = True
    anchors = (
        {0, F - 1} if anchor_frames in ("rows", "columns", "both") else set()
    )
    for fq in range(F):
        lo, hi = bounds[fq]
        rows = slice(vs + fq * TPF, vs + (fq + 1) * TPF)
        if anchor_frames in ("rows", "both") and fq in anchors:
            keep[rows] = True
            continue
        for fk in range(F):
            in_window = max(lo, 0) <= fk <= min(hi, F - 1)
            anchor_col = anchor_frames in ("columns", "both") and fk in anchors
            if in_window or anchor_col:
                keep[rows, vs + fk * TPF : vs + (fk + 1) * TPF] = True
    scores = (q.permute(1, 0, 2) @ k.permute(1, 2, 0)) * scale
    scores.masked_fill_(~keep, -math.inf)
    return (scores.softmax(-1) @ v.permute(1, 0, 2)).permute(1, 0, 2)


def test_loop_window_matches_eager_oracle():
    torch.manual_seed(0)
    layout = _tiny_layout()
    bounds = [(0, 1), (0, 1), (2, 3), (2, 3)]
    q = torch.randn(12, 3, 8)
    k = torch.randn(12, 3, 8)
    v = torch.randn(12, 3, 8)
    scale = 8**-0.5
    plan = va._Plan(layout, bounds, "both", torch.device("cpu"))

    def dense(qr, kr, vr, s):
        return va._sdpa_dense(qr, kr, vr, s, [va.SDPBackend.MATH])

    got = va._loop_window(plan, q, k, v, scale, dense)
    want = _eager_oracle(q, k, v, layout, bounds, scale, "both")
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


class _FakeAttn:
    def __init__(self):
        self.softmax_impl = "flex"
        self.anchor_frames = "both"

    def __delattr__(self, name):
        object.__delattr__(self, name)

    def _window_softmax(self, *a):
        return "original"


class _FakeBlock:
    def __init__(self):
        self.attn = _FakeAttn()


class _FakeTransformer:
    def __init__(self, n=2):
        self.transformer_blocks = [_FakeBlock() for _ in range(n)]


def test_install_upstream_name_sets_softmax_impl(monkeypatch):
    _fake_cuda(monkeypatch, (12, 0))
    model = _FakeTransformer()
    assert va.install_backend(model, "ref") == "ref"
    assert all(b.attn.softmax_impl == "ref" for b in model.transformer_blocks)


def test_install_extended_wraps_window_softmax(monkeypatch):
    _fake_cuda(monkeypatch, (12, 0))
    model = _FakeTransformer()
    sentinel = object()
    monkeypatch.setattr(
        va,
        "make_window_softmax",
        lambda b: lambda *a, anchor_frames=None: (sentinel, b, anchor_frames),
    )
    assert va.install_backend(model, "fmha-v2") == "fmha-v2"
    for block in model.transformer_blocks:
        assert block.attn.softmax_impl == "decomposed"
        out = block.attn._window_softmax(1, 2, 3, None, None, 0.1, True)
        assert out == (sentinel, "fmha-v2", "both")


def test_runner_pins_backend_when_none(monkeypatch):
    import omni_infinity.arch.vdn as vdn_mod

    _fake_cuda(monkeypatch, (12, 0))

    class _FakePipe:
        transformer = object()

        def __init__(self):
            self.load_kwargs = None

        def load_components(self, **kw):
            self.load_kwargs = kw

        def to(self, device):
            return self

    fake = _FakePipe()
    monkeypatch.setattr(vdn_mod, "_load_modular_pipeline", lambda c, w: fake)
    monkeypatch.setattr(vdn_mod, "attach_caches", lambda runner, **kw: None)
    vdn_mod.VdnRunner.from_pretrained(device="cpu")
    assert fake.load_kwargs["softmax_backend"] == {"transformer": "decomposed"}


def test_install_extended_requires_anchor_frames(monkeypatch):
    _fake_cuda(monkeypatch, (12, 0))
    monkeypatch.setattr(
        va, "make_window_softmax", lambda b: lambda *a, **k: None
    )
    model = _FakeTransformer()
    for block in model.transformer_blocks:
        del block.attn.anchor_frames

    class _NoAnchor:
        softmax_impl = "flex"

    model.transformer_blocks[0].attn = _NoAnchor()
    with pytest.raises(RuntimeError, match="anchor_frames"):
        va.install_backend(model, "fmha-v2")


def test_resolve_probes_cuda_zero_for_cpu_device(monkeypatch):
    _fake_cuda(monkeypatch, (12, 0))
    resolved, reason = va.resolve_backend(None, "cpu")
    assert resolved == "decomposed" and "sm120" in reason
