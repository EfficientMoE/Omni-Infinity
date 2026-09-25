# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""VdnRunner load-kwarg mapping, tested against a fake pipeline."""

import pytest
import torch

from omni_infinity.arch import vdn


class _FakePipeline:
    def __init__(self):
        self.load_kwargs = None
        self.moved_to = None
        self.transformer = object()
        self.text_encoder = torch.nn.Linear(2, 2)
        self.vae = torch.nn.Linear(2, 2)
        self.audio_vae = torch.nn.Linear(2, 2)

    def load_components(self, **kwargs):
        self.load_kwargs = kwargs

    def to(self, device):
        self.moved_to = device
        return self


@pytest.fixture
def fake(monkeypatch):
    pipe = _FakePipeline()
    monkeypatch.setattr(
        vdn, "_load_modular_pipeline", lambda checkpoint, workflow: pipe
    )
    return pipe


def test_default_is_8_step_bf16(fake):
    runner = vdn.VdnRunner.from_pretrained(device="cpu")
    assert fake.load_kwargs["trust_remote_code"] is True
    assert fake.load_kwargs["torch_dtype"] is torch.bfloat16
    assert "subfolder" not in fake.load_kwargs
    assert "fp8" not in fake.load_kwargs
    assert fake.moved_to == "cpu"
    assert runner.default_evaluations == 8


def test_50_step_variant_selects_stage_b_subfolder(fake):
    runner = vdn.VdnRunner.from_pretrained(variant="50-step", device="cpu")
    assert fake.load_kwargs["subfolder"] == {
        "transformer": "stage-b-step-2000/diffusers"
    }
    assert runner.default_evaluations == 50


def test_fp8_and_backend_are_per_component(fake):
    vdn.VdnRunner.from_pretrained(
        fp8=True, softmax_backend="decomposed", device="cpu"
    )
    assert fake.load_kwargs["fp8"] == {"transformer": True}
    assert fake.load_kwargs["softmax_backend"] == {"transformer": "decomposed"}


def test_block_stream_requires_offload(fake):
    with pytest.raises(ValueError, match="offload"):
        vdn.VdnRunner.from_pretrained(
            block_stream_blocks_per_group=1, device="cpu"
        )


def test_unknown_variant_raises(fake):
    with pytest.raises(ValueError, match="variant"):
        vdn.VdnRunner.from_pretrained(variant="9-step", device="cpu")


def test_sigma_grid_points_are_evals_plus_one():
    assert vdn._sigma_grid_points(8) == 9
    assert vdn._sigma_grid_points(50) == 51


def test_generate_reports_each_denoising_step_and_removes_hook():
    calls = {}

    class FakeTransformer(torch.nn.Module):
        def forward(self, value):
            return value

    class FakeState:
        values = {
            "videos": ["video"],
            "audio": torch.zeros(2, 4),
            "sampling_rate": 48000,
            "latents": torch.zeros(1),
            "audio_latents": torch.zeros(2),
        }

    class FakePipeline:
        def __init__(self):
            self.transformer = FakeTransformer()

        def __call__(self, **kwargs):
            calls.update(kwargs)
            for _ in range(3):
                self.transformer(torch.ones(1))
            return FakeState()

    pipeline = FakePipeline()
    progress = []
    first = object()
    last = object()
    vdn.VdnRunner(pipeline, default_evaluations=8).generate(
        "prompt",
        num_evaluations=3,
        image=first,
        last_image=last,
        step_callback=lambda completed, total: progress.append(
            (completed, total)
        ),
    )

    assert calls["num_inference_steps"] == 4
    assert calls["image"] is first
    assert calls["last_image"] is last
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert pipeline.transformer._forward_hooks == {}
