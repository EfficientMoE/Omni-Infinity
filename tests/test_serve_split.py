# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from dataclasses import dataclass

import pytest
import torch

from omni_infinity.serve.roles import Role
from omni_infinity.serve.split import (
    STAGE_BLOCKS,
    STAGE_ORDER,
    prepare_state,
    run_all_stages,
    run_stage,
)


@dataclass
class _Input:
    name: str
    default: object = None
    kwargs_type: str | None = None


class _Block:
    def __init__(self, name):
        self.name = name

    def __call__(self, pipeline, state):
        trace = state.get("trace")
        trace.append(self.name)
        state.set(self.name + "_out", torch.ones(2) * len(trace), None)
        return pipeline, state


class _Blocks:
    inputs = [
        _Input("prompt"),
        _Input("trace"),
        _Input("num_inference_steps", default=8),
    ]

    def __init__(self):
        self.sub_blocks = OrderedDict(
            (name, _Block(name))
            for name in (
                "before_encode",
                "text_encoder",
                "vae_encoder",
                "denoise",
                "decode",
            )
        )


class _FakePipeline:
    def __init__(self):
        self._blocks = _Blocks()


def test_stage_blocks_cover_all_top_level_steps_once():
    names = [n for role in STAGE_ORDER for n in STAGE_BLOCKS[role]]
    assert names == [
        "before_encode",
        "text_encoder",
        "vae_encoder",
        "denoise",
        "decode",
    ]


def test_prepare_state_applies_kwargs_and_defaults():
    state = prepare_state(_FakePipeline(), prompt="hi", trace=[])
    assert state.get("prompt") == "hi"
    assert state.get("num_inference_steps") == 8


def test_prepare_state_expands_kwargs_type_dicts():
    pipeline = _FakePipeline()
    pipeline._blocks.inputs = [
        _Input("prompt"),
        _Input("trace"),
        _Input(None, kwargs_type="denoiser_input_fields"),
    ]
    state = prepare_state(
        pipeline,
        prompt="p",
        trace=[],
        denoiser_input_fields={"sigma_shift": 1.5},
    )
    assert state.get("sigma_shift") == 1.5


def test_prepare_state_rejects_unknown_inputs():
    with pytest.raises(ValueError, match="unexpected pipeline inputs"):
        prepare_state(_FakePipeline(), bogus=1)


def test_run_stage_runs_only_the_role_subset():
    pipeline = _FakePipeline()
    state = prepare_state(pipeline, prompt="p", trace=[])
    state = run_stage(pipeline, Role.ENCODER, state)
    assert state.get("trace") == [
        "before_encode",
        "text_encoder",
        "vae_encoder",
    ]
    assert state.get("denoise_out") is None


def test_run_stage_rejects_role_all():
    pipeline = _FakePipeline()
    state = prepare_state(pipeline, prompt="p", trace=[])
    with pytest.raises(ValueError, match="no stage blocks"):
        run_stage(pipeline, Role.ALL, state)


def test_split_run_matches_sequential_order():
    pipeline = _FakePipeline()
    state = run_all_stages(pipeline, prompt="p", trace=[])
    assert state.get("trace") == [
        "before_encode",
        "text_encoder",
        "vae_encoder",
        "denoise",
        "decode",
    ]
    assert torch.equal(state.get("decode_out"), torch.ones(2) * 5)


def test_rejects_pipeline_without_sub_blocks():
    with pytest.raises(TypeError, match="modular sub-blocks"):
        prepare_state(object(), prompt="p")
