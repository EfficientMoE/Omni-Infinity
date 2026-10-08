# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import queue

import pytest
import torch

from omni_infinity.serve.app import ServerSettings
from omni_infinity.serve.roles import (
    HandoffQueue,
    Role,
    StagePayload,
    parse_device_map,
    parse_role,
)


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("OMNI_ROLE", "OMNI_DEVICE_MAP", "OMNI_DEVICE"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


class TestParseRole:
    def test_valid_roles(self):
        assert parse_role("all") is Role.ALL
        assert parse_role("encoder") is Role.ENCODER
        assert parse_role("denoiser") is Role.DENOISER
        assert parse_role("decoder") is Role.DECODER

    def test_case_and_whitespace(self):
        assert parse_role(" Encoder ") is Role.ENCODER

    def test_invalid_role(self):
        with pytest.raises(ValueError, match="OMNI_ROLE"):
            parse_role("transcoder")


class TestParseDeviceMap:
    def test_empty_uses_default(self):
        mapping = parse_device_map(None, default_device="cuda:3")
        assert mapping == {
            Role.ENCODER: "cuda:3",
            Role.DENOISER: "cuda:3",
            Role.DECODER: "cuda:3",
        }

    def test_full_map(self):
        mapping = parse_device_map(
            "encoder=cuda:0,denoiser=cuda:1,decoder=cuda:2"
        )
        assert mapping[Role.ENCODER] == "cuda:0"
        assert mapping[Role.DENOISER] == "cuda:1"
        assert mapping[Role.DECODER] == "cuda:2"

    def test_partial_map_falls_back(self):
        mapping = parse_device_map("denoiser=cuda:1", default_device="cuda:0")
        assert mapping[Role.ENCODER] == "cuda:0"
        assert mapping[Role.DENOISER] == "cuda:1"

    def test_malformed_entry(self):
        with pytest.raises(ValueError, match="role=device"):
            parse_device_map("encoder:cuda:0")

    def test_unknown_role(self):
        with pytest.raises(ValueError, match="OMNI_ROLE"):
            parse_device_map("vae=cuda:0")

    def test_all_rejected(self):
        with pytest.raises(ValueError, match="not 'all'"):
            parse_device_map("all=cuda:0")


class TestServerSettingsRole:
    def test_defaults(self, clean_env):
        settings = ServerSettings.from_env()
        assert settings.role == "all"
        assert dict(settings.device_map) == {
            "encoder": "cuda",
            "denoiser": "cuda",
            "decoder": "cuda",
        }

    def test_role_env(self, clean_env):
        clean_env.setenv("OMNI_ROLE", "denoiser")
        clean_env.setenv("OMNI_DEVICE_MAP", "denoiser=cuda:1")
        settings = ServerSettings.from_env()
        assert settings.role == "denoiser"
        assert dict(settings.device_map)["denoiser"] == "cuda:1"

    def test_device_map_inherits_omni_device(self, clean_env):
        clean_env.setenv("OMNI_DEVICE", "cuda:2")
        settings = ServerSettings.from_env()
        assert dict(settings.device_map) == {
            "encoder": "cuda:2",
            "denoiser": "cuda:2",
            "decoder": "cuda:2",
        }

    def test_invalid_role_fails_fast(self, clean_env):
        clean_env.setenv("OMNI_ROLE", "bogus")
        with pytest.raises(ValueError, match="OMNI_ROLE"):
            ServerSettings.from_env()


class TestStagePayload:
    def test_to_device_copies_tensors_and_meta(self):
        payload = StagePayload(
            job_id="j1",
            stage=Role.ENCODER,
            tensors={"embeds": torch.ones(2, 3)},
            meta={"seed": 0},
        )
        moved = payload.to_device("cpu")
        assert moved.job_id == "j1"
        assert moved.stage is Role.ENCODER
        assert torch.equal(moved.tensors["embeds"], payload.tensors["embeds"])
        moved.meta["seed"] = 1
        assert payload.meta["seed"] == 0


class TestHandoffQueue:
    def test_fifo_ordering(self):
        handoff = HandoffQueue("cpu", maxsize=4)
        for index in range(3):
            handoff.put(
                StagePayload(
                    job_id=f"j{index}",
                    stage=Role.ENCODER,
                    tensors={"x": torch.full((2,), float(index))},
                )
            )
        received = [handoff.get() for _ in range(3)]
        assert [p.job_id for p in received] == ["j0", "j1", "j2"]
        assert received[2].tensors["x"][0].item() == 2.0

    def test_backpressure(self):
        handoff = HandoffQueue("cpu", maxsize=1)
        handoff.put(StagePayload(job_id="a", stage=Role.ENCODER, tensors={}))
        with pytest.raises(queue.Full):
            handoff.put(
                StagePayload(job_id="b", stage=Role.ENCODER, tensors={}),
                timeout=0.05,
            )
        assert handoff.get().job_id == "a"

    def test_get_timeout(self):
        handoff = HandoffQueue("cpu")
        with pytest.raises(queue.Empty):
            handoff.get(timeout=0.05)

    @pytest.mark.gpu
    def test_cuda_d2d_copy(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("needs 2 GPUs")
        handoff = HandoffQueue("cuda:1")
        payload = StagePayload(
            job_id="j",
            stage=Role.DENOISER,
            tensors={"latents": torch.randn(1, 4, 8, 16, 16, device="cuda:0")},
        )
        handoff.put(payload)
        out = handoff.get()
        assert out.tensors["latents"].device == torch.device("cuda", 1)
        torch.cuda.synchronize()
        assert torch.equal(
            out.tensors["latents"].cpu(),
            payload.tensors["latents"].cpu(),
        )
