# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import time

import numpy as np
import pytest
import torch
from PIL import Image

from omni_infinity.registry import resolve_profile
from omni_infinity.runner import GenerationResult
from omni_infinity.serve.models import (
    TERMINAL_STATUSES,
    GenerationRequest,
    JobStatus,
)
from omni_infinity.serve.service import InvalidMedia, JobService
from omni_infinity.serve.store import JobStore


def test_decode_image_rejects_bad_base64_non_images_and_oversized_payloads():
    with pytest.raises(InvalidMedia, match="invalid base64 for first frame"):
        JobService._decode_image("!!!!", "first frame")

    raw = base64.b64encode(b"not an image").decode()
    with pytest.raises(InvalidMedia, match="invalid image for last frame"):
        JobService._decode_image(raw, "last frame")

    huge = base64.b64encode(b"\x00" * (20 * 1024 * 1024 + 1)).decode()
    with pytest.raises(InvalidMedia, match="first frame exceeds the 20 MiB"):
        JobService._decode_image(huge, "first frame")


def test_decode_image_returns_none_and_keeps_png_bytes():
    assert JobService._decode_image(None, "first frame") is None
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    assert JobService._decode_image(encoded, "first frame") == buffer.getvalue()


def test_save_and_load_image_round_trip_rgb(tmp_path):
    store = JobStore(tmp_path)
    service = JobService(object(), resolve_profile("h3-dense", []), store)
    record = store.create(GenerationRequest(type="fl2va", prompt="prompt"))
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    try:
        service.save_image(record.id, "input-first.png", buffer.getvalue())
        loaded = service.load_image(record.id, "input-first.png")
        assert loaded.size == (8, 8)
        assert loaded.mode == "RGB"
        assert service.load_image(record.id, "input-last.png") is None
        with pytest.raises(InvalidMedia, match="could not save"):
            service.save_image(record.id, "input-last.png", b"not an image")
    finally:
        service.shutdown()


def _wait(store, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = store.get(job_id)
        if record.status in TERMINAL_STATUSES:
            return record
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def _artifact():
    frames = np.zeros((4, 16, 16, 3), dtype=np.float32)
    return GenerationResult(
        videos=[frames],
        audio=torch.zeros(2, 1600, dtype=torch.float32),
        sampling_rate=16000,
        latents=None,
        audio_latents=None,
    )


def test_vdn_job_rejects_a_canvas_below_768p(tmp_path):
    class Runner:
        def generate(self, *args, **kwargs):
            raise AssertionError("runner must not start")

    store = JobStore(tmp_path)
    service = JobService(Runner(), resolve_profile("vdn-hybrid", []), store)
    try:
        record = service.submit(
            GenerationRequest(
                type="fl2va",
                prompt="prompt",
                model_arch="vdn-hybrid",
                resolution="256p",
            )
        )
        failed = _wait(store, record.id)
    finally:
        service.shutdown()

    assert failed.status == JobStatus.FAILED
    assert failed.error == "ValueError: vdn-hybrid requires the 768p canvas"


def test_vdn_job_forwards_steps_as_evaluations(tmp_path):
    seen = {}

    class Runner:
        def generate(self, prompt, *, num_evaluations, step_callback, **kwargs):
            seen["prompt"] = prompt
            seen["num_evaluations"] = num_evaluations
            seen["kwargs"] = kwargs
            for step in range(1, num_evaluations + 1):
                step_callback(step, num_evaluations)
            return _artifact()

    store = JobStore(tmp_path)
    service = JobService(Runner(), resolve_profile("vdn-hybrid", []), store)
    try:
        record = service.submit(
            GenerationRequest(
                type="fl2va",
                prompt="a red ball bouncing",
                model_arch="vdn-hybrid",
                resolution="768p",
                num_inference_steps=4,
            )
        )
        done = _wait(store, record.id)
    finally:
        service.shutdown()

    assert done.status == JobStatus.SUCCEEDED
    assert done.progress.completed_steps == 4
    assert seen["prompt"] == "a red ball bouncing"
    assert seen["num_evaluations"] == 4
    assert "resolution" not in seen["kwargs"]
    assert "num_inference_steps" not in seen["kwargs"]
