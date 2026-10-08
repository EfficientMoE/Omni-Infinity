# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""P7 spike: split stages across GPUs, fully resident per role.

Loads the H3 pipeline to host RAM, places text_encoder / transformer /
VAEs on three GPUs (the plan's "96 GB per role, no streaming" memory
upside), runs the split stages with explicit D2D state handoffs, and
reports per-stage latency, handoff bytes/GB/s, and bitwise parity
against the committed single-GPU goldens.

Run (see docs/multigpu_platform.md for device-pair guidance):
  CUDA_VISIBLE_DEVICES=0,1,2 HF_HOME=... HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 OMNI_H3_CHECKPOINT=<snapshot> \
  python benchmarks/multigpu_split_spike.py
"""

import dataclasses
import os
import time
from pathlib import Path

import torch

from omni_infinity.runner import ReferenceRunner, resolve_resolution
from omni_infinity.serve.roles import Role
from omni_infinity.serve.split import prepare_state, run_stage

_CYCLE_SENTINEL = object()

GOLDENS = Path("tests/fixtures/goldens/fl2va_goldens.pt")
REFERENCE = Path("tests/fixtures/ref.png")

ROLE_DEVICES = {
    Role.ENCODER: os.environ.get("OMNI_ENC_DEVICE", "cuda:0"),
    Role.DENOISER: os.environ.get("OMNI_DEN_DEVICE", "cuda:1"),
    Role.DECODER: os.environ.get("OMNI_DEC_DEVICE", "cuda:2"),
}
ROLE_COMPONENTS = {
    Role.ENCODER: ("text_encoder", "vae"),
    Role.DENOISER: ("transformer",),
    Role.DECODER: ("audio_vae",),
}


def _contains_sentinel(value, visited):
    if value is _CYCLE_SENTINEL:
        return True
    if id(value) in visited:
        return False
    visited.add(id(value))
    if isinstance(value, (list, tuple)):
        return any(_contains_sentinel(item, visited) for item in value)
    if isinstance(value, dict):
        return any(_contains_sentinel(item, visited) for item in value.values())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_sentinel(getattr(value, field.name), visited)
            for field in dataclasses.fields(value)
        )
    if hasattr(value, "__dict__") and not isinstance(
        value, (torch.nn.Module, torch.Generator, type)
    ):
        return any(
            _contains_sentinel(item, visited) for item in vars(value).values()
        )
    return False


def _move(value, device, counter, seen):
    if torch.is_tensor(value):
        key = id(value)
        if key in seen:
            return seen[key]
        if value.is_cuda and str(value.device) != device:
            counter[0] += value.numel() * value.element_size()
            moved = value.to(device)
        else:
            moved = value
        seen[key] = moved
        return moved
    key = id(value)
    if key in seen:
        return seen[key]
    if isinstance(value, list):
        seen[key] = value
        value[:] = [_move(item, device, counter, seen) for item in value]
        return value
    if isinstance(value, tuple):
        seen[key] = _CYCLE_SENTINEL
        items = tuple(_move(item, device, counter, seen) for item in value)
        if _contains_sentinel(items, set()):
            raise RuntimeError(
                "cycle through an immutable tuple in pipeline state; "
                "aliasing cannot be preserved"
            )
        if hasattr(value, "_fields"):
            moved = type(value)(*items)
        else:
            moved = type(value)(items)
        seen[key] = moved
        return moved
    if isinstance(value, dict):
        seen[key] = value
        for item_key in list(value):
            value[item_key] = _move(value[item_key], device, counter, seen)
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        seen[key] = value
        frozen = value.__dataclass_params__.frozen
        for field in dataclasses.fields(value):
            moved = _move(getattr(value, field.name), device, counter, seen)
            if frozen:
                object.__setattr__(value, field.name, moved)
            else:
                setattr(value, field.name, moved)
        return value
    if (
        hasattr(value, "__dict__")
        and not isinstance(value, (torch.nn.Module, torch.Generator, type))
        and value.__class__.__module__ != "builtins"
    ):
        seen[key] = value
        for name in list(vars(value)):
            setattr(
                value, name, _move(vars(value)[name], device, counter, seen)
            )
        return value
    return value


def audit_state(state, device):
    stray = []
    seen = set()

    def walk(path, value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if torch.is_tensor(value):
            if value.is_cuda and str(value.device) != device:
                stray.append((path, str(value.device), tuple(value.shape)))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(f"{path}[{index}]", item)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(f"{path}.{key}", item)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                walk(f"{path}.{field.name}", getattr(value, field.name))
        elif hasattr(value, "__dict__") and not isinstance(
            value, (torch.nn.Module, torch.Generator, type)
        ):
            for name, item in vars(value).items():
                walk(f"{path}.{name}", item)

    for key, value in state.values.items():
        walk(key, value)
    return stray


def move_state(state, device):
    counter = [0]
    seen = {}
    start = time.perf_counter()
    for key, value in list(state.values.items()):
        state.values[key] = _move(value, device, counter, seen)
    torch.cuda.synchronize()
    return counter[0], time.perf_counter() - start


def main():
    from PIL import Image

    payload = torch.load(GOLDENS, weights_only=False)
    assert payload["torch_version"] == torch.__version__, "stack mismatch"
    print("loading pipeline to host RAM (full resident, no streaming)...")
    t0 = time.perf_counter()
    runner = ReferenceRunner.from_pretrained(
        os.environ["OMNI_H3_CHECKPOINT"],
        device="cpu",
        offload=False,
        store_dir=os.environ.get("OMNI_H3_STORE"),
        store_components=("transformer", "vae", "audio_vae"),
    )
    pipe = runner.pipeline
    print(f"  load: {time.perf_counter() - t0:.1f}s")

    for role, names in ROLE_COMPONENTS.items():
        for name in names:
            module = getattr(pipe, name, None)
            if module is not None and hasattr(module, "to"):
                module.to(ROLE_DEVICES[role])
                print(f"  {name} -> {ROLE_DEVICES[role]}")
    for gpu in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(gpu) / 2**30
        print(f"  cuda:{gpu} allocated: {used:.1f} GiB")

    height, width = resolve_resolution(payload["resolution"])
    state = prepare_state(
        pipe,
        prompt=payload["prompt"],
        image=Image.open(REFERENCE).convert("RGB"),
        height=height,
        width=width,
        num_frames=payload["frames"],
        num_inference_steps=payload["steps"],
        generator=torch.Generator("cpu").manual_seed(payload["seed"]),
        output_type="np",
    )

    current_device = {"value": torch.device(ROLE_DEVICES[Role.ENCODER])}
    original_property = type(pipe).__dict__.get("_execution_device")
    type(pipe)._execution_device = property(
        lambda self: current_device["value"]
    )

    timings = {}
    try:
        for role in (Role.ENCODER, Role.DENOISER, Role.DECODER):
            current_device["value"] = torch.device(ROLE_DEVICES[role])
            if role is Role.DECODER:
                start = time.perf_counter()
                pipe.vae.to(ROLE_DEVICES[role])
                torch.cuda.synchronize()
                print(
                    "  vae migrated to decoder device in "
                    f"{(time.perf_counter() - start) * 1e3:.0f} ms "
                    "(real role processes each hold their own copy; the "
                    "keyframe encode also needs it on the encoder)"
                )
            moved, copy_s = move_state(state, ROLE_DEVICES[role])
            strays = audit_state(state, ROLE_DEVICES[role])
            if strays:
                for path, device, shape in strays:
                    print(f"  STRAY {path} on {device} {shape}")
                raise SystemExit("state handoff left stray tensors behind")
            start = time.perf_counter()
            state = run_stage(pipe, role, state)
            torch.cuda.synchronize()
            timings[role.value] = (time.perf_counter() - start, moved, copy_s)
    finally:
        if original_property is not None:
            type(pipe)._execution_device = original_property
        else:
            del type(pipe)._execution_device

    for name, (stage_s, moved, copy_s) in timings.items():
        gbps = moved / copy_s / 1e9 if copy_s > 0 and moved else 0.0
        print(
            f"  {name:9s} stage={stage_s:7.2f}s "
            f"handoff={moved / 2**20:8.1f} MiB in {copy_s * 1e3:7.1f} ms "
            f"({gbps:5.1f} GB/s)"
        )

    latents = state.get("latents").cpu()
    bitwise = torch.equal(latents, payload["latents"])
    audio = state.get("audio_latents")
    if payload["audio_latents"] is None:
        audio_ok = audio is None
    else:
        audio_ok = audio is not None and torch.equal(
            audio.cpu(), payload["audio_latents"]
        )
    print(f"  video latents bitwise vs goldens: {bitwise}")
    print(f"  audio latents bitwise vs goldens: {audio_ok}")
    if not (bitwise and audio_ok):
        rel = (latents - payload["latents"]).norm() / payload["latents"].norm()
        print(f"  rms_rel={rel:.6f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
