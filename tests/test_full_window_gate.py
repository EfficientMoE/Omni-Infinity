# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from argparse import Namespace

from examples import fl2va_smoke


def test_vram_window_defaults_to_denoise(monkeypatch):
    monkeypatch.setattr("sys.argv", ["fl2va_smoke.py", "--prompt", "test"])

    assert fl2va_smoke.parse_args().vram_window == "denoise"


def test_full_vram_window_bounds_generate(monkeypatch, capsys):
    events = []

    class _Runner:
        pipeline = object()

        def generate(self, *args, **kwargs):
            events.append("generate")
            return Namespace(latents=None)

    args = Namespace(
        prompt="test",
        seed=0,
        steps=1,
        resolution="256p",
        frames=8,
        checkpoint="checkpoint",
        device="cuda",
        offload=False,
        output_dir=fl2va_smoke.Path("."),
        record_goldens=None,
        store_dir=None,
        store_components="vae,audio_vae",
        adaln_host_cache=False,
        transformer_fp8=False,
        fp8_skip_last_blocks=0,
        block_stream_blocks_per_group=0,
        block_stream_to_disk=None,
        stream_text_encoder=False,
        max_vram="22GiB",
        vram_window="full",
        goldens=None,
    )
    monkeypatch.setattr(fl2va_smoke, "parse_args", lambda: args)
    monkeypatch.setattr(
        fl2va_smoke.ReferenceRunner,
        "from_pretrained",
        lambda *args, **kwargs: _Runner(),
    )
    monkeypatch.setattr(
        fl2va_smoke,
        "DenoiseMemoryProbe",
        lambda device: (_ for _ in ()).throw(
            AssertionError("denoise probe selected for full window")
        ),
    )
    monkeypatch.setattr(
        fl2va_smoke.torch.cuda,
        "reset_peak_memory_stats",
        lambda device: events.append("reset"),
    )
    monkeypatch.setattr(
        fl2va_smoke.torch.cuda,
        "max_memory_allocated",
        lambda device: events.append("max") or 1,
    )
    monkeypatch.setattr(fl2va_smoke, "export_outputs", lambda *args: None)
    monkeypatch.setattr(fl2va_smoke.Path, "mkdir", lambda *args, **kwargs: None)

    assert fl2va_smoke.main() == 0
    assert events == ["reset", "generate", "max"]
    output = capsys.readouterr().out
    assert "full pipeline-window max_memory_allocated" in output
