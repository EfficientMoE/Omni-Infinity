# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Issue #24 C1 — exact cross-request condition cache.

The contract under test: a hit requires the *entire* condition to match
(prompt string, then the condition image bytes, in order); a shared
``[Shot 1]`` opener is not a hit; entries are host-resident with an LRU
memory tier and an optional ``.pt`` disk tier; and none of it is on by
default, so the bitwise parity gates never see it.
"""

import pytest
import torch
from PIL import Image

from omni_infinity.caches.condition import (
    ConditionCache,
    build_conditioned_pipeline,
    condition_key,
    declared_inputs,
)
from omni_infinity.runner import GenerationResult, ReferenceRunner


def _image(color):
    return Image.new("RGB", (8, 8), color)


class TestConditionKey:
    def test_identical_condition_gives_identical_key(self):
        assert condition_key("ns", "a red ball", (None,)) == condition_key(
            "ns", "a red ball", (None,)
        )

    def test_shared_shot1_opener_is_not_a_hit(self):
        # The three shipped T2VA examples share only the [Shot 1]
        # opener; a prefix match must NOT produce the same key.
        first = condition_key(
            "ns", "[Shot 1] A chef plates pasta. [Shot 2] Steam rises.", ()
        )
        second = condition_key(
            "ns", "[Shot 1] A chef plates pasta. [Shot 2] Rain falls.", ()
        )
        assert first != second

    def test_prompt_image_boundary_cannot_shift(self):
        # Length-prefixed hashing: moving bytes across the prompt/media
        # boundary must change the key.
        assert condition_key("ns", "ab", (b"c",)) != condition_key(
            "ns", "a", (b"bc",)
        )

    def test_media_order_and_position_matter(self):
        image = _image("red")
        assert condition_key("ns", "p", (image, None)) != condition_key(
            "ns", "p", (None, image)
        )

    def test_pil_content_hashing(self):
        assert condition_key("ns", "p", (_image("red"),)) == condition_key(
            "ns", "p", (_image("red"),)
        )
        assert condition_key("ns", "p", (_image("red"),)) != condition_key(
            "ns", "p", (_image("blue"),)
        )

    def test_namespace_separates_models(self):
        assert condition_key("h3-dense|a", "p", ()) != condition_key(
            "vdn-hybrid|b", "p", ()
        )

    def test_tensor_and_scalar_media(self):
        key = condition_key("ns", "p", (torch.ones(2, 2), 256, 256, 120))
        assert key == condition_key(
            "ns", "p", (torch.ones(2, 2), 256, 256, 120)
        )
        assert key != condition_key(
            "ns", "p", (torch.ones(2, 2), 256, 256, 240)
        )

    def test_unhashable_media_raises_type_error(self):
        with pytest.raises(TypeError, match="canonically hash"):
            condition_key("ns", "p", (object(),))


class TestConditionCache:
    def test_put_get_round_trip_and_stats(self):
        cache = ConditionCache(max_entries=2)
        key = condition_key("ns", "p", ())
        assert cache.get(key) is None
        cache.put(key, {"prompt_embeds": torch.ones(3, 4)})
        entry = cache.get(key)
        assert entry is not None
        assert torch.equal(entry.values["prompt_embeds"], torch.ones(3, 4))
        stats = cache.stats()
        assert stats["hits"] == 1 and stats["misses"] == 1

    def test_put_requires_prompt_embeds(self):
        cache = ConditionCache()
        assert cache.put("k", {"text_token_tags": torch.zeros(2)}) is None
        assert cache.get("k") is None

    def test_none_values_are_dropped_not_stored(self):
        cache = ConditionCache()
        entry = cache.put(
            "k",
            {
                "prompt_embeds": torch.ones(2),
                "text_token_tags": None,
                "condition_latents": None,
            },
        )
        assert set(entry.values) == {"prompt_embeds"}

    def test_entries_are_stored_on_cpu(self):
        cache = ConditionCache()
        entry = cache.put("k", {"prompt_embeds": torch.ones(2)})
        assert entry.values["prompt_embeds"].device.type == "cpu"

    def test_lru_eviction_by_entries(self):
        cache = ConditionCache(max_entries=2)
        for name in ("a", "b", "c"):
            cache.put(name, {"prompt_embeds": torch.ones(1)})
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_lru_eviction_by_bytes_keeps_newest(self):
        cache = ConditionCache(max_entries=10, max_bytes=64)
        cache.put("a", {"prompt_embeds": torch.ones(16)})  # 64 bytes
        cache.put("b", {"prompt_embeds": torch.ones(16)})
        assert cache.get("a") is None
        assert cache.get("b") is not None

    def test_disk_tier_round_trip(self, tmp_path):
        first = ConditionCache(cache_dir=tmp_path)
        first.put(
            "abc123",
            {
                "prompt_embeds": torch.full((2, 3), 7.0),
                "condition_latents": [torch.ones(1, 2)],
            },
        )
        # A fresh process (new cache instance) reads the same file.
        second = ConditionCache(cache_dir=tmp_path)
        entry = second.get("abc123")
        assert entry is not None
        assert torch.equal(
            entry.values["prompt_embeds"], torch.full((2, 3), 7.0)
        )
        assert torch.equal(
            entry.values["condition_latents"][0], torch.ones(1, 2)
        )

    def test_corrupt_disk_entry_is_a_miss(self, tmp_path):
        (tmp_path / "bad.pt").write_bytes(b"not a checkpoint")
        cache = ConditionCache(cache_dir=tmp_path)
        assert cache.get("bad") is None

    @pytest.mark.parametrize(
        "payload",
        [{}, {"prompt_embeds": "not-a-tensor"}],
    )
    def test_invalid_disk_payload_is_a_miss(self, tmp_path, payload):
        torch.save(payload, tmp_path / "bad.pt")
        cache = ConditionCache(cache_dir=tmp_path)
        assert cache.get("bad") is None
        assert cache.stats()["misses"] == 1

    def test_unwritable_disk_tier_fails_open(self, tmp_path, monkeypatch):
        cache = ConditionCache(cache_dir=tmp_path)

        def fail_mkstemp(**_kwargs):
            raise OSError("disk unavailable")

        monkeypatch.setattr("tempfile.mkstemp", fail_mkstemp)
        entry = cache.put("key", {"prompt_embeds": torch.ones(2)})
        assert entry is not None
        assert cache.get("key") is entry


class _RecordingState:
    def __init__(self, values):
        self.values = values


class _FakeFullPipeline:
    """Records call kwargs; exposes encode intermediates in its state."""

    def __init__(self):
        self.calls = []
        self.prompt_embeds = torch.arange(6, dtype=torch.float32).view(2, 3)
        self.text_token_tags = torch.tensor([0, 1])

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _RecordingState(
            {
                "prompt_embeds": self.prompt_embeds,
                "text_token_tags": self.text_token_tags,
                "videos": ["video"],
                "audio": torch.zeros(2, 4),
                "sampling_rate": 48000,
                "latents": torch.zeros(1),
                "audio_latents": torch.zeros(2),
            }
        )


class _FakeConditionedPipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _RecordingState(
            {
                "videos": ["conditioned-video"],
                "audio": torch.zeros(2, 4),
                "sampling_rate": 48000,
                "latents": torch.zeros(1),
                "audio_latents": torch.zeros(2),
            }
        )


class TestRunnerIntegration:
    def _runner(self, monkeypatch, conditioned):
        full = _FakeFullPipeline()
        runner = ReferenceRunner(
            full,
            condition_cache=ConditionCache(),
            condition_namespace="test-ns",
            device="cpu",
        )
        monkeypatch.setattr(
            "omni_infinity.caches.condition.build_conditioned_pipeline",
            lambda pipeline: conditioned,
        )
        return runner, full

    def test_miss_captures_then_hit_skips_the_encoders(self, monkeypatch):
        conditioned = _FakeConditionedPipeline()
        runner, full = self._runner(monkeypatch, conditioned)

        first = runner.generate("a red ball bouncing", seed=0)
        assert isinstance(first, GenerationResult)
        assert len(full.calls) == 1 and not conditioned.calls
        assert runner.condition_cache.stats()["entries"] == 1

        second = runner.generate("a red ball bouncing", seed=1)
        assert len(full.calls) == 1
        assert len(conditioned.calls) == 1
        injected = conditioned.calls[0]
        assert torch.equal(injected["prompt_embeds"], full.prompt_embeds)
        assert torch.equal(injected["text_token_tags"], full.text_token_tags)
        # The generation knobs still flow to the conditioned pipeline.
        assert injected["num_inference_steps"] == 8
        assert injected["generator"].initial_seed() == 1
        assert second.videos == ["conditioned-video"]

    def test_different_prompt_is_a_miss(self, monkeypatch):
        conditioned = _FakeConditionedPipeline()
        runner, full = self._runner(monkeypatch, conditioned)
        runner.generate("a red ball bouncing")
        runner.generate("a blue ball bouncing")
        assert len(full.calls) == 2 and not conditioned.calls

    def test_different_resolution_is_a_miss(self, monkeypatch):
        conditioned = _FakeConditionedPipeline()
        runner, full = self._runner(monkeypatch, conditioned)
        runner.generate("a red ball", resolution="256p")
        runner.generate("a red ball", resolution="512p")
        assert len(full.calls) == 2 and not conditioned.calls

    def test_no_conditioned_pipeline_fails_open(self, monkeypatch):
        runner, full = self._runner(monkeypatch, None)
        runner.generate("a red ball bouncing")
        runner.generate("a red ball bouncing")
        # Both runs use the full pipeline; nothing crashes.
        assert len(full.calls) == 2

    def test_unhashable_reference_fails_open(self, monkeypatch):
        conditioned = _FakeConditionedPipeline()
        runner, full = self._runner(monkeypatch, conditioned)
        runner.generate("p", references=[object()])
        runner.generate("p", references=[object()])
        assert len(full.calls) == 2 and not conditioned.calls
        assert runner.condition_cache.stats()["entries"] == 0

    def test_cache_off_by_default(self):
        # The golden path: a runner without the opt-in flag has no cache
        # and generate() behaves exactly as before.
        full = _FakeFullPipeline()
        runner = ReferenceRunner(full)
        assert runner.condition_cache is None
        runner.generate("a red ball bouncing")
        runner.generate("a red ball bouncing")
        assert len(full.calls) == 2


class TestDeclaredInputFiltering:
    def test_injection_filtered_to_declared_inputs(self, monkeypatch):
        class Spec:
            def __init__(self, name):
                self.name = name

        class Block:
            def __init__(self, names):
                self.inputs = [Spec(name) for name in names]

        class Blocks:
            sub_blocks = {
                "denoise": Block(
                    [
                        "prompt_embeds",
                        "num_inference_steps",
                        "generator",
                        "output_type",
                        "height",
                        "width",
                        "num_frames",
                    ]
                )
            }

        conditioned = _FakeConditionedPipeline()
        conditioned.blocks = Blocks()
        full = _FakeFullPipeline()
        runner = ReferenceRunner(
            full,
            condition_cache=ConditionCache(),
            condition_namespace="ns",
            device="cpu",
        )
        monkeypatch.setattr(
            "omni_infinity.caches.condition.build_conditioned_pipeline",
            lambda pipeline: conditioned,
        )
        runner.generate("a red ball bouncing")
        runner.generate("a red ball bouncing")
        injected = conditioned.calls[0]
        # 'prompt' and 'text_token_tags' are not declared inputs of the
        # reduced pipeline, so they are dropped from the call.
        assert "prompt" not in injected
        assert "text_token_tags" not in injected
        assert "prompt_embeds" in injected


class TestAgainstRealModularBlocks:
    """Structural gates against the real diffusers 0.40 block specs, the
    same pattern as ``test_h3_blocks_accept_runner_call_parameters``."""

    def test_h3_blocks_expose_the_cached_names(self):
        pytest.importorskip("diffusers")
        from diffusers import MiniMaxH3Blocks

        blocks = MiniMaxH3Blocks()
        intermediates = {
            spec.name
            for block in blocks.sub_blocks.values()
            for spec in getattr(block, "intermediate_outputs", [])
        }
        for name in ConditionCache().capture:
            assert name in intermediates, name

    def test_reduced_pipeline_skips_encoders_and_accepts_injection(self):
        pytest.importorskip("diffusers")
        from diffusers import MiniMaxH3Blocks

        pipeline = MiniMaxH3Blocks().init_pipeline()
        conditioned = build_conditioned_pipeline(pipeline)
        assert conditioned is not None
        remaining = set(conditioned.blocks.sub_blocks)
        assert "text_encoder" not in remaining
        assert "vae_encoder" not in remaining
        assert "denoise" in remaining and "decode" in remaining
        inputs = declared_inputs(conditioned)
        for name in ConditionCache().capture:
            assert name in inputs, name
        # And the request generator is consumed only past the encoders,
        # so a hit replays the same RNG stream.
        assert "generator" in inputs

    def test_reduced_pipeline_shares_component_objects(self):
        pytest.importorskip("diffusers")
        from diffusers import MiniMaxH3Blocks

        pipeline = MiniMaxH3Blocks().init_pipeline()
        marker = torch.nn.Linear(1, 1)
        pipeline.update_components(transformer=marker)
        conditioned = build_conditioned_pipeline(pipeline)
        assert conditioned.components["transformer"] is marker
