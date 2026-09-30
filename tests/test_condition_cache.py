# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from omni_infinity.caches.condition import ConditionCache, condition_key


class _UnsafePayload:
    pass


def test_identical_condition_gives_identical_key():
    kwargs = dict(
        namespace="ReferenceRunner:/ckpt",
        prompt="[Shot 1] a",
        media=(b"png", None),
        height=368,
        width=640,
        num_frames=120,
    )
    digest = condition_key(**kwargs)

    assert digest == condition_key(**kwargs)
    assert len(digest) == 64
    int(digest, 16)


def test_shared_shot1_opener_is_not_a_hit():
    common = dict(
        namespace="ReferenceRunner:/ckpt",
        media=(),
        height=368,
        width=640,
        num_frames=120,
    )
    left = condition_key(prompt="[Shot 1] same\n[Shot 2] left", **common)
    right = condition_key(prompt="[Shot 1] same\n[Shot 2] right", **common)

    assert left != right


def test_canvas_and_checkpoint_and_none_slots_are_in_the_key():
    base = dict(
        namespace="ReferenceRunner:/ckpt",
        prompt="p",
        media=(b"a", None),
        height=368,
        width=640,
        num_frames=120,
    )

    assert condition_key(**base) != condition_key(**{**base, "height": 512})
    assert condition_key(**base) != condition_key(
        **{**base, "namespace": "ReferenceRunner:/other"}
    )
    assert condition_key(**base) != condition_key(
        **{**base, "media": (None, b"a")}
    )


def test_unhashable_media_raises_type_error():
    with pytest.raises(TypeError, match="unsupported hash value"):
        condition_key(
            namespace="ReferenceRunner:/ckpt",
            prompt="p",
            media=(object(),),
            height=368,
            width=640,
            num_frames=120,
        )


def test_cache_requires_prompt_embeds_and_tracks_hits_and_misses():
    cache = ConditionCache()

    assert cache.put("missing", {"text_token_tags": torch.ones(1)}) is None
    assert cache.get("missing") is None

    source = torch.zeros(2) + 1
    entry = cache.put("present", {"prompt_embeds": source})
    assert entry is not None
    stored = entry.to("cpu")
    assert stored["prompt_embeds"].device.type == "cpu"
    assert torch.equal(stored["prompt_embeds"], source)

    loaded = cache.get("present")
    assert loaded is not None
    assert torch.equal(loaded.to("cpu")["prompt_embeds"], source)
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_cache_validates_capacity_and_required_capture_names():
    with pytest.raises(ValueError, match="max_entries"):
        ConditionCache(max_entries=0)
    with pytest.raises(ValueError, match="required"):
        ConditionCache(capture=("text_token_tags",))


def test_max_entries_keeps_the_newest_key():
    cache = ConditionCache(max_entries=1)
    cache.put("old", {"prompt_embeds": torch.zeros(1)})
    cache.put("new", {"prompt_embeds": torch.ones(1)})

    assert cache.get("old") is None
    assert cache.get("new") is not None


def test_max_bytes_drops_the_oldest_entry():
    entry_bytes = torch.zeros(2).nbytes
    cache = ConditionCache(max_bytes=entry_bytes)
    cache.put("old", {"prompt_embeds": torch.zeros(2)})
    cache.put("new", {"prompt_embeds": torch.ones(2)})

    assert cache.get("old") is None
    loaded = cache.get("new")
    assert loaded is not None
    assert torch.equal(loaded.to("cpu")["prompt_embeds"], torch.ones(2))


def test_disk_cache_round_trip_uses_cpu_tensors(tmp_path):
    source = torch.arange(3)
    first = ConditionCache(cache_dir=tmp_path)
    first.put(
        "disk",
        {
            "prompt_embeds": source,
            "condition_latents": None,
            "ignored": torch.ones(1),
        },
    )

    second = ConditionCache(cache_dir=tmp_path)
    loaded = second.get("disk")

    assert loaded is not None
    assert set(loaded.to("cpu")) == {"prompt_embeds"}
    assert torch.equal(loaded.to("cpu")["prompt_embeds"], source)


def test_truncated_disk_entry_is_a_miss(tmp_path):
    (tmp_path / "broken.pt").write_bytes(b"not a torch archive")

    cache = ConditionCache(cache_dir=tmp_path)

    assert cache.get("broken") is None


def test_disk_load_rejects_non_tensor_payload_with_weights_only(tmp_path):
    torch.save(_UnsafePayload(), tmp_path / "unsafe.pt")

    cache = ConditionCache(cache_dir=tmp_path)

    assert cache.get("unsafe") is None
