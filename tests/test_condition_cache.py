# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import pytest

from omni_infinity.caches.condition import condition_key


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
