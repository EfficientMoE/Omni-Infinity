# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import pytest

from omni_infinity.streaming import NativeChunker, StreamInput, StreamRequest


def test_native_input_changes_only_the_next_chunk():
    request = StreamRequest(type="fl2va", prompt="idle", source="native")
    chunker = NativeChunker(chunk_frames=2, chunks=2)
    iterator = chunker.iter_chunks(request)
    first = next(iterator)
    chunker.push_input(StreamInput(action="forward", prompt="run"))
    second = next(iterator)
    assert first.prompt == "idle"
    assert first.instruction is None
    assert second.prompt == "run"
    assert second.action == "forward"
    assert second.instruction == "forward"
    assert second.done is True
    assert first.video_bytes and second.video_bytes
    assert chunker.result is not None
    assert chunker.init
    with pytest.raises(StopIteration):
        next(iterator)


def test_native_key_release_does_not_repeat_the_action():
    request = StreamRequest(type="fl2va", prompt="idle", source="native")
    chunker = NativeChunker(chunk_frames=2, chunks=2)
    iterator = chunker.iter_chunks(request)
    next(iterator)
    chunker.push_input(StreamInput(action="forward", down=False))
    released = next(iterator)
    assert released.action is None
    assert released.instruction is None
