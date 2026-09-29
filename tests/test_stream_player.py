# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import os

import numpy as np
import pytest
import torch

from omni_infinity.client.player import (
    decode_fragmented,
    key_to_input,
    prompts_from_messages,
    read_stdin_line,
)
from omni_infinity.streaming import CODEC, fragment_clip


def test_key_map_and_prompt_timeline():
    frames = np.zeros((4, 16, 16, 3), dtype=np.float32)
    audio = torch.zeros(2, 4 * 48_000 // 24, dtype=torch.float32)
    init, fragments = fragment_clip(frames, audio, 48_000, chunk_frames=2)
    messages = [
        {
            "type": "init",
            "codec": CODEC,
            "init_b64": base64.b64encode(init).decode(),
        },
        {
            "type": "chunk",
            "pts": 0.0,
            "prompt": "idle",
            "instruction": None,
            "video_b64": base64.b64encode(fragments[0].video_bytes).decode(),
            "audio_b64": None,
            "done": False,
        },
        {
            "type": "chunk",
            "pts": fragments[1].pts,
            "prompt": "run",
            "instruction": "forward",
            "video_b64": base64.b64encode(fragments[1].video_bytes).decode(),
            "audio_b64": None,
            "done": True,
        },
    ]

    assert key_to_input("ArrowUp", True).action == "forward"
    assert key_to_input("ArrowUp", False).down is False
    assert key_to_input("x", True) is None
    assert prompts_from_messages(messages) == [
        (0.0, "idle", None),
        (fragments[1].pts, "run", "forward"),
    ]
    assert decode_fragmented(messages) == 4


def test_read_stdin_line_is_cancellable():
    async def scenario():
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd) as stream:
                task = asyncio.create_task(read_stdin_line(stream))
                await asyncio.sleep(0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        finally:
            os.close(write_fd)

    asyncio.run(scenario())
