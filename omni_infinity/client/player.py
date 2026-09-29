# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io

import av

from omni_infinity.streaming import StreamInput

KEY_ACTIONS = {
    "ArrowUp": "forward",
    "ArrowDown": "back",
    "ArrowLeft": "left",
    "ArrowRight": "right",
    " ": "jump",
}


def key_to_input(key: str, down: bool) -> StreamInput | None:
    action = KEY_ACTIONS.get(key)
    if action is None:
        return None
    return StreamInput(key=key, down=down, action=action)


def prompts_from_messages(
    messages: list[dict],
) -> list[tuple[float, str, str | None]]:
    return [
        (
            float(message["pts"]),
            str(message["prompt"]),
            message.get("instruction"),
        )
        for message in messages
        if message.get("type") == "chunk"
    ]


def decode_fragmented(messages: list[dict]) -> int:
    payload = bytearray()
    for message in messages:
        if message.get("type") == "init":
            payload.extend(base64.b64decode(message["init_b64"]))
        elif message.get("type") == "chunk" and message.get("video_b64"):
            payload.extend(base64.b64decode(message["video_b64"]))
    with av.open(io.BytesIO(payload)) as container:
        return sum(1 for _ in container.decode(video=0))
