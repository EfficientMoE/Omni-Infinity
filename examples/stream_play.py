# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys

import websockets

from omni_infinity.client import (
    decode_fragmented,
    key_to_input,
    read_stdin_line,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play an Omni stream")
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--player", choices=("av", "ffplay", "mpv"), default="av"
    )
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def _player(command: str) -> subprocess.Popen | None:
    if command == "ffplay":
        return subprocess.Popen(
            ["ffplay", "-autoexit", "-i", "pipe:0"],
            stdin=subprocess.PIPE,
        )
    if command == "mpv":
        return subprocess.Popen(
            ["mpv", "--no-cache", "-"],
            stdin=subprocess.PIPE,
        )
    return None


def _media_bytes(message: dict) -> bytes:
    if message["type"] == "init":
        return base64.b64decode(message["init_b64"])
    if message["type"] == "chunk" and message.get("video_b64"):
        return base64.b64decode(message["video_b64"])
    return b""


async def _send_inputs(socket, stopped: asyncio.Event) -> None:
    while not stopped.is_set():
        line = await read_stdin_line(sys.stdin)
        if not line:
            return
        key = line.rstrip("\n")
        incoming = key_to_input(key, True)
        if incoming is not None:
            await socket.send(incoming.model_dump_json())


async def play(args: argparse.Namespace) -> None:
    messages: list[dict] = []
    process = _player(args.player)
    stopped = asyncio.Event()
    async with websockets.connect(args.url) as socket:
        input_task = (
            asyncio.create_task(_send_inputs(socket, stopped))
            if args.interactive
            else None
        )
        try:
            async for raw in socket:
                message = json.loads(raw)
                messages.append(message)
                if process is not None and process.stdin is not None:
                    process.stdin.write(_media_bytes(message))
                    process.stdin.flush()
                if message["type"] == "chunk":
                    print(
                        f"{message['pts']:.3f} {message['prompt']} "
                        f"{message.get('instruction') or ''}"
                    )
                if message["type"] in {"end", "error"}:
                    break
        finally:
            stopped.set()
            if input_task is not None:
                input_task.cancel()
            if process is not None and process.stdin is not None:
                process.stdin.close()
                process.wait()
    if args.player == "av":
        print(f"decoded {decode_fragmented(messages)} video frames")


def main() -> None:
    asyncio.run(play(parse_args()))


if __name__ == "__main__":
    main()
