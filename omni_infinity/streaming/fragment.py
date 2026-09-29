# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import io
from dataclasses import dataclass
from fractions import Fraction

import av
import numpy as np
import torch

CODEC: str = 'video/mp4; codecs="avc1.42E01E,mp4a.40.2"'


@dataclass(frozen=True)
class MediaFragment:
    index: int
    pts: float
    duration: float
    keyframe: bool
    video_bytes: bytes


def _box_end(data: bytes, offset: int) -> int:
    size = int.from_bytes(data[offset:offset + 4], "big")
    if size == 1:
        return offset + int.from_bytes(data[offset + 8:offset + 16], "big")
    if size == 0:
        return len(data)
    return offset + size


def _find_box(data: bytes, box_type: bytes, start: int = 0) -> int:
    offset = start
    while offset + 8 <= len(data):
        if data[offset + 4:offset + 8] == box_type:
            return offset
        next_offset = _box_end(data, offset)
        if next_offset <= offset:
            break
        offset = next_offset
    raise ValueError(f"MP4 box {box_type!r} not found")


def _split_fragments(
    data: bytes,
    frame_count: int,
    chunk_frames: int,
    fps: int,
    video_keyframes: tuple[bool, ...],
) -> tuple[bytes, tuple[MediaFragment, ...]]:
    moov = _find_box(data, b"moov")
    init_end = _box_end(data, moov)
    fragment_starts = []
    offset = init_end
    while offset < len(data):
        moof = _find_box(data, b"moof", offset)
        fragment_starts.append(moof)
        offset = _box_end(data, moof)
        try:
            offset = _find_box(data, b"moof", offset)
        except ValueError:
            break

    expected_fragments = (frame_count + chunk_frames - 1) // chunk_frames
    if len(fragment_starts) != expected_fragments:
        raise RuntimeError("encoded GOP boundaries do not match chunk_frames")
    if len(video_keyframes) < frame_count:
        raise RuntimeError("encoded video packet count does not match frames")

    fragments = []
    for index, start in enumerate(fragment_starts):
        end = (
            fragment_starts[index + 1]
            if index + 1 < len(fragment_starts)
            else len(data)
        )
        frames_in_fragment = min(
            chunk_frames, frame_count - index * chunk_frames
        )
        keyframe = video_keyframes[index * chunk_frames]
        if not keyframe:
            raise RuntimeError(
                "encoded fragment does not start with a keyframe"
            )
        fragments.append(
            MediaFragment(
                index=index,
                pts=index * chunk_frames / fps,
                duration=frames_in_fragment / fps,
                keyframe=keyframe,
                video_bytes=data[start:end],
            )
        )
    return data[:init_end], tuple(fragments)


def fragment_clip(
    frames: np.ndarray,
    audio: torch.Tensor,
    sample_rate: int,
    *,
    fps: int = 24,
    chunk_frames: int = 24,
) -> tuple[bytes, tuple[MediaFragment, ...]]:
    if not isinstance(frames, np.ndarray) or frames.ndim != 4:
        raise ValueError("frames must have shape (N, H, W, 3)")
    if frames.shape[0] <= 0 or frames.shape[1] <= 0 or frames.shape[2] <= 0:
        raise ValueError("frames must have positive dimensions")
    if frames.shape[3] != 3 or frames.dtype != np.float32:
        raise ValueError("frames must be float32 with shape (N, H, W, 3)")
    if not np.isfinite(frames).all() or not ((0 <= frames).all() and
                                             (frames <= 1).all()):
        raise ValueError("frames must contain finite values in [0, 1]")
    if not isinstance(audio, torch.Tensor) or audio.ndim != 2:
        raise ValueError("audio must have shape (2, samples)")
    if audio.shape[0] != 2 or audio.shape[1] <= 0:
        raise ValueError("audio must have shape (2, samples)")
    if audio.dtype != torch.float32:
        raise ValueError("audio must be float32")
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    if not isinstance(chunk_frames, int) or chunk_frames <= 0:
        raise ValueError("chunk_frames must be a positive integer")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")

    frame_count, height, width, _ = frames.shape
    output = io.BytesIO()
    container = av.open(
        output,
        mode="w",
        format="mp4",
        options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
    )
    video_stream = container.add_stream("libx264", rate=fps)
    video_stream.width = width
    video_stream.height = height
    video_stream.pix_fmt = "yuv420p"
    video_stream.time_base = Fraction(1, fps)
    video_stream.gop_size = chunk_frames
    video_stream.codec_context.max_b_frames = 0
    video_stream.options = {
        "g": str(chunk_frames),
        "keyint_min": str(chunk_frames),
        "sc_threshold": "0",
    }

    audio_stream = container.add_stream("aac", rate=sample_rate)
    audio_stream.layout = "stereo"
    audio_stream.sample_rate = sample_rate

    video_packets = []
    for index, frame in enumerate(frames):
        video_frame = av.VideoFrame.from_ndarray(
            np.clip(frame * 255, 0, 255).astype(np.uint8), format="rgb24"
        )
        video_frame.pts = index
        video_frame.time_base = Fraction(1, fps)
        if index % chunk_frames == 0:
            video_frame.pict_type = av.video.frame.PictureType.I
        video_packets.extend(video_stream.encode(video_frame))
    video_packets.extend(video_stream.encode())
    video_keyframes = tuple(packet.is_keyframe for packet in video_packets)

    audio_array = audio.detach().cpu().numpy()
    audio_frame = av.AudioFrame.from_ndarray(
        audio_array, format="fltp", layout="stereo"
    )
    audio_frame.sample_rate = sample_rate
    audio_packets = audio_stream.encode(audio_frame)
    audio_packets.extend(audio_stream.encode())
    packets = video_packets + audio_packets
    packets.sort(
        key=lambda packet: (
            (packet.dts if packet.dts is not None else packet.pts)
            * packet.time_base
            if packet.dts is not None or packet.pts is not None
            else float("inf")
        )
    )
    for packet in packets:
        container.mux(packet)
    container.close()

    return _split_fragments(
        output.getvalue(),
        frame_count,
        chunk_frames,
        fps,
        video_keyframes,
    )
