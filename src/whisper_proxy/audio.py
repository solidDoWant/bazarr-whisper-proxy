import struct

import numpy as np


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TiB"


class AudioTooLarge(Exception):
    def __init__(self, actual_bytes: int, max_bytes: int) -> None:
        super().__init__(f"audio too large: {actual_bytes} > {max_bytes} bytes")
        self.actual_bytes = actual_bytes
        self.max_bytes = max_bytes
        self.actual_human = _fmt_bytes(actual_bytes)
        self.max_human = _fmt_bytes(max_bytes)


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    # Prepend the 44-byte canonical WAV header directly: one copy of the audio.
    # Writing through wave + BytesIO.getvalue() held two extra copies at once.
    sample_width = 2  # s16le
    block_align = channels * sample_width
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm_bytes),
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        sample_width * 8,
        b"data",
        len(pcm_bytes),
    )
    return header + pcm_bytes


def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    # Convert straight to float32 and scale in place. Dividing the int16 array
    # by a Python float first built a float64 copy twice the size of the result.
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    samples /= np.float32(32768.0)
    return samples


def head_clip(
    pcm_bytes: bytes,
    seconds: float,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bytes:
    frame_size = channels * 2  # s16le
    max_bytes = int(seconds * sample_rate) * frame_size
    return pcm_bytes[:max_bytes] if len(pcm_bytes) > max_bytes else pcm_bytes


def window_clip(
    pcm_bytes: bytes,
    start_sample: int,
    window_samples: int,
    channels: int = 1,
) -> bytes:
    frame_size = channels * 2
    start_byte = start_sample * frame_size
    end_byte = (start_sample + window_samples) * frame_size
    return pcm_bytes[start_byte:end_byte]


def assert_within_size_limit(pcm_bytes: bytes, max_bytes: int) -> None:
    if len(pcm_bytes) > max_bytes:
        raise AudioTooLarge(len(pcm_bytes), max_bytes)
