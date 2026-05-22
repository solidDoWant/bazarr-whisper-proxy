import io
import wave

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
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # s16le = 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return (samples / 32768.0).astype(np.float32)


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
