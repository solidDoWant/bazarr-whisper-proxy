import io
import wave

import numpy as np


class AudioTooLarge(Exception):
    pass


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


def assert_within_size_limit(pcm_bytes: bytes, max_bytes: int) -> None:
    if len(pcm_bytes) > max_bytes:
        raise AudioTooLarge(f"audio too large: {len(pcm_bytes)} > {max_bytes} bytes")
