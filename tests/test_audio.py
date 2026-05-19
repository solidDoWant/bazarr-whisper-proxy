import struct
import wave
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from whisper_proxy.audio import (
    AudioTooLarge,
    assert_within_size_limit,
    head_clip,
    pcm_to_float32,
    pcm_to_wav,
)

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # s16le


def make_pcm(seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    n = int(seconds * sample_rate)
    samples = (np.arange(n, dtype=np.int16) % 1000).astype(np.int16)
    return samples.tobytes()


# --- Criterion 1: soundfile reads back with sample-exact equality ---


def test_pcm_to_wav_soundfile_roundtrip() -> None:
    pcm = make_pcm(1.0)
    wav = pcm_to_wav(pcm)
    data, sr = sf.read(BytesIO(wav), dtype="int16")
    assert sr == SAMPLE_RATE
    expected = np.frombuffer(pcm, dtype=np.int16)
    np.testing.assert_array_equal(data, expected)


# --- Criterion 2: wave.open validates format fields ---


def test_pcm_to_wav_wave_open_metadata() -> None:
    pcm = make_pcm(0.5)
    wav = pcm_to_wav(pcm)
    with wave.open(BytesIO(wav)) as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2


def test_pcm_to_wav_soundfile_accepts_as_valid() -> None:
    pcm = make_pcm(0.5)
    wav = pcm_to_wav(pcm)
    _, sr = sf.read(BytesIO(wav))
    assert sr == SAMPLE_RATE


# --- Criterion 3: float32 conversion is within 1 ULP of value/32768.0 ---


def test_pcm_to_float32_known_values() -> None:
    int_vals = [0, 32767, -32768]
    pcm = struct.pack("<3h", *int_vals)
    result = pcm_to_float32(pcm)
    assert result.dtype == np.float32
    assert len(result) == 3
    expected = np.array([v / 32768.0 for v in int_vals], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


# --- Criterion 4: head_clip of 60s returns exactly 30*16000*2 bytes ---


def test_head_clip_truncates_to_requested_seconds() -> None:
    pcm = make_pcm(60.0)
    clipped = head_clip(pcm, seconds=30.0)
    assert len(clipped) == 30 * 16000 * 2


# --- Criterion 5: head_clip on shorter input returns input unchanged ---


def test_head_clip_short_input_unchanged() -> None:
    pcm = make_pcm(5.0)
    clipped = head_clip(pcm, seconds=30.0)
    assert clipped == pcm


# --- Criterion 6: result byte length is always a multiple of channels * sample_width ---


def test_head_clip_always_sample_aligned() -> None:
    pcm = make_pcm(60.0) + b"\x00"  # extra byte to stress alignment
    for seconds in [1.0, 5.0, 30.0, 60.0]:
        clipped = head_clip(pcm, seconds=seconds)
        assert len(clipped) % (CHANNELS * SAMPLE_WIDTH) == 0


def test_head_clip_exact_30s_is_frame_aligned() -> None:
    pcm = make_pcm(60.0)
    clipped = head_clip(pcm, seconds=30.0)
    assert len(clipped) % (CHANNELS * SAMPLE_WIDTH) == 0


# --- Criterion 7: assert_within_size_limit ---


def test_size_limit_passes_at_exact_limit() -> None:
    pcm = b"\x00" * 100
    assert_within_size_limit(pcm, max_bytes=100)


def test_size_limit_passes_below_limit() -> None:
    pcm = b"\x00" * 50
    assert_within_size_limit(pcm, max_bytes=100)


def test_size_limit_raises_when_over_by_one() -> None:
    pcm = b"\x00" * 101
    with pytest.raises(AudioTooLarge):
        assert_within_size_limit(pcm, max_bytes=100)


def test_size_limit_raises_for_any_nonzero_on_zero_limit() -> None:
    with pytest.raises(AudioTooLarge):
        assert_within_size_limit(b"\x00", max_bytes=0)


# --- Criterion 8: all helpers are pure (same input → byte-equal output) ---


def test_pcm_to_wav_is_pure() -> None:
    pcm = make_pcm(1.0)
    assert pcm_to_wav(pcm) == pcm_to_wav(pcm)


def test_pcm_to_float32_is_pure() -> None:
    pcm = make_pcm(1.0)
    np.testing.assert_array_equal(pcm_to_float32(pcm), pcm_to_float32(pcm))


def test_head_clip_is_pure() -> None:
    pcm = make_pcm(60.0)
    assert head_clip(pcm, 30.0) == head_clip(pcm, 30.0)


def test_assert_within_size_limit_is_pure() -> None:
    pcm = b"\x00" * 50
    assert_within_size_limit(pcm, max_bytes=100)
    assert_within_size_limit(pcm, max_bytes=100)  # no state change
