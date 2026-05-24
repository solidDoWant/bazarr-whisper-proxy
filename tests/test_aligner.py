"""Tests for the forced-aligner wrapper (spec 06)."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import numpy as np
import pytest

from whisper_proxy._types import TranscriptionSegment
from whisper_proxy.aligner import AlignmentFailed, Word, align
from whisper_proxy.audio import pcm_to_float32
from whisper_proxy.config import Settings

FIXTURE_PCM = Path(__file__).parent / "fixtures" / "audio" / "short_en.pcm"
FIXTURE_TRANSCRIPT = "hello world from bazarr"
FIXTURE_LANGUAGE = "en"
SAMPLE_RATE = 16000


def _load_fixture() -> np.ndarray:
    raw = FIXTURE_PCM.read_bytes()
    return pcm_to_float32(raw)


def _default_settings() -> Settings:
    return Settings()


def _single_segment(audio: np.ndarray, text: str = FIXTURE_TRANSCRIPT) -> list[TranscriptionSegment]:
    duration = len(audio) / SAMPLE_RATE
    return [TranscriptionSegment(start_sec=0.0, end_sec=duration, text=text)]


# ---------------------------------------------------------------------------
# Criterion 5: empty transcript raises AlignmentFailed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_transcript_raises():
    audio = _load_fixture()
    with pytest.raises(AlignmentFailed):
        await align(audio, _single_segment(audio, ""), FIXTURE_LANGUAGE, _settings=_default_settings())


@pytest.mark.asyncio
async def test_whitespace_only_transcript_raises():
    audio = _load_fixture()
    with pytest.raises(AlignmentFailed):
        await align(
            audio,
            _single_segment(audio, "   \t\n  "),
            FIXTURE_LANGUAGE,
            _settings=_default_settings(),
        )


@pytest.mark.asyncio
async def test_empty_segments_list_raises():
    audio = _load_fixture()
    with pytest.raises(AlignmentFailed):
        await align(audio, [], FIXTURE_LANGUAGE, _settings=_default_settings())


# ---------------------------------------------------------------------------
# Criterion 6: digital silence does not hang and returns [] or raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_silence_audio_no_hang():
    silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    # Must complete (not hang) and either return [] or raise AlignmentFailed
    try:
        result = await asyncio.wait_for(
            align(
                silence,
                _single_segment(silence),
                FIXTURE_LANGUAGE,
                _settings=_default_settings(),
            ),
            timeout=30.0,
        )
        assert isinstance(result, list)
    except AlignmentFailed:
        pass  # also acceptable


# ---------------------------------------------------------------------------
# Criterion 8: align() does not block the event loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_does_not_block_event_loop(monkeypatch: pytest.MonkeyPatch):
    """Criterion 8: align() offloads to a thread so the event loop stays responsive."""

    def _slow_sync(
        audio: object,
        transcript: object,
        language: object,
        model_path: object,
        device: object,
        precision: object,
        cache_dir: object,
        batch_size: object,
        window_sec: object,
    ) -> list[Word]:
        time.sleep(0.5)
        return []

    monkeypatch.setattr("whisper_proxy.aligner._align_sync", _slow_sync)

    audio = _load_fixture()
    fast_done_at: list[float] = []
    align_done_at: list[float] = []

    async def fast_path() -> None:
        await asyncio.sleep(0.1)
        fast_done_at.append(time.monotonic())

    async def slow_path() -> None:
        await align(
            audio,
            _single_segment(audio),
            FIXTURE_LANGUAGE,
            _settings=_default_settings(),
        )
        align_done_at.append(time.monotonic())

    await asyncio.gather(fast_path(), slow_path())

    assert fast_done_at[0] < align_done_at[0], (
        "event loop was blocked: fast coroutine finished after align()"
    )


# ---------------------------------------------------------------------------
# The remaining criteria require the real ONNX model to be present/downloaded.
# They are marked with a custom mark so CI can gate them separately.
# ---------------------------------------------------------------------------

needs_model = pytest.mark.skipif(
    os.environ.get("SKIP_ALIGNER_INTEGRATION") == "1",
    reason="SKIP_ALIGNER_INTEGRATION=1 — skipping tests that need the ONNX model",
)


@needs_model
@pytest.mark.asyncio
async def test_returns_word_objects():
    """Criterion 1: align() returns ≥ 3 Word objects for the fixture clip."""
    audio = _load_fixture()
    words = await align(
        audio, _single_segment(audio), FIXTURE_LANGUAGE, _settings=_default_settings()
    )
    assert len(words) >= 3, f"expected ≥3 words, got {len(words)}: {words}"
    assert all(isinstance(w, Word) for w in words)


@needs_model
@pytest.mark.asyncio
async def test_word_time_bounds():
    """Criterion 2: every word satisfies 0 ≤ start < end ≤ duration + 0.05."""
    audio = _load_fixture()
    duration = len(audio) / SAMPLE_RATE
    words = await align(
        audio, _single_segment(audio), FIXTURE_LANGUAGE, _settings=_default_settings()
    )
    for w in words:
        assert w.start_sec >= 0.0, f"negative start: {w}"
        assert w.end_sec > w.start_sec, f"end ≤ start: {w}"
        assert w.end_sec <= duration + 0.05, (
            f"end exceeds audio duration: {w} (duration={duration})"
        )


@needs_model
@pytest.mark.asyncio
async def test_words_sorted_by_start():
    """Criterion 3: words are sorted by start_sec ascending."""
    audio = _load_fixture()
    words = await align(
        audio, _single_segment(audio), FIXTURE_LANGUAGE, _settings=_default_settings()
    )
    starts = [w.start_sec for w in words]
    assert starts == sorted(starts), f"words not sorted: {starts}"


@needs_model
@pytest.mark.asyncio
async def test_second_call_is_faster(monkeypatch: pytest.MonkeyPatch):
    """Criterion 4: first call pays cold-load cost; subsequent calls don't."""
    import whisper_proxy.aligner as _mod

    # Force a cold start regardless of test-suite ordering.
    monkeypatch.setattr(_mod, "_compiled_model", None)
    monkeypatch.setattr(_mod, "_tokenizer", None)

    audio = _load_fixture()
    s = _default_settings()

    t0 = time.monotonic()
    await align(audio, _single_segment(audio), FIXTURE_LANGUAGE, _settings=s)
    first = time.monotonic() - t0

    t1 = time.monotonic()
    await align(audio, _single_segment(audio), FIXTURE_LANGUAGE, _settings=s)
    second = time.monotonic() - t1

    assert second < first, f"second call ({second:.2f}s) not faster than first ({first:.2f}s)"


@needs_model
@pytest.mark.asyncio
async def test_concurrent_calls_do_not_corrupt():
    """Criterion 7: two concurrent align() calls each return a consistent word list."""
    audio = _load_fixture()
    s = _default_settings()

    results = await asyncio.gather(
        align(audio, _single_segment(audio), FIXTURE_LANGUAGE, _settings=s),
        align(audio, _single_segment(audio), FIXTURE_LANGUAGE, _settings=s),
    )
    words_a, words_b = results

    # Both should return the same number of words (deterministic alignment).
    assert len(words_a) == len(words_b), (
        f"concurrent calls returned different counts: {len(words_a)} vs {len(words_b)}"
    )
    for wa, wb in zip(words_a, words_b, strict=False):
        assert wa.token == wb.token, f"token mismatch: {wa.token!r} vs {wb.token!r}"
        assert abs(wa.start_sec - wb.start_sec) < 1e-4
        assert abs(wa.end_sec - wb.end_sec) < 1e-4


@needs_model
@pytest.mark.asyncio
async def test_multi_segment_offsets_apply():
    """Words from a non-zero-offset segment should be shifted into global time."""
    audio = _load_fixture()
    duration = len(audio) / SAMPLE_RATE
    # Two halves of the same clip, each labelled with the same transcript.
    # The second segment's words must come back with timestamps >= midpoint.
    half = duration / 2
    segments = [
        TranscriptionSegment(start_sec=0.0, end_sec=half, text=FIXTURE_TRANSCRIPT),
        TranscriptionSegment(start_sec=half, end_sec=duration, text=FIXTURE_TRANSCRIPT),
    ]
    words = await align(audio, segments, FIXTURE_LANGUAGE, _settings=_default_settings())
    # Some words should land in each half.
    assert any(w.start_sec < half for w in words)
    assert any(w.start_sec >= half for w in words)
    # All timestamps inside the audio.
    for w in words:
        assert 0.0 <= w.start_sec <= duration + 0.05
        assert w.end_sec <= duration + 0.05
