"""Idle unloading of the compiled aligner model (ALIGNER_IDLE_UNLOAD_SEC).

These tests stub out model loading and alignment, so they run without the
aligner model or a GPU.
"""

import asyncio
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import whisper_proxy.aligner as aligner
from whisper_proxy._types import TranscriptionSegment, Word
from whisper_proxy.config import Settings

_AUDIO = np.zeros(16000, dtype=np.float32)
_SEGMENTS = [TranscriptionSegment(start_sec=0.0, end_sec=1.0, text="hello")]


@pytest.fixture(autouse=True)
def stub_model(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Replace model compile + alignment with cheap stand-ins.

    Yields a single-element list counting how many times the model was loaded.
    """
    loads = [0]

    def fake_ensure_model(*_args: Any) -> None:
        with aligner._load_lock:
            if aligner._compiled_model is None:
                aligner._compiled_model = object()
                loads[0] += 1

    def fake_align_segments_sync(*args: Any) -> list[Word]:
        fake_ensure_model(*args)
        return [Word(token="hello", start_sec=0.0, end_sec=0.5)]

    monkeypatch.setattr(aligner, "_ensure_model", fake_ensure_model)
    monkeypatch.setattr(aligner, "_align_segments_sync", fake_align_segments_sync)
    _reset_state()
    yield loads
    _reset_state()


def _reset_state() -> None:
    with aligner._load_lock:
        if aligner._unload_timer is not None:
            aligner._unload_timer.cancel()
        aligner._unload_timer = None
        aligner._compiled_model = None
        aligner._active_calls = 0


def _settings(idle_unload_sec: float) -> Settings:
    return Settings(ALIGNER_IDLE_UNLOAD_SEC=idle_unload_sec)


async def _wait_for(predicate: Any, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


async def test_model_unloads_after_idle_period() -> None:
    words = await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=_settings(0.05))

    assert [w.token for w in words] == ["hello"]
    assert aligner._compiled_model is not None
    assert await _wait_for(lambda: aligner._compiled_model is None)


async def test_zero_keeps_model_loaded() -> None:
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=_settings(0))

    await asyncio.sleep(0.2)
    assert aligner._compiled_model is not None
    assert aligner._unload_timer is None


async def test_not_unloaded_while_another_call_is_running() -> None:
    aligner._begin_use()  # simulate an alignment still in progress
    try:
        await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=_settings(0.05))
        await asyncio.sleep(0.2)
        assert aligner._compiled_model is not None
    finally:
        aligner._end_use(0.05)

    assert await _wait_for(lambda: aligner._compiled_model is None)


async def test_new_call_cancels_pending_unload(stub_model: list[int]) -> None:
    s = _settings(0.3)
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)
    await asyncio.sleep(0.2)
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)
    await asyncio.sleep(0.2)

    # 0.4s after the first call but only 0.2s after the second: still loaded.
    assert aligner._compiled_model is not None
    assert stub_model[0] == 1


async def test_stale_timer_does_not_unload_after_reuse() -> None:
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=_settings(0))
    with aligner._load_lock:
        armed_generation = aligner._use_generation
    aligner._begin_use()
    aligner._end_use(0)

    aligner._unload_if_idle(armed_generation, 0.05)

    assert aligner._compiled_model is not None


async def test_model_reloads_after_unload(stub_model: list[int]) -> None:
    s = _settings(0.05)
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)
    assert await _wait_for(lambda: aligner._compiled_model is None)

    words = await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)

    assert [w.token for w in words] == ["hello"]
    assert stub_model[0] == 2
