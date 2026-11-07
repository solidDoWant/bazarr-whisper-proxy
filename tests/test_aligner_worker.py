"""The aligner worker process: lifecycle, idle shutdown, and the audio file handoff.

Most tests swap the process pool for an in-process thread pool so the job can be
stubbed; test_real_worker_process_round_trip spawns the real worker. None of
them need the aligner model or a GPU.
"""

import asyncio
import concurrent.futures
import os
import struct
import time
from collections.abc import Iterator
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import numpy as np
import pytest

import whisper_proxy.aligner as aligner
from whisper_proxy._types import TranscriptionSegment, Word
from whisper_proxy.audio import pcm_to_float32
from whisper_proxy.config import Settings

_AUDIO = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
_SEGMENTS = [TranscriptionSegment(start_sec=0.0, end_sec=1.0, text="hello")]


class _Workers:
    """Records worker pools created by the stubbed factory and files the job saw."""

    def __init__(self) -> None:
        self.created: list[concurrent.futures.ThreadPoolExecutor] = []
        self.seen_paths: list[str] = []
        self.fail_with: BaseException | None = None

    def make(self, *_args: Any) -> concurrent.futures.Executor:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.created.append(pool)
        return pool

    def job(self, path: str, n_samples: int, *_args: Any) -> list[Word]:
        self.seen_paths.append(path)
        audio = np.memmap(path, dtype=np.float32, mode="r", shape=(n_samples,))
        np.testing.assert_array_equal(audio, _AUDIO)
        if self.fail_with is not None:
            raise self.fail_with
        return [Word(token="hello", start_sec=0.0, end_sec=0.5)]


@pytest.fixture
def workers(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Workers]:
    w = _Workers()
    monkeypatch.setattr(aligner, "_make_worker", w.make)
    monkeypatch.setattr(aligner, "_align_file_job", w.job)
    _reset()
    yield w
    _reset()


def _reset() -> None:
    aligner.shutdown_worker()
    with aligner._worker_lock:
        aligner._active_calls = 0


def _settings(idle_unload_sec: float) -> Settings:
    return Settings(ALIGNER_IDLE_UNLOAD_SEC=idle_unload_sec)


async def _wait_for(predicate: Any, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


async def test_worker_stops_after_idle_period(workers: _Workers) -> None:
    words = await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=_settings(0.05))

    assert [w.token for w in words] == ["hello"]
    assert aligner._worker is workers.created[0]
    assert await _wait_for(lambda: aligner._worker is None)
    assert workers.created[0]._shutdown


async def test_zero_keeps_worker_running(workers: _Workers) -> None:
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=_settings(0))

    await asyncio.sleep(0.2)
    assert aligner._worker is workers.created[0]
    assert aligner._shutdown_timer is None


async def test_not_stopped_while_another_call_is_running(workers: _Workers) -> None:
    s = _settings(0.05)
    other = aligner._acquire_worker(s)  # simulate an alignment still in progress
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)
    await asyncio.sleep(0.2)
    assert aligner._worker is other

    aligner._release_worker(other, 0.05, broken=False)
    assert await _wait_for(lambda: aligner._worker is None)


async def test_new_call_cancels_pending_shutdown(workers: _Workers) -> None:
    s = _settings(0.3)
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)
    await asyncio.sleep(0.2)
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)
    await asyncio.sleep(0.2)

    # 0.4s after the first call but only 0.2s after the second: still running.
    assert aligner._worker is not None
    assert len(workers.created) == 1


async def test_stale_timer_does_not_stop_reused_worker(workers: _Workers) -> None:
    s = _settings(0)
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)
    with aligner._worker_lock:
        armed_generation = aligner._use_generation
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)

    aligner._shutdown_if_idle(armed_generation, 0.05)

    assert aligner._worker is workers.created[0]


async def test_new_worker_after_idle_stop(workers: _Workers) -> None:
    s = _settings(0.05)
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)
    assert await _wait_for(lambda: aligner._worker is None)

    words = await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)

    assert [w.token for w in words] == ["hello"]
    assert len(workers.created) == 2


async def test_audio_file_deleted_after_success_and_failure(workers: _Workers) -> None:
    s = _settings(0)
    await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)
    workers.fail_with = aligner.AlignmentFailed("boom")
    with pytest.raises(aligner.AlignmentFailed, match="boom"):
        await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)

    assert len(workers.seen_paths) == 2
    assert not any(os.path.exists(p) for p in workers.seen_paths)


async def test_dead_worker_fails_request_and_is_replaced(workers: _Workers) -> None:
    s = _settings(0)
    workers.fail_with = BrokenProcessPool("worker died")
    with pytest.raises(aligner.AlignmentFailed, match="worker process died"):
        await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)
    assert aligner._worker is None

    workers.fail_with = None
    words = await aligner.align(_AUDIO, _SEGMENTS, "en", _settings=s)

    assert [w.token for w in words] == ["hello"]
    assert len(workers.created) == 2


async def test_cancelled_request_keeps_worker_in_use_until_job_ends(
    workers: _Workers, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    loop = asyncio.get_running_loop()

    def slow_job(path: str, n_samples: int, *args: Any) -> list[Word]:
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return workers.job(path, n_samples, *args)

    monkeypatch.setattr(aligner, "_align_file_job", slow_job)
    task = asyncio.create_task(aligner.align(_AUDIO, _SEGMENTS, "en", _settings=_settings(0.05)))
    assert await _wait_for(lambda: aligner._active_calls == 1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)
    assert aligner._worker is not None  # job still running in the worker

    release.set()
    assert await _wait_for(lambda: aligner._worker is None)
    assert not any(os.path.exists(p) for p in workers.seen_paths)


def test_audio_file_from_pcm_matches_pcm_to_float32() -> None:
    pcm = struct.pack("<5h", 0, 1, -1, 32767, -32768)
    audio = aligner.AudioFile.from_pcm(pcm)
    try:
        data = np.memmap(audio.path, dtype=np.float32, mode="r", shape=(audio.n_samples,))
        np.testing.assert_array_equal(data, pcm_to_float32(pcm))
    finally:
        audio.delete()
    assert not os.path.exists(audio.path)


def test_empty_audio_is_an_alignment_failure() -> None:
    with pytest.raises(aligner.AlignmentFailed, match="no audio"):
        aligner.AudioFile.from_pcm(b"")


async def test_real_worker_process_round_trip() -> None:
    """Spawn the real worker. An empty transcript fails before any model load,
    so this exercises the file handoff and exception pickling without a model."""
    _reset()
    try:
        segments = [TranscriptionSegment(start_sec=0.0, end_sec=1.0, text="   ")]
        with pytest.raises(aligner.AlignmentFailed, match="transcript is empty"):
            await aligner.align(_AUDIO, segments, "en", _settings=_settings(0))
        assert isinstance(aligner._worker, concurrent.futures.ProcessPoolExecutor)
    finally:
        _reset()
