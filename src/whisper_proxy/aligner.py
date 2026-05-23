from __future__ import annotations

import asyncio
import concurrent.futures
import ctypes
import gc
import logging
import math
import os
import threading
from typing import Any

import numpy as np
import onnxruntime
from ctc_forced_aligner import (
    MODEL_URL,
    SAMPLING_FREQ,
    Tokenizer,
    ensure_onnx_model,
    get_alignments,
    get_spans,
    postprocess_results,
    preprocess_text,
    time_to_frame,
)

from ._types import TranscriptionSegment, Word
from .config import Settings

logger = logging.getLogger(__name__)

# ISO 639-1 → ISO 639-3 for ctc_forced_aligner's text_normalize
_ISO2_TO_ISO3: dict[str, str] = {
    "af": "afr",
    "sq": "sqi",
    "am": "amh",
    "ar": "ara",
    "hy": "hye",
    "az": "aze",
    "be": "bel",
    "bn": "ben",
    "bs": "bos",
    "bg": "bul",
    "ca": "cat",
    "zh": "cmn",
    "hr": "hrv",
    "cs": "ces",
    "da": "dan",
    "nl": "nld",
    "en": "eng",
    "et": "est",
    "fi": "fin",
    "fr": "fra",
    "ka": "kat",
    "de": "deu",
    "el": "ell",
    "gu": "guj",
    "ha": "hau",
    "he": "heb",
    "hi": "hin",
    "hu": "hun",
    "id": "ind",
    "it": "ita",
    "ja": "jpn",
    "kn": "kan",
    "kk": "kaz",
    "ky": "kir",
    "ko": "kor",
    "lv": "lav",
    "lt": "lit",
    "mk": "mkd",
    "ms": "msa",
    "ml": "mal",
    "mt": "mlt",
    "mr": "mar",
    "mn": "mon",
    "my": "mya",
    "ne": "nep",
    "no": "nor",
    "ps": "pus",
    "fa": "fas",
    "pl": "pol",
    "pt": "por",
    "ro": "ron",
    "ru": "rus",
    "sr": "srp",
    "si": "sin",
    "sk": "slk",
    "sl": "slv",
    "so": "som",
    "es": "spa",
    "sw": "swa",
    "sv": "swe",
    "tl": "tgl",
    "ta": "tam",
    "te": "tel",
    "th": "tha",
    "tr": "tur",
    "uk": "ukr",
    "ur": "urd",
    "uz": "uzb",
    "vi": "vie",
    "cy": "cym",
    "xh": "xho",
    "yi": "yid",
    "yo": "yor",
    "zu": "zul",
}


class AlignmentFailed(Exception):
    pass


# Process-level model state — loaded once, reused on all subsequent calls.
_load_lock = threading.Lock()
_ort_session: Any | None = None
_tokenizer: Any | None = None

_cpu_count = (
    len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
)
# Cap ONNX intra-op threads: each thread gets its own arena allocation. More threads
# means more arenas, not necessarily more throughput for audio processing.
_ONNX_THREADS = min(_cpu_count, 4)

_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_cpu_count, thread_name_prefix="aligner"
)

# libc handle for malloc_trim — releases glibc arena pages back to the OS after
# inference, since ONNX arenas are backed by standard malloc.
try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.malloc_trim.restype = ctypes.c_int
    _libc.malloc_trim.argtypes = [ctypes.c_size_t]
except Exception:
    _libc = None


def _malloc_trim() -> None:
    if _libc is not None:
        _libc.malloc_trim(0)


def _ensure_model(model_path: str) -> None:
    global _ort_session, _tokenizer
    if _ort_session is not None:
        return

    with _load_lock:
        if _ort_session is not None:
            return

        expanded = os.path.expanduser(model_path)
        ensure_onnx_model(expanded, MODEL_URL)
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = _ONNX_THREADS
        # Disable the CPU memory arena so ORT returns pages to the allocator
        # rather than holding onto peak usage permanently between calls.
        opts.enable_cpu_mem_arena = False
        # Disable memory pattern pre-planning — paired with arena=False this
        # means each inference allocates only what it needs and frees promptly.
        opts.enable_mem_pattern = False
        _ort_session = onnxruntime.InferenceSession(expanded, sess_options=opts)
        _tokenizer = Tokenizer()
        logger.info("aligner model loaded from %s", expanded)


def _to_iso3(lang: str) -> str:
    return _ISO2_TO_ISO3.get(lang.lower(), lang)


def _generate_emissions_streaming(
    session: Any,
    audio_waveform: np.ndarray,
    window_length: int,
    context_length: int,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    """Memory-efficient replacement for ctc_forced_aligner.generate_emissions.

    The library version pre-allocates all windows as a single numpy array
    (~0.5 GB for a 2-hour film) before running any inference. This version
    streams window slices directly into each ONNX batch, keeping only one
    batch of input in memory at a time.
    """
    context = context_length * SAMPLING_FREQ
    window = window_length * SAMPLING_FREQ
    extension = math.ceil(audio_waveform.shape[0] / window) * window - audio_waveform.shape[0]
    padded = np.pad(audio_waveform, (context, context + extension), mode="constant")
    num_windows = (padded.shape[0] - 2 * context) // window

    emissions_list: list[np.ndarray] = []
    i = 0
    while i < num_windows:
        batch_indices = range(i, min(i + batch_size, num_windows))
        # Build only this batch's windows — not all windows at once.
        batch = np.stack(
            [padded[j * window : j * window + window + 2 * context] for j in batch_indices],
            axis=0,
        ).astype(np.float32)
        outputs = session.run(["logits"], {"input_values": batch})
        emissions_list.append(outputs[0])
        i += len(batch_indices)

    emissions = np.concatenate(emissions_list, axis=0)
    del emissions_list

    ctx_frames = time_to_frame(context_length)
    # Trim context frames from each window's output.
    end_trim = -ctx_frames + 1 if ctx_frames > 1 else None
    emissions = emissions[:, ctx_frames:end_trim, :]
    emissions = emissions.reshape(-1, emissions.shape[-1])

    ext_frames = time_to_frame(extension / SAMPLING_FREQ)
    if ext_frames > 0:
        emissions = emissions[:-ext_frames, :]

    # Log-softmax via the log-sum-exp trick: avoids materialising multiple
    # full-size exp() copies at once, unlike the library's naive formulation.
    emissions = emissions.astype(np.float32)
    max_vals = emissions.max(axis=-1, keepdims=True)
    emissions -= max_vals
    np.exp(emissions, out=emissions)
    log_sum = np.log(emissions.sum(axis=-1, keepdims=True))
    np.log(emissions, out=emissions)
    emissions -= log_sum

    # Append <star> token column (zeros in log-space = prob 1 unnormalised,
    # matching the library's convention).
    star_col = np.zeros((emissions.shape[0], 1), dtype=np.float32)
    emissions = np.concatenate([emissions, star_col], axis=1)

    stride = math.ceil(audio_waveform.shape[0] * 1000 / emissions.shape[0] / SAMPLING_FREQ)
    return emissions, stride


def _align_sync(
    audio: np.ndarray,
    transcript: str,
    language: str,
    model_path: str,
    batch_size: int,
    window_sec: int,
) -> list[Word]:
    if not transcript.strip():
        raise AlignmentFailed("transcript is empty")

    # Digital silence — return empty rather than feeding garbage to the model.
    if np.abs(audio).max() < 1e-6:
        return []

    _ensure_model(model_path)

    iso3 = _to_iso3(language)

    try:
        emissions, stride = _generate_emissions_streaming(
            _ort_session,
            audio,
            window_length=window_sec,
            context_length=2,
            batch_size=batch_size,
        )
    except Exception as exc:
        raise AlignmentFailed(f"emission generation failed: {exc}") from exc

    tokens_starred, text_starred = preprocess_text(
        transcript,
        romanize=True,
        language=iso3,
    )

    if not any(t != "<star>" for t in tokens_starred):
        raise AlignmentFailed("transcript produced no alignable tokens after normalization")

    try:
        segments, scores, blank = get_alignments(emissions, tokens_starred, _tokenizer)
        del emissions
        spans = get_spans(tokens_starred, segments, blank)
        results = postprocess_results(text_starred, spans, stride, scores)
    except (AssertionError, ValueError, Exception) as exc:
        raise AlignmentFailed(f"alignment failed: {exc}") from exc
    finally:
        # Return glibc arena pages to the OS now that inference is done.
        gc.collect()
        _malloc_trim()

    return sorted(
        (Word(token=r["text"], start_sec=r["start"], end_sec=r["end"]) for r in results),
        key=lambda w: w.start_sec,
    )


def _align_segments_sync(
    audio: np.ndarray,
    segments: list[TranscriptionSegment],
    language: str,
    model_path: str,
    batch_size: int,
    window_sec: int,
) -> list[Word]:
    if not segments:
        raise AlignmentFailed("no segments to align")

    total_samples = audio.shape[0]
    all_words: list[Word] = []
    succeeded = 0
    last_error: Exception | None = None

    for seg in segments:
        start_sample = max(0, round(seg.start_sec * SAMPLING_FREQ))
        end_sample = min(total_samples, round(seg.end_sec * SAMPLING_FREQ))
        if end_sample <= start_sample:
            logger.warning(
                "skipping segment with empty audio range: %.3fs-%.3fs",
                seg.start_sec,
                seg.end_sec,
            )
            continue

        slice_audio = audio[start_sample:end_sample]

        try:
            seg_words = _align_sync(
                slice_audio,
                seg.text,
                language,
                model_path,
                batch_size,
                window_sec,
            )
        except AlignmentFailed as exc:
            # A single failing segment shouldn't lose the rest of the alignment.
            # Track the last error so we can surface it if every segment fails.
            logger.warning(
                "segment alignment failed (%.3fs-%.3fs): %s",
                seg.start_sec,
                seg.end_sec,
                exc,
            )
            last_error = exc
            continue
        finally:
            del slice_audio

        succeeded += 1
        offset = seg.start_sec
        all_words.extend(
            Word(token=w.token, start_sec=w.start_sec + offset, end_sec=w.end_sec + offset)
            for w in seg_words
        )

    if succeeded == 0:
        raise AlignmentFailed(f"all {len(segments)} segments failed to align: {last_error}")

    all_words.sort(key=lambda w: w.start_sec)
    return all_words


async def align(
    audio_float32: np.ndarray,
    segments: list[TranscriptionSegment],
    language: str,
    *,
    _settings: Settings | None = None,
) -> list[Word]:
    s = _settings if _settings is not None else Settings()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor,
        _align_segments_sync,
        audio_float32,
        segments,
        language,
        s.ALIGNER_MODEL_PATH,
        s.ALIGNER_BATCH_SIZE,
        s.ALIGNER_WINDOW_SEC,
    )
