from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnxruntime
from ctc_forced_aligner import (
    MODEL_URL,
    Tokenizer,
    ensure_onnx_model,
    generate_emissions,
    get_alignments,
    get_spans,
    postprocess_results,
    preprocess_text,
)

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


@dataclass(frozen=True)
class Word:
    token: str
    start_sec: float
    end_sec: float


# Process-level model state — loaded once, reused on all subsequent calls.
_load_lock = threading.Lock()
_ort_session: Any | None = None
_tokenizer: Any | None = None

_cpu_count = (
    len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
)
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_cpu_count, thread_name_prefix="aligner"
)


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
        opts.intra_op_num_threads = _cpu_count
        _ort_session = onnxruntime.InferenceSession(expanded, sess_options=opts)
        _tokenizer = Tokenizer()
        logger.info("aligner model loaded from %s", expanded)


def _to_iso3(lang: str) -> str:
    return _ISO2_TO_ISO3.get(lang.lower(), lang)


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
        emissions, stride = generate_emissions(
            _ort_session,
            audio,
            window_length=window_sec,
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
        spans = get_spans(tokens_starred, segments, blank)
        results = postprocess_results(text_starred, spans, stride, scores)
    except (AssertionError, ValueError, Exception) as exc:
        raise AlignmentFailed(f"alignment failed: {exc}") from exc

    return sorted(
        (Word(token=r["text"], start_sec=r["start"], end_sec=r["end"]) for r in results),
        key=lambda w: w.start_sec,
    )


async def align(
    audio_float32: np.ndarray,
    transcript: str,
    language: str,
    *,
    _settings: Settings | None = None,
) -> list[Word]:
    s = _settings if _settings is not None else Settings()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor,
        _align_sync,
        audio_float32,
        transcript,
        language,
        s.ALIGNER_MODEL_PATH,
        s.ALIGNER_BATCH_SIZE,
        s.ALIGNER_WINDOW_SEC,
    )
