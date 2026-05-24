from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import os
import threading
from typing import Any

import numpy as np
import openvino as ov
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
_compiled_model: Any | None = None
_tokenizer: Any | None = None

# Two workers: one can be running OpenVINO inference while the other does the
# numpy-bound post-processing for the previous segment.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="aligner")


def _resolve_ir_path(model_path: str) -> str:
    """Accept either a .xml (OpenVINO IR) or .onnx path.

    For .onnx, convert on first use and cache the IR next to the source —
    keeps the dev workflow working when only the upstream ONNX is available.
    Production images bake the .xml directly so this branch is a no-op there.
    """
    expanded = os.path.expanduser(model_path)
    if expanded.endswith(".xml"):
        return expanded
    if expanded.endswith(".onnx"):
        xml_path = expanded[:-5] + ".xml"
        bin_path = xml_path[:-4] + ".bin"
        if not os.path.exists(xml_path):
            ensure_onnx_model(expanded, MODEL_URL)
            logger.info("converting ONNX → OpenVINO IR (one-time): %s", xml_path)
            # Serialize to .partial.{xml,bin} then atomic-rename so an
            # interrupted convert never leaves a partially-written .xml on
            # disk that future runs would happily try to load. OpenVINO
            # requires the on-disk path to end in .xml, so the temp suffix
            # is inserted before the extension rather than appended.
            tmp_xml = xml_path[:-4] + ".partial.xml"
            tmp_bin = bin_path[:-4] + ".partial.bin"
            model = ov.Core().read_model(expanded)
            ov.serialize(model, tmp_xml, tmp_bin)
            os.rename(tmp_bin, bin_path)
            os.rename(tmp_xml, xml_path)
        return xml_path
    raise AlignmentFailed(f"ALIGNER_MODEL_PATH must end in .xml or .onnx, got {model_path!r}")


def _ensure_model(
    model_path: str,
    device: str,
    precision: str,
    cache_dir: str,
) -> None:
    global _compiled_model, _tokenizer
    if _compiled_model is not None:
        return

    with _load_lock:
        if _compiled_model is not None:
            return

        xml_path = _resolve_ir_path(model_path)
        core = ov.Core()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            core.set_property({"CACHE_DIR": cache_dir})

        model = core.read_model(xml_path)
        # Dynamic batch + dynamic time: the streaming emission generator passes
        # variable-sized last batches and audio shapes vary per segment.
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": precision,
        }
        _compiled_model = core.compile_model(model, device, config=config)
        _tokenizer = Tokenizer()
        logger.info(
            "aligner model compiled device=%s precision=%s path=%s",
            device,
            precision,
            xml_path,
        )


def _to_iso3(lang: str) -> str:
    return _ISO2_TO_ISO3.get(lang.lower(), lang)


def _generate_emissions_streaming(
    compiled: Any,
    audio_waveform: np.ndarray,
    window_length: int,
    context_length: int,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    """Memory-efficient replacement for ctc_forced_aligner.generate_emissions.

    The library version pre-allocates all windows as a single numpy array
    (~0.5 GB for a 2-hour film) before running any inference. This version
    streams window slices directly into each batch, keeping only one batch
    of input in memory at a time.
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
        batch = np.stack(
            [padded[j * window : j * window + window + 2 * context] for j in batch_indices],
            axis=0,
        ).astype(np.float32)
        result = compiled({"input_values": batch})
        # OVDict — single output ("logits"); take it positionally to avoid
        # depending on the exported tensor name.
        emissions_list.append(np.asarray(next(iter(result.values()))))
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
    device: str,
    precision: str,
    cache_dir: str,
    batch_size: int,
    window_sec: int,
) -> list[Word]:
    if not transcript.strip():
        raise AlignmentFailed("transcript is empty")

    # Digital silence — return empty rather than feeding garbage to the model.
    if np.abs(audio).max() < 1e-6:
        return []

    _ensure_model(model_path, device, precision, cache_dir)

    iso3 = _to_iso3(language)

    try:
        emissions, stride = _generate_emissions_streaming(
            _compiled_model,
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

    return sorted(
        (Word(token=r["text"], start_sec=r["start"], end_sec=r["end"]) for r in results),
        key=lambda w: w.start_sec,
    )


def _align_segments_sync(
    audio: np.ndarray,
    segments: list[TranscriptionSegment],
    language: str,
    model_path: str,
    device: str,
    precision: str,
    cache_dir: str,
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
                device,
                precision,
                cache_dir,
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
        s.ALIGNER_DEVICE,
        s.ALIGNER_PRECISION,
        s.ALIGNER_CACHE_DIR,
        s.ALIGNER_BATCH_SIZE,
        s.ALIGNER_WINDOW_SEC,
    )
