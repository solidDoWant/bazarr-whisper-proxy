from typing import Literal

from pydantic import AnyHttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # OpenArc backend
    OPENARC_BASE_URL: AnyHttpUrl = AnyHttpUrl("http://localhost:8000")
    OPENARC_MODEL: str = "qwen3-asr-0_6b-int8-asym"
    OPENARC_READ_TIMEOUT: int = 3000
    OPENARC_CONNECT_TIMEOUT: int = 5

    # Forced aligner
    ALIGNER_MODEL: str = "mms-1b-all"
    # Path to OpenVINO IR (.xml) or the upstream MMS ONNX (.onnx). When an
    # .onnx is supplied, the aligner converts it to IR on first use and
    # caches the result alongside it (dev convenience; production images
    # bake the .xml directly).
    ALIGNER_MODEL_PATH: str = "~/ctc_forced_aligner/model.xml"
    # OpenVINO device string. AUTO lets the runtime pick the best available
    # device; specify "GPU.1", "GPU.0", or "CPU" to pin. Multi-device (e.g.
    # "MULTI:GPU.1,GPU.0") rounds across devices for concurrent requests.
    ALIGNER_DEVICE: str = "AUTO"
    # Inference precision hint: f16 is the default on GPU, f32 on CPU.
    ALIGNER_PRECISION: Literal["f16", "f32"] = "f16"
    # Persists OpenVINO's compiled-blob cache so cold starts don't pay the
    # device-specific compile cost on every container restart. Empty disables.
    ALIGNER_CACHE_DIR: str = ""
    ALIGNER_BATCH_SIZE: int = 4
    ALIGNER_WINDOW_SEC: int = 30

    # Language detection — center-first shifting-window algorithm (task 15)
    LANG_DETECT_WINDOW_SEC: int = 10
    LANG_DETECT_MAX_ATTEMPTS: int = 6
    LANG_DETECT_SHIFT_SEC: int = 15
    LANG_DETECT_MIN_TEXT_CHARS: int = 20
    # comma-separated substrings stripped from transcription text before the length check
    LANG_DETECT_HALLUCINATION_PATTERNS: str = ""

    # no-op; superseded by LANG_DETECT_WINDOW_SEC (kept for backward compatibility)
    LANG_DETECT_HEAD_SEC: int = 30

    # SRT assembly
    CUE_MAX_CHARS: int = 84
    CUE_MAX_SEC: float = 6.0
    CUE_MIN_SEC: float = 1.0
    CUE_SILENCE_MS: int = 700
    CUE_MIN_CHARS: int = 20
    CUE_MAX_MERGE_GAP_SEC: float = 1.5

    # Audio
    MAX_AUDIO_BYTES: int = 200_000_000

    # Lingarr translation backend (Phase 2 — task=translate)
    # Unset → task=translate returns 422 (feature disabled).
    LINGARR_BASE_URL: AnyHttpUrl | None = None
    LINGARR_API_KEY: str = ""
    LINGARR_TIMEOUT: int = 600
    LINGARR_TARGET_LANGUAGE: str = "en"
    LINGARR_DEFAULT_MEDIA_TYPE: str = "Episode"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"

    @field_validator("OPENARC_READ_TIMEOUT")
    @classmethod
    def read_timeout_under_bazarr_limit(cls, v: int) -> int:
        if v < 3600:
            return v
        raise ValueError(f"OPENARC_READ_TIMEOUT must be < 3600 (Bazarr's read timeout); got {v}")

    @model_validator(mode="after")
    def cue_min_max_order(self) -> Settings:
        if self.CUE_MIN_SEC < self.CUE_MAX_SEC:
            return self
        raise ValueError(
            f"CUE_MIN_SEC ({self.CUE_MIN_SEC}) must be less than CUE_MAX_SEC ({self.CUE_MAX_SEC})"
        )
