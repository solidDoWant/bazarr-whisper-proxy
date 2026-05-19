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
    ALIGNER_MODEL_PATH: str = "~/ctc_forced_aligner/model.onnx"
    ALIGNER_BATCH_SIZE: int = 16
    ALIGNER_WINDOW_SEC: int = 30

    # Language detection
    LANG_DETECT_HEAD_SEC: int = 30

    # SRT assembly
    CUE_MAX_CHARS: int = 84
    CUE_MAX_SEC: float = 6.0
    CUE_MIN_SEC: float = 1.0
    CUE_SILENCE_MS: int = 700

    # Audio
    MAX_AUDIO_BYTES: int = 200_000_000

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
