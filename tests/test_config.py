from pathlib import Path

import pytest
from pydantic import ValidationError

from whisper_proxy.config import Settings


def make(**kwargs: object) -> Settings:
    """Instantiate Settings with env-file loading disabled and optional overrides."""
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def test_defaults() -> None:
    s = make()
    assert str(s.OPENARC_BASE_URL).rstrip("/") == "http://localhost:8000"
    assert s.OPENARC_MODEL == "qwen3-asr-0_6b-int8-asym"
    assert s.OPENARC_READ_TIMEOUT == 3000
    assert s.OPENARC_CONNECT_TIMEOUT == 5
    assert s.ALIGNER_MODEL == "mms-1b-all"
    assert s.ALIGNER_BATCH_SIZE == 4
    assert s.ALIGNER_WINDOW_SEC == 30
    assert s.LANG_DETECT_WINDOW_SEC == 10
    assert s.LANG_DETECT_MAX_ATTEMPTS == 6
    assert s.LANG_DETECT_SHIFT_SEC == 15
    assert s.LANG_DETECT_MIN_TEXT_CHARS == 20
    assert s.LANG_DETECT_HALLUCINATION_PATTERNS == ""
    assert s.LANG_DETECT_HEAD_SEC == 30
    assert s.CUE_MAX_CHARS == 84
    assert s.CUE_MAX_SEC == 6.0
    assert s.CUE_MIN_SEC == 1.0
    assert s.CUE_SILENCE_MS == 700
    assert s.MAX_AUDIO_BYTES == 200_000_000
    assert s.LOG_LEVEL == "INFO"
    assert s.LOG_FORMAT == "json"


def test_invalid_url() -> None:
    with pytest.raises(ValidationError) as exc_info:
        make(OPENARC_BASE_URL="not-a-url")
    assert "OPENARC_BASE_URL" in str(exc_info.value)


def test_read_timeout_too_large() -> None:
    with pytest.raises(ValidationError) as exc_info:
        make(OPENARC_READ_TIMEOUT=4000)
    assert "OPENARC_READ_TIMEOUT" in str(exc_info.value)


def test_read_timeout_exactly_3600_rejected() -> None:
    with pytest.raises(ValidationError):
        make(OPENARC_READ_TIMEOUT=3600)


def test_read_timeout_3599_accepted() -> None:
    s = make(OPENARC_READ_TIMEOUT=3599)
    assert s.OPENARC_READ_TIMEOUT == 3599


def test_max_audio_bytes_type_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_AUDIO_BYTES", "banana")
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)  # type: ignore[call-arg]
    assert "MAX_AUDIO_BYTES" in str(exc_info.value)


def test_log_format_invalid() -> None:
    with pytest.raises(ValidationError) as exc_info:
        make(LOG_FORMAT="xml")  # type: ignore[arg-type]
    assert "LOG_FORMAT" in str(exc_info.value)


def test_cue_min_exceeds_max() -> None:
    with pytest.raises(ValidationError) as exc_info:
        make(CUE_MAX_SEC=0.0, CUE_MIN_SEC=10.0)
    msg = str(exc_info.value)
    assert "CUE_MIN_SEC" in msg or "CUE_MAX_SEC" in msg


def test_cue_min_equal_max_rejected() -> None:
    with pytest.raises(ValidationError):
        make(CUE_MAX_SEC=2.0, CUE_MIN_SEC=2.0)


_REQUIRED_VARS = [
    "OPENARC_BASE_URL",
    "OPENARC_MODEL",
    "OPENARC_READ_TIMEOUT",
    "OPENARC_CONNECT_TIMEOUT",
    "ALIGNER_MODEL",
    "ALIGNER_BATCH_SIZE",
    "ALIGNER_WINDOW_SEC",
    "LANG_DETECT_WINDOW_SEC",
    "LANG_DETECT_MAX_ATTEMPTS",
    "LANG_DETECT_SHIFT_SEC",
    "LANG_DETECT_MIN_TEXT_CHARS",
    "LANG_DETECT_HALLUCINATION_PATTERNS",
    "LANG_DETECT_HEAD_SEC",
    "CUE_MAX_CHARS",
    "CUE_MAX_SEC",
    "CUE_MIN_SEC",
    "CUE_SILENCE_MS",
    "MAX_AUDIO_BYTES",
    "LOG_LEVEL",
    "LOG_FORMAT",
]


def test_example_env_exists() -> None:
    repo_root = Path(__file__).parent.parent
    assert (repo_root / "config.example.env").is_file()


def test_example_env_covers_all_vars() -> None:
    repo_root = Path(__file__).parent.parent
    content = (repo_root / "config.example.env").read_text()
    for var in _REQUIRED_VARS:
        assert var in content, f"config.example.env is missing {var}"


def test_example_env_vars_are_commented_out() -> None:
    repo_root = Path(__file__).parent.parent
    content = (repo_root / "config.example.env").read_text()
    for var in _REQUIRED_VARS:
        # Each var should appear only as a commented-out line, not as an active assignment
        active = f"\n{var}="
        assert active not in content, f"{var} is not commented out in config.example.env"
