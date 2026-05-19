import pycountry
import pytest

from whisper_proxy.lang import name_to_alpha2, normalize_name


def test_canonical_english() -> None:
    assert name_to_alpha2("English") == "en"


def test_case_insensitive() -> None:
    assert name_to_alpha2("english") == "en"
    assert name_to_alpha2("ENGLISH") == "en"
    assert name_to_alpha2("EnGlIsH") == "en"


def test_whitespace_tolerant() -> None:
    assert name_to_alpha2("  English  ") == "en"
    assert name_to_alpha2("\tEnglish\n") == "en"


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("Spanish", "es"),
        ("French", "fr"),
        ("German", "de"),
        ("Japanese", "ja"),
        ("Mandarin", "zh"),
        ("Cantonese", "zh"),
        ("Portuguese", "pt"),
        ("Russian", "ru"),
        ("Arabic", "ar"),
        ("Korean", "ko"),
    ],
)
def test_spec_languages(name: str, code: str) -> None:
    assert name_to_alpha2(name) == code


def test_unknown_returns_und() -> None:
    assert name_to_alpha2("Klingon") == "und"


def test_empty_returns_und() -> None:
    assert name_to_alpha2("") == "und"


def test_empty_does_not_raise() -> None:
    name_to_alpha2("")


def test_whitespace_only_returns_und() -> None:
    assert name_to_alpha2("   ") == "und"


def test_normalize_lowercases() -> None:
    assert normalize_name("English") == "english"


def test_normalize_trims_and_lowercases() -> None:
    assert normalize_name("  Mandarin Chinese ") == "mandarin chinese"


def test_normalize_preserves_internal_spaces() -> None:
    assert normalize_name("Modern Greek") == "modern greek"


def test_full_alpha2_set_covered() -> None:
    """Criterion 5: every ISO 639-1 alpha-2 code is reachable via its canonical name."""
    langs_with_alpha2 = [lang for lang in pycountry.languages if getattr(lang, "alpha_2", None)]
    assert len(langs_with_alpha2) == 184, (
        f"expected 184 alpha-2 languages in pycountry, got {len(langs_with_alpha2)}"
    )
    for lang in langs_with_alpha2:
        assert name_to_alpha2(lang.name) == lang.alpha_2.lower(), lang.name
