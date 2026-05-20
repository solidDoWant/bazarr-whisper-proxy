"""Language name → ISO 639-1 alpha-2 mapping.

Backed by ``pycountry`` for canonical ISO short-names, with a small
override table for names OpenArc emits that don't resolve to an alpha-2
language in pycountry (``"Mandarin"`` and ``"Cantonese"`` resolve to
alpha-3-only entries; we want ``"zh"``).
"""

from __future__ import annotations

import pycountry

UNDETERMINED = "und"

# Names OpenArc reports that don't resolve to an alpha-2 in pycountry.
# Add new entries here when an unmapped name shows up in the logs.
_OVERRIDES: dict[str, str] = {
    "mandarin": "zh",
    "mandarin chinese": "zh",
    "cantonese": "zh",
}


def normalize_name(name: str) -> str:
    """Lowercase and trim ``name`` for use as Bazarr's ``detected_language``."""
    return name.strip().lower()


def name_to_alpha2(name: str) -> str:
    """Return the ISO 639-1 alpha-2 code for ``name``, or ``"und"`` if unknown.

    Lookups are case-insensitive and whitespace-tolerant.
    """
    key = normalize_name(name)
    if not key:
        return UNDETERMINED

    if key in _OVERRIDES:
        return _OVERRIDES[key]

    try:
        lang = pycountry.languages.lookup(key)
    except LookupError:
        return UNDETERMINED

    code = getattr(lang, "alpha_2", None)
    return code.lower() if code else UNDETERMINED


# Qwen3-ASR (the OpenArc backend) validates the `language` param against this
# exact capitalised list and 5xx's on anything else — including alpha-2 codes
# like "en" (which it auto-capitalises to "En" before validating, so even the
# casing-only fix doesn't help). We translate the alpha-2 we got from Bazarr
# to the English capitalised name OpenArc expects.
_ALPHA2_TO_OPENARC: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "tl": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


def alpha2_to_openarc_language(code: str | None) -> str | None:
    """Translate a Bazarr-supplied alpha-2 to the capitalised name OpenArc accepts.

    Returns ``None`` for ``None`` input or for codes OpenArc doesn't support —
    in either case the caller should call OpenArc without the ``language`` param
    and let it auto-detect.
    """
    if not code:
        return None
    return _ALPHA2_TO_OPENARC.get(code.strip().lower())
