"""Language name → ISO 639-1 alpha-2 mapping.

Backed by ``pycountry`` for canonical ISO short-names, with a small
override table for names OpenArc emits that don't resolve to an alpha-2
language in pycountry (``"Mandarin"`` and ``"Cantonese"`` → ``"zh"``).
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


