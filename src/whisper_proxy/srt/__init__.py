"""SRT assembly package — public API."""

from .._types import Word
from .segment import Cue, SegmentPolicy, words_to_cues
from .writer import cues_to_srt

__all__ = ["Cue", "SegmentPolicy", "cues_to_srt", "words_to_cues", "words_to_srt"]


def words_to_srt(words: list[Word], policy: SegmentPolicy) -> str:
    """Convert a word-level alignment to canonical SRT text.

    Returns ``""`` for empty input; otherwise a trailing blank line is guaranteed.
    """
    return cues_to_srt(words_to_cues(words, policy))
