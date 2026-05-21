"""Word-to-cue grouping for SRT assembly (spec §4.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from .._types import Word

# Sentence-final punctuation (spec criterion 9).
# Includes ellipsis (U+2026), CJK full-stop (U+3002), and full-width ! (U+FF01) / ? (U+FF1F).
_SENTENCE_FINAL: frozenset[str] = frozenset(".!?…。！？")  # noqa: RUF001


@dataclass(frozen=True)
class SegmentPolicy:
    max_chars: int = 84
    max_sec: float = 6.0
    min_sec: float = 1.0
    silence_ms: int = 700
    # Cues shorter than min_chars are merged with a neighbor when budget allows.
    # Set to 0 to disable merging.
    min_chars: int = 20
    # Don't merge across gaps larger than this (seconds).
    max_merge_gap_sec: float = 1.5


class Cue(NamedTuple):
    start_sec: float
    end_sec: float
    lines: tuple[str, ...]  # 1 or 2 elements


def _is_sentence_end(token: str) -> bool:
    return bool(token) and token[-1] in _SENTENCE_FINAL


def _split_text(text: str) -> tuple[str, ...]:
    """Wrap text into at most two display lines, splitting near the midpoint."""
    tokens = text.split()
    if len(tokens) == 1 or len(text) <= 42:
        return (text,)
    mid = len(text) // 2
    best_idx = 1
    best_dist = float("inf")
    pos = 0
    for i in range(len(tokens) - 1):
        pos += len(tokens[i])
        dist = abs(pos - mid)
        if dist < best_dist:
            best_dist = dist
            best_idx = i + 1
        pos += 1  # space
    return (" ".join(tokens[:best_idx]), " ".join(tokens[best_idx:]))


def _split_to_lines(words: list[Word]) -> tuple[str, ...]:
    """Wrap words into at most two lines, splitting near the text midpoint."""
    return _split_text(" ".join(w.token for w in words))


def _merge_short_cues(cues: list[Cue], policy: SegmentPolicy) -> list[Cue]:
    """Merge cues shorter than min_chars with a neighbor when budget allows.

    Prefers merging with the previous cue (natural continuation), falls back
    to the next. Repeats until no further merges are possible.
    """
    if policy.min_chars == 0 or len(cues) <= 1:
        return cues

    changed = True
    while changed:
        changed = False
        result: list[Cue] = []
        i = 0
        while i < len(cues):
            cue = cues[i]
            text = " ".join(cue.lines)

            if len(text) >= policy.min_chars:
                result.append(cue)
                i += 1
                continue

            # Try merging backward into the already-built result.
            if result:
                prev = result[-1]
                prev_text = " ".join(prev.lines)
                merged_text = prev_text + " " + text
                gap = cue.start_sec - prev.end_sec
                merged_dur = cue.end_sec - prev.start_sec
                if (
                    gap <= policy.max_merge_gap_sec
                    and len(merged_text) <= policy.max_chars
                    and merged_dur <= policy.max_sec
                ):
                    result[-1] = Cue(
                        start_sec=prev.start_sec,
                        end_sec=cue.end_sec,
                        lines=_split_text(merged_text),
                    )
                    i += 1
                    changed = True
                    continue

            # Fall back to merging forward with the next cue.
            if i + 1 < len(cues):
                nxt = cues[i + 1]
                nxt_text = " ".join(nxt.lines)
                merged_text = text + " " + nxt_text
                gap = nxt.start_sec - cue.end_sec
                merged_dur = nxt.end_sec - cue.start_sec
                if (
                    gap <= policy.max_merge_gap_sec
                    and len(merged_text) <= policy.max_chars
                    and merged_dur <= policy.max_sec
                ):
                    result.append(Cue(
                        start_sec=cue.start_sec,
                        end_sec=nxt.end_sec,
                        lines=_split_text(merged_text),
                    ))
                    i += 2
                    changed = True
                    continue

            result.append(cue)
            i += 1

        cues = result

    return cues


def words_to_cues(words: list[Word], policy: SegmentPolicy) -> list[Cue]:
    """Group words into Cues per the segmentation policy."""
    if not words:
        return []

    silence_sec = policy.silence_ms / 1000.0
    groups: list[list[Word]] = []
    current: list[Word] = []

    def flush() -> None:
        if current:
            groups.append(list(current))
            current.clear()

    for i, word in enumerate(words):
        next_word = words[i + 1] if i + 1 < len(words) else None

        # Force-break before this word if adding it would breach either hard limit,
        # but only when there is already content to flush (must always add ≥1 word).
        if current:
            joined = " ".join(w.token for w in current) + " " + word.token
            dur = word.end_sec - current[0].start_sec
            if len(joined) > policy.max_chars or dur > policy.max_sec:
                flush()

        current.append(word)

        if _is_sentence_end(word.token):
            flush()
        elif next_word is not None and (next_word.start_sec - word.end_sec) >= silence_sec:
            flush()

    flush()

    # Build Cue objects, extending end_sec to enforce min_sec (clamped to next cue start).
    cues: list[Cue] = []
    for j, group in enumerate(groups):
        start = group[0].start_sec
        end = group[-1].end_sec
        if end - start < policy.min_sec:
            next_start = groups[j + 1][0].start_sec if j + 1 < len(groups) else float("inf")
            end = min(start + policy.min_sec, next_start)
        cues.append(Cue(start_sec=start, end_sec=end, lines=_split_to_lines(group)))

    return _merge_short_cues(cues, policy)
