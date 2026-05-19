"""SRT writer: Cue list → canonical SRT text (spec §4.5)."""

from __future__ import annotations

from .segment import Cue


def _timecode(sec: float) -> str:
    """Format seconds as HH:MM:SS,mmm (comma as millisecond separator)."""
    total_ms = round(sec * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cues_to_srt(cues: list[Cue]) -> str:
    """Emit canonical SRT text from a list of Cues, trailing blank line included."""
    if not cues:
        return ""

    parts: list[str] = []
    for idx, cue in enumerate(cues, 1):
        parts.append(str(idx))
        parts.append(f"{_timecode(cue.start_sec)} --> {_timecode(cue.end_sec)}")
        parts.extend(cue.lines)
        parts.append("")  # blank separator line

    return "\n".join(parts) + "\n"


def fallback_srt(text: str, duration_sec: float) -> str:
    """Single-cue SRT spanning the full duration, used when alignment fails."""
    cue = Cue(start_sec=0.0, end_sec=max(duration_sec, 0.001), lines=(text,))
    return cues_to_srt([cue])
