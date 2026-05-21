"""Tests for the SRT segmenter and writer (spec 07)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pysrt
import pysubs2
import pytest

from whisper_proxy._types import Word
from whisper_proxy.srt import SegmentPolicy, words_to_srt
from whisper_proxy.srt.segment import words_to_cues

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "words"
DEFAULT_POLICY = SegmentPolicy()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_words(name: str) -> list[Word]:
    data = json.loads((FIXTURE_DIR / name).read_text())
    return [Word(token=d["token"], start_sec=d["start_sec"], end_sec=d["end_sec"]) for d in data]


def _parse_pysrt(text: str) -> pysrt.SubRipFile:
    return pysrt.from_string(text, error_handling=pysrt.ERROR_RAISE)


def _parse_pysubs2(text: str) -> pysubs2.SSAFile:
    return pysubs2.SSAFile.from_string(text)


def _displayable_chars(cue_text: str) -> int:
    return sum(len(line) for line in cue_text.splitlines())


# ---------------------------------------------------------------------------
# Criterion 13: empty input → empty string
# ---------------------------------------------------------------------------


def test_empty_words_returns_empty_string():
    assert words_to_srt([], DEFAULT_POLICY) == ""


# ---------------------------------------------------------------------------
# Criterion 14: purity
# ---------------------------------------------------------------------------


def test_pure_same_output_on_repeated_calls():
    words = _load_words("sentences.json")
    out1 = words_to_srt(words, DEFAULT_POLICY)
    out2 = words_to_srt(words, DEFAULT_POLICY)
    assert out1 == out2


# ---------------------------------------------------------------------------
# Fixture round-trip: criteria 1 and 2
# ---------------------------------------------------------------------------


@pytest.fixture(params=[p.name for p in sorted(FIXTURE_DIR.glob("*.json"))])
def fixture_srt(request: pytest.FixtureRequest) -> str:
    words = _load_words(request.param)
    return words_to_srt(words, DEFAULT_POLICY)


def test_pysrt_parses_fixture(fixture_srt: str) -> None:
    """Criterion 1: pysrt accepts every fixture output."""
    if not fixture_srt:
        pytest.skip("empty SRT")
    _parse_pysrt(fixture_srt)


def test_pysubs2_parses_fixture(fixture_srt: str) -> None:
    """Criterion 2: pysubs2 accepts every fixture output."""
    if not fixture_srt:
        pytest.skip("empty SRT")
    _parse_pysubs2(fixture_srt)


# ---------------------------------------------------------------------------
# Criterion 3: max_chars per cue
# ---------------------------------------------------------------------------


def test_no_cue_exceeds_max_chars():
    words = _load_words("long_sentence.json")
    policy = SegmentPolicy(max_chars=40)
    cues = words_to_cues(words, policy)
    for cue in cues:
        total = sum(len(line) for line in cue.lines)
        assert total <= policy.max_chars, f"cue text {total!r} chars exceeds max {policy.max_chars}"


# ---------------------------------------------------------------------------
# Criterion 4: max_sec per cue
# ---------------------------------------------------------------------------


def test_no_cue_exceeds_max_sec():
    words = _load_words("long_sentence.json")
    policy = SegmentPolicy(max_sec=3.0)
    cues = words_to_cues(words, policy)
    for cue in cues:
        dur = cue.end_sec - cue.start_sec
        assert dur <= policy.max_sec + 1e-9, f"cue duration {dur:.3f}s exceeds max {policy.max_sec}"


# ---------------------------------------------------------------------------
# Criterion 5: min_sec enforced (unless word span is shorter)
# ---------------------------------------------------------------------------


def test_min_sec_enforced():
    words = [
        Word(token="Hi.", start_sec=0.0, end_sec=0.2),  # span 0.2 s < min 1.0
        Word(token="Bye.", start_sec=2.0, end_sec=2.2),
    ]
    policy = SegmentPolicy(min_sec=1.0, min_chars=0)
    cues = words_to_cues(words, policy)
    assert cues[0].end_sec == pytest.approx(1.0), "min_sec not applied"


def test_min_sec_clamped_to_next_start():
    words = [
        Word(token="Hi.", start_sec=0.0, end_sec=0.1),
        Word(token="Bye.", start_sec=0.4, end_sec=0.6),
    ]
    policy = SegmentPolicy(min_sec=1.0, min_chars=0)
    cues = words_to_cues(words, policy)
    # min_sec would push end to 1.0 but next start is 0.4
    assert cues[0].end_sec == pytest.approx(0.4), "not clamped to next start"


# ---------------------------------------------------------------------------
# Criterion 6: sequential numbering starting at 1
# ---------------------------------------------------------------------------


def test_sequential_numbering():
    words = _load_words("sentences.json")
    text = words_to_srt(words, DEFAULT_POLICY)
    numbers = [int(m.group()) for m in re.finditer(r"^\d+$", text, re.MULTILINE)]
    assert numbers == list(range(1, len(numbers) + 1)), f"numbering wrong: {numbers}"
    assert numbers[0] == 1


# ---------------------------------------------------------------------------
# Criterion 7: comma separator, exactly 3 ms digits
# ---------------------------------------------------------------------------


def test_timecode_format():
    words = _load_words("simple.json")
    text = words_to_srt(words, DEFAULT_POLICY)
    timecodes = re.findall(r"\d{2}:\d{2}:\d{2},\d{3}", text)
    assert timecodes, "no timecodes found"
    # Verify no dot-separated timecodes exist
    assert not re.search(r"\d{2}:\d{2}:\d{2}\.\d+", text), "dot separator found"
    # Verify every timecode line has the arrow separator
    for line in text.splitlines():
        if " --> " in line:
            parts = line.split(" --> ")
            assert len(parts) == 2
            for tc in parts:
                assert re.fullmatch(r"\d{2}:\d{2}:\d{2},\d{3}", tc), f"bad timecode: {tc!r}"


# ---------------------------------------------------------------------------
# Criterion 8: no overlapping cues
# ---------------------------------------------------------------------------


def test_no_overlapping_cues():
    words = _load_words("long_sentence.json")
    policy = SegmentPolicy(max_chars=30, max_sec=2.0)
    cues = words_to_cues(words, policy)
    for i in range(len(cues) - 1):
        assert cues[i].end_sec <= cues[i + 1].start_sec + 1e-9, (
            f"cue {i} end {cues[i].end_sec:.3f} > cue {i + 1} start {cues[i + 1].start_sec:.3f}"
        )


# ---------------------------------------------------------------------------
# Criterion 9: sentence-final punctuation always ends the cue
# ---------------------------------------------------------------------------


def test_sentence_final_punctuation_ends_cue():
    words = [
        Word(token="Hello.", start_sec=0.0, end_sec=0.5),
        Word(token="World", start_sec=0.6, end_sec=0.9),
        Word(token="here.", start_sec=1.0, end_sec=1.4),
    ]
    # min_chars=0 disables the merge pass so we test raw segmentation.
    policy = SegmentPolicy(min_chars=0)
    cues = words_to_cues(words, policy)
    # "Hello." should be its own cue; "World here." its own
    assert len(cues) == 2
    assert "Hello." in cues[0].lines[0]
    assert "World" in cues[1].lines[0]


def test_unicode_sentence_final_punctuation():
    words = _load_words("unicode_punct.json")
    # min_chars=0 disables the merge pass so we test raw segmentation.
    policy = SegmentPolicy(min_chars=0)
    cues = words_to_cues(words, policy)
    # Each token ending with CJK sentence-final punct should end its cue
    sentence_ends = [w for w in words if w.token[-1] in "。！？"]  # noqa: RUF001
    assert len(cues) == len(sentence_ends)


# ---------------------------------------------------------------------------
# Criterion 10: silence-based split within long sentence
# ---------------------------------------------------------------------------


def test_silence_split_within_sentence():
    words = _load_words("silence_break.json")
    # min_chars=0 disables the merge pass so we test that silence creates the break.
    policy = SegmentPolicy(silence_ms=700, min_chars=0)
    cues = words_to_cues(words, policy)
    # Gap between word index 2 (end=1.4) and 3 (start=2.2) is 800ms > 700ms
    # So there should be at least 2 cues
    assert len(cues) >= 2


# ---------------------------------------------------------------------------
# Criterion 11: at most two lines per cue
# ---------------------------------------------------------------------------


def test_at_most_two_lines_per_cue():
    words = _load_words("two_line_wrap.json")
    cues = words_to_cues(words, DEFAULT_POLICY)
    for cue in cues:
        assert len(cue.lines) <= 2, f"cue has {len(cue.lines)} lines"


# ---------------------------------------------------------------------------
# Criterion 12: trailing blank line
# ---------------------------------------------------------------------------


def test_trailing_blank_line():
    words = _load_words("simple.json")
    text = words_to_srt(words, DEFAULT_POLICY)
    assert text.endswith("\n\n") or text.endswith("\n"), "no trailing newline"
    # Must end with at least one blank line (two consecutive newlines)
    assert text.endswith("\n\n"), "output does not end with trailing blank line"


# ---------------------------------------------------------------------------
# Short-cue merge pass (min_chars / max_merge_gap_sec)
# ---------------------------------------------------------------------------


def test_short_cue_merged_with_neighbor():
    # "it." is 3 chars < default min_chars=20; should merge into the previous cue.
    words = [
        Word(token="Come", start_sec=0.0, end_sec=0.3),
        Word(token="on,", start_sec=0.35, end_sec=0.6),
        Word(token="do", start_sec=0.65, end_sec=0.9),
        Word(token="it.", start_sec=2.0, end_sec=2.5),  # separated by 1.1 s silence
    ]
    policy = SegmentPolicy(min_chars=20, max_merge_gap_sec=1.5)
    cues = words_to_cues(words, policy)
    # All four words should end up in one cue (gap 1.1 s < 1.5 s, text fits).
    assert len(cues) == 1
    full_text = " ".join(cues[0].lines)
    assert "Come on, do it." in full_text


def test_large_gap_prevents_merge():
    # Gap of 3 s exceeds max_merge_gap_sec=1.5, so the short cue is kept alone.
    words = [
        Word(token="Come", start_sec=0.0, end_sec=0.3),
        Word(token="on,", start_sec=0.35, end_sec=0.6),
        Word(token="do", start_sec=0.65, end_sec=0.9),
        Word(token="it.", start_sec=3.9, end_sec=4.4),  # 3 s gap
    ]
    policy = SegmentPolicy(min_chars=20, max_merge_gap_sec=1.5)
    cues = words_to_cues(words, policy)
    assert len(cues) == 2
