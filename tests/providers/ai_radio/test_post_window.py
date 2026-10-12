"""Tests for reading where the singing starts from synced lyrics."""

from __future__ import annotations

import pytest

from music_assistant.models.audio_analysis import AudioAnalysisData
from music_assistant.providers.ai_radio.post_window import (
    choose_vocal_onset,
    is_sung,
    lyric_onset,
    voice_coverage,
)


@pytest.mark.parametrize(
    "line",
    [
        "Hello darkness, my old friend",
        "(ooh ooh)",
        "(yeah!)",
        "[Ooh, baby]",
        "(Chorus of angels)",
    ],
)
def test_lines_that_are_sung(line: str) -> None:
    """Anything that is not blank, filler or a bare section label counts as singing."""
    assert is_sung(line)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        "♪",
        "♪ ♪ ♪",
        "...",
        "* * *",
        "[Intro]",
        "Chorus",
        "(Instrumental)",
        "[Verse 2]",
        "(Chorus I)",
        "[Chorus x2]",
        "[Guitar Solo]",
        "[Pre-Chorus]",
        "Bridge:",
        "- instrumental -",
    ],
)
def test_lines_that_are_not_sung(line: str) -> None:
    """Blank lines, symbol-only filler and section labels are scenery, not singing."""
    assert not is_sung(line)


@pytest.mark.parametrize("lyrics", [None, ""])
def test_no_lyrics_have_no_onset(lyrics: str | None) -> None:
    """Without lyrics there is nothing to place the voice by."""
    assert lyric_onset(lyrics) is None


def test_unsynced_lyrics_have_no_onset() -> None:
    """Plain lyrics carry no timing, so they cannot place the voice either."""
    assert lyric_onset("First line\nSecond line") is None


def test_onset_is_the_first_sung_line_not_the_first_timestamp() -> None:
    """A zeroed header and section labels sit well before the first sung word."""
    lyrics = "[00:00.00] ♪\n[00:04.10][Intro]\n[00:12.40]First sung line\n[00:15.00]Second line"
    assert lyric_onset(lyrics) == pytest.approx(12.4)


def test_tags_ahead_of_the_lyrics_are_ignored() -> None:
    """Artist, title and length tags are not lines of the song."""
    lyrics = "[ar:Some Artist]\n[ti:Some Song]\n[length:03:30]\n[00:08.50]Hello"
    assert lyric_onset(lyrics) == 8.5


def test_a_line_sung_more_than_once_counts_from_its_first_time() -> None:
    """A chorus written once with several timestamps is sung first at its earliest one."""
    lyrics = "[00:40.00][00:10.00]Chorus line\n[00:20.00]Verse line"
    assert lyric_onset(lyrics) == 10.0


def test_word_timings_do_not_hide_the_line() -> None:
    """Enhanced LRC word timings are stripped before the line is read."""
    assert lyric_onset("[00:07.00]<00:07.00>Hey <00:07.40>there") == 7.0


@pytest.mark.parametrize(
    ("timestamp", "seconds"),
    [
        pytest.param("[01:02.50]", 62.5, id="past the first minute"),
        pytest.param("[00:09:25]", 9.25, id="colon before the fraction"),
    ],
)
def test_timestamp_forms(timestamp: str, seconds: float) -> None:
    """Minutes carry over into seconds, and either fraction separator is read."""
    assert lyric_onset(f"{timestamp}Hello") == pytest.approx(seconds)


def test_backing_vocals_count_as_singing() -> None:
    """Mistaking a real vocal for scenery is what would put the host on top of the singer."""
    assert lyric_onset("[00:03.00](ooh ooh)\n[00:10.00]Main vocal") == 3.0


def test_an_untimed_line_cannot_set_the_onset() -> None:
    """A line without its own timestamp says nothing about when it is sung."""
    lyrics = "[00:04.00][Intro]\nan untimed line\n[00:11.00]Real line"
    assert lyric_onset(lyrics) == 11.0


def test_lyrics_with_only_labels_have_no_onset() -> None:
    """An instrumental with nothing but labels and filler gives no vocal entry."""
    assert lyric_onset("[00:00.00][Intro]\n[00:20.00](Instrumental)\n[00:30.00]♪") is None


def _timeline(*runs: tuple[float, float]) -> AudioAnalysisData:
    """Build a 90 s analysis that hears a confident voice only during the given runs."""
    probabilities = [0.02] * 1800
    for start, end in runs:
        for index in range(round(start / 0.05), round(end / 0.05)):
            probabilities[index] = 0.95
    return AudioAnalysisData(duration=90.0, vocal_activity=probabilities)


def test_unbroken_voice_covers_the_whole_stretch() -> None:
    """A singer who carries on through a stretch is heard across all of it."""
    assert voice_coverage(_timeline((3.0, 90.0)), 3.0, 9.0) == pytest.approx(1.0, abs=0.05)


def test_a_count_in_covers_little_of_the_stretch() -> None:
    """A short burst that stops leaves most of the stretch without voice."""
    assert voice_coverage(_timeline((3.0, 4.0), (12.0, 90.0)), 3.0, 12.0) < 0.3


def test_an_analysis_without_a_timeline_hears_nothing() -> None:
    """A missing timeline says nothing about the stretch."""
    assert voice_coverage(AudioAnalysisData(duration=90.0), 3.0, 9.0) == 0.0


@pytest.mark.parametrize(
    ("lyrics", "runs", "expected"),
    [
        pytest.param(None, ((6.25, 90.0),), 6.0, id="analysis alone"),
        pytest.param(7.0, None, 7.0, id="lyrics alone"),
        pytest.param(None, None, None, id="neither"),
        pytest.param(12.0, ((3.25, 4.25), (12.0, 90.0)), 12.0, id="count-in, lyrics lead"),
        pytest.param(9.3, ((3.25, 90.0),), 3.0, id="unbroken voice, lyrics are late"),
        pytest.param(4.0, ((7.25, 90.0),), 4.0, id="analysis a little later"),
        pytest.param(2.0, ((8.25, 90.0),), 8.0, id="analysis clearly later"),
    ],
)
def test_choose_vocal_onset(
    lyrics: float | None, runs: tuple[tuple[float, float], ...] | None, expected: float | None
) -> None:
    """The lyric onset leads unless the analysis clearly hears the singer elsewhere."""
    analysis = _timeline(*runs) if runs is not None else None
    onset = choose_vocal_onset(lyrics, analysis)
    if expected is None:
        assert onset is None
    else:
        assert onset == pytest.approx(expected)
