"""Tests for reading where the singing starts, from synced lyrics and from audio analysis."""

from __future__ import annotations

from typing import Any

import pytest

from music_assistant.models.audio_analysis import AudioAnalysisData
from music_assistant.providers.ai_radio.constants import POST_AUDIO_ONSET_LEAD
from music_assistant.providers.ai_radio.post_window import (
    audio_onset,
    choose_vocal_onset,
    is_sung,
    lyric_onset,
)

# 1800 bins over 180 s: one bin per 0.1 s
_DURATION = 180.0
_BINS = 1800


def _vocal_timeline(*runs: tuple[float, float, float]) -> AudioAnalysisData:
    """Build a stored analysis whose vocal activity is quiet except for the given runs."""
    probabilities = [0.02] * _BINS
    for start, end, level in runs:
        for index in range(round(start * 10), round(end * 10)):
            probabilities[index] = level
    return AudioAnalysisData(duration=_DURATION, extra_data={"vocal_activity": probabilities})


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


@pytest.mark.parametrize(
    ("offset_tag", "expected"),
    [
        pytest.param("[offset:+1500]", 8.0, id="positive offset: sung earlier than stamped"),
        pytest.param("[offset:-1500]", 11.0, id="negative offset: sung later than stamped"),
        pytest.param("[offset:0]", 9.5, id="zero offset"),
    ],
)
def test_an_offset_header_shifts_the_onset(offset_tag: str, expected: float) -> None:
    """The LRC offset applies to every timestamp, the first sung line's included."""
    lyrics = f"[ar:Someone]\n{offset_tag}\n[00:00.00]♪\n[00:09.50]First line"
    assert lyric_onset(lyrics) == pytest.approx(expected)


def test_an_offset_cannot_push_the_onset_before_the_start() -> None:
    """An offset larger than the first timestamp lands on the start, not before it."""
    assert lyric_onset("[offset:+5000]\n[00:02.00]First line") == 0.0


def test_an_untimed_line_cannot_set_the_onset() -> None:
    """A line without its own timestamp says nothing about when it is sung."""
    lyrics = "[00:04.00][Intro]\nan untimed line\n[00:11.00]Real line"
    assert lyric_onset(lyrics) == 11.0


def test_lyrics_with_only_labels_have_no_onset() -> None:
    """An instrumental with nothing but labels and filler gives no vocal entry."""
    assert lyric_onset("[00:00.00][Intro]\n[00:20.00](Instrumental)\n[00:30.00]♪") is None


# --- where the audio analysis first hears a voice ---


def test_audio_onset_is_where_the_first_voice_starts() -> None:
    """A confident, sustained run of vocal activity marks the entry."""
    assert audio_onset(_vocal_timeline((12.0, 40.0, 0.95))) == pytest.approx(12.0)


def test_a_weak_blip_in_the_intro_is_not_the_onset() -> None:
    """A short run that never reaches Smart Fades' confidence gate is not a voice."""
    analysis = _vocal_timeline((3.0, 3.5, 0.6), (12.0, 40.0, 0.95))
    assert audio_onset(analysis) == pytest.approx(12.0)


def test_a_timeline_without_a_voice_has_no_onset() -> None:
    """An instrumental timeline gives no vocal entry."""
    assert audio_onset(_vocal_timeline()) is None


@pytest.mark.parametrize(
    "analysis",
    [
        pytest.param(None, id="no analysis"),
        pytest.param(AudioAnalysisData(duration=_DURATION), id="no timeline"),
        pytest.param(
            AudioAnalysisData(duration=_DURATION, extra_data={"vocal_activity": [0.9] * 100}),
            id="wrong length",
        ),
        pytest.param(
            AudioAnalysisData(extra_data={"vocal_activity": [0.9] * _BINS}), id="no duration"
        ),
    ],
)
def test_audio_onset_needs_a_valid_timeline(analysis: Any) -> None:
    """Anything short of a validated timeline says nothing about the vocal entry."""
    assert audio_onset(analysis) is None


# --- choosing between the two ---


@pytest.mark.parametrize(
    ("lyrics", "audio", "expected"),
    [
        pytest.param(None, None, None, id="neither"),
        pytest.param(7.0, None, 7.0, id="lyrics only"),
        pytest.param(None, 12.0, 12.0, id="audio only"),
        pytest.param(8.0, 20.0, 20.0, id="audio well after the lyrics"),
        pytest.param(8.0, 8.0 + POST_AUDIO_ONSET_LEAD, 8.0, id="audio exactly the lead after"),
        pytest.param(8.0, 11.0, 8.0, id="audio a little after"),
        pytest.param(8.0, 3.0, 8.0, id="audio before the lyrics"),
    ],
)
def test_the_lyrics_lead_unless_the_audio_hears_the_voice_well_after(
    lyrics: float | None, audio: float | None, expected: float | None
) -> None:
    """The audio onset replaces the lyric one only when it is more than the lead later."""
    assert choose_vocal_onset(lyrics, audio) == expected
