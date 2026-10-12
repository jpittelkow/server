"""
Vocal-onset detection for posts: the second the singing starts.

Two sources can tell: synced lyrics, and the vocal-activity timeline in the stored audio
analysis. Lyric parsing is left to :func:`normalize_lrc_lyrics`, which strips ID tags and word timings,
expands multi-timestamp lines and sorts the result, so only the leading timestamp of a
normalised line is read here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from music_assistant.controllers.streams.smart_fades.vocal import (
    DEFAULT_VOCAL_CONFIG,
    build_vocal_windows,
    first_vocal_onset,
    parse_vocal_probabilities,
)
from music_assistant.helpers.lyrics import normalize_lrc_lyrics

from .constants import POST_ANALYSIS_ONSET_LEAD, POST_ANALYSIS_VOICE_COVERAGE

if TYPE_CHECKING:
    from music_assistant.models.audio_analysis import AudioAnalysisData

# the leading [mm:ss.xx] of a line that normalize_lrc_lyrics has already produced
_LEADING_TIMESTAMP_RE = re.compile(r"^\[(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)\]\s*(.*)$")

# A line that is only one of these, bracketed or not, is a section label rather than
# singing. Kept short on purpose: a bracketed line NOT on the list ("(ooh ooh)", "(yeah!)")
# counts as sung, because mistaking a vocal for a label puts the host on top of the singer.
_SECTION_MARKERS = frozenset(
    {
        "intro",
        "introduction",
        "verse",
        "pre-chorus",
        "prechorus",
        "pre chorus",
        "chorus",
        "refrain",
        "bridge",
        "outro",
        "instrumental",
        "instrumental break",
        "solo",
        "guitar solo",
        "drum solo",
        "interlude",
        "hook",
        "break",
        "breakdown",
        "music",
        "musical interlude",
        "silence",
        "no vocals",
    }
)

# a marker may carry an index - "[Verse 2]", "(Chorus I)"
_MARKER_INDEX_RE = re.compile(r"\s*[0-9ivxIVX]+\s*$")

# symbol-only filler used for instrumental stretches
_SYMBOL_ONLY_RE = re.compile(r"^[\s♩-♯*~\-_.·•…]+$")


def is_sung(text: str) -> bool:
    """
    Return whether a lyric line represents someone actually singing.

    False for blank lines, symbol-only filler and bracketed structural markers.
    True for everything else, including bracketed backing vocals.

    :param text: The line's text, with its timestamp already removed.
    """
    stripped = text.strip()
    if not stripped or _SYMBOL_ONLY_RE.match(stripped):
        return False
    inner = stripped
    if (inner.startswith("[") and inner.endswith("]")) or (
        inner.startswith("(") and inner.endswith(")")
    ):
        inner = inner[1:-1]
    candidate = _MARKER_INDEX_RE.sub("", inner.strip().lower()).strip(" :-")
    return candidate not in _SECTION_MARKERS


def lyric_onset(lrc_lyrics: str | None) -> float | None:
    """
    Return the second at which singing starts, or None when it cannot be told.

    :param lrc_lyrics: Synced lyrics in LRC format, may be None or empty.
    """
    if not (normalized := normalize_lrc_lyrics(lrc_lyrics)):
        return None
    # the earliest timestamp is often a zeroed header, a note marker or a section label
    for line in normalized.splitlines():
        if not (match := _LEADING_TIMESTAMP_RE.match(line.strip())):
            continue
        if not is_sung(match.group(3)):
            continue
        return int(match.group(1)) * 60 + float(match.group(2).replace(":", "."))
    return None


def voice_coverage(analysis: AudioAnalysisData, start: float, end: float) -> float:
    """
    Return the share of a stretch the analysis hears as voice, from 0 to 1.

    Reads the vocal windows with Smart Fades' default confidence gates, so a weak blip
    does not count. A missing or malformed timeline hears nothing.

    :param analysis: The track's stored audio analysis.
    :param start: Start of the stretch, in seconds.
    :param end: End of the stretch, in seconds.
    """
    if end <= start or (timeline := parse_vocal_probabilities(analysis)) is None:
        return 0.0
    assert analysis.duration is not None  # guaranteed by the validated timeline
    mask = build_vocal_windows(
        timeline.probabilities,
        timeline.frame_duration,
        0.0,
        analysis.duration,
        beat_duration=60.0 / analysis.bpm if analysis.bpm and analysis.bpm > 0.0 else None,
        config=DEFAULT_VOCAL_CONFIG,
    )
    heard = sum(max(0.0, min(right, end) - max(left, start)) for left, right in mask.windows)
    return heard / (end - start)


def choose_vocal_onset(lyrics: float | None, analysis: AudioAnalysisData | None) -> float | None:
    """
    Return the second the singing starts, from whichever sources can tell.

    The lyric onset leads. The analysed onset replaces it when it is more than
    POST_ANALYSIS_ONSET_LEAD seconds later, or when it is earlier and the analysis hears
    voice through almost all of the stretch up to the lyric line. With one source, that
    one is used.

    :param lyrics: Onset read from synced lyrics, or None.
    :param analysis: The track's stored audio analysis, or None.
    """
    heard = first_vocal_onset(analysis) if analysis is not None else None
    if lyrics is None or heard is None:
        return lyrics if heard is None else heard
    if heard - lyrics > POST_ANALYSIS_ONSET_LEAD:
        return heard
    # an earlier analysed onset is usually a count-in, a chant or crowd noise that stops
    # before the singing; unbroken voice up to the lyric line means the lyrics are late
    assert analysis is not None  # heard is only set from an analysis
    if heard < lyrics and voice_coverage(analysis, heard, lyrics) >= POST_ANALYSIS_VOICE_COVERAGE:
        return heard
    return lyrics
