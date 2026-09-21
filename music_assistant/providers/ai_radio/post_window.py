"""
Vocal-onset detection for posts.

A *post* is the DJ talking over a record's instrumental opening and stopping as
the vocal arrives. Placing one needs a single fact that audio alone does not
give cheaply: when does the singing start? :func:`lyric_onset` reads it from the
first genuinely sung line of synced lyrics. Everything else is arithmetic on a
measured recording, done where the clip is rendered.

This is a pure function over plain strings, testable without audio, a queue or
a network.

Parsing is deliberately delegated: :func:`normalize_lrc_lyrics` already strips
ID tags and word timings, expands multi-timestamp lines and sorts the result
chronologically, so the only thing read here is the leading timestamp of an
already-normalised line.
"""

from __future__ import annotations

import re

from music_assistant.helpers.lyrics import normalize_lrc_lyrics

# the leading [mm:ss.xx] of a line that normalize_lrc_lyrics has already produced
_LEADING_TIMESTAMP_RE = re.compile(r"^\[(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)\]\s*(.*)$")

# Structural markers. A line that is only one of these - with or without
# brackets - is scenery rather than singing and must not set the onset. The list
# is deliberately short: a bracketed line that is NOT on it (backing vocals like
# "(ooh ooh)", ad-libs like "(yeah!)") counts as sung, because mistaking a real
# vocal for scenery is what puts the host on top of the singer.
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

    Taking the earliest timestamp would not do: synced lyrics routinely open
    with a zeroed header, a musical-note marker or a section label, all of which
    sit well before the first sung word.

    :param lrc_lyrics: Synced lyrics in LRC format, may be None or empty.
    """
    if not (normalized := normalize_lrc_lyrics(lrc_lyrics)):
        return None
    for line in normalized.splitlines():
        if not (match := _LEADING_TIMESTAMP_RE.match(line.strip())):
            continue
        if not is_sung(match.group(3)):
            continue
        return int(match.group(1)) * 60 + float(match.group(2).replace(":", "."))
    return None
