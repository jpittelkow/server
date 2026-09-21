"""Utility helpers for AI Radio."""

from __future__ import annotations

import datetime
import random
import re
from typing import Any

from music_assistant_models.errors import MusicAssistantError

from music_assistant.helpers.datetime import utc

from .constants import EMPTY_SECTION_ID
from .models import Slot


def utc_now_iso() -> str:
    """Return a UTC ISO timestamp."""
    return utc().isoformat()


def format_ai_radio_timestamp(moment: datetime.datetime) -> str:
    """Format a moment for the <timestamp> placeholder, spelling out the weekday and month."""
    # spelled out so the LLM never has to derive the weekday from a numeric date
    return moment.strftime("%A %d %B %Y, %H:%M %Z")


def slugify(value: str) -> str:
    """Create a slug from arbitrary text."""
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "station"


def is_empty_section(section_id: str) -> bool:
    """Return True when this section acts as a no-op marker."""
    return section_id.strip().upper() == EMPTY_SECTION_ID


def track_songinfo(track: dict[str, Any] | None) -> str:
    """Return a display string for a track dictionary."""
    if not track:
        return ""
    value = str(track.get("songinfo") or "").strip()
    if value:
        return value
    artist = str(track.get("artist") or "").strip()
    name = str(track.get("name") or "").strip()
    return f"{artist} - {name}".strip(" -")


# --- AI RADIO POST FORK ------------------------------------------------------
# Upstream builds the track placeholder as f"{artist} - {name}" and nothing
# more, so a section with web_search disabled has two strings to work from and
# can produce only invented specifics or empty atmosphere. Both failure modes
# were measured on 2026-09-20; see claude-docs/ma-ai-radio.md.
#
# Everything added below already sits on the Track object the caller is holding,
# so this costs no lookup. That matters specifically for posts: the clip has to
# be rendered and duration-probed before the host track's ffmpeg chain is built,
# so it cannot absorb a network round-trip. Enriching the placeholder buys back
# what a web search would have fetched, for free.
MAX_DESCRIBED_GENRES = 3

# A track's embedded album stub carries `year` but NOT `album_type`, so a
# compilation's reissue year is indistinguishable from an original release year
# without a lookup - and a lookup is the one thing this must not do. Measured on
# 2026-09-20: 13.3 % of sampled tracks sit on a compilation-looking album, where
# stating the year would have the DJ date the song wrongly (Sam Cooke, who died
# in 1964, sitting on "the 2012 album The Platinum Collection"). Where a name
# matches, the album is still named - only the year is withheld.
_COMPILATION_HINT_RE = re.compile(
    r"collection|compilation|greatest hits|best of|very best|anthology|essential"
    r"|platinum|anniversary|singles|\bhits\b|definitive|ultimate|retrospective"
    r"|box set|vol\.|volume",
    re.IGNORECASE,
)

# Real album names in this library include an email address and rip-format tags
# ("Ultimix 151 (Mp3)"). An album clause is only worth adding if the name would
# not embarrass the host when read aloud, so obvious junk is dropped entirely.
_IMPLAUSIBLE_ALBUM_RE = re.compile(
    r"@|https?://|www\.|\[(?:mp3|flac|wav|aac)\]|\((?:mp3|flac|wav|aac)\)"
    r"|\b\d{2,3}\s?kbps\b",
    re.IGNORECASE,
)


def _album_clause(album: Any) -> str:
    """
    Return the "from the ... album ..." clause for a track's album, or "".

    :param album: The album stub hanging off a Track, or None.
    """
    name = str(getattr(album, "name", "") or "").strip()
    if not name or _IMPLAUSIBLE_ALBUM_RE.search(name):
        return ""
    year = getattr(album, "year", None)
    if year and not _COMPILATION_HINT_RE.search(name):
        return f"from the {year} album {name}"
    return f"from the album {name}"


def describe_track(media_item: Any, artist: str = "", name: str = "") -> str:
    """
    Return the description of a track that a section prompt will see.

    Degrades to exactly the upstream ``artist - name`` string whenever the
    richer fields are absent, so a provider that supplies no album or metadata
    produces the old output rather than something malformed.

    :param media_item: The Track object, or None when the caller holds only strings.
    :param artist: Artist name the caller has already resolved, if any.
    :param name: Track name the caller has already resolved, if any.
    """
    title = str(name or getattr(media_item, "name", "") or "").strip()
    if not artist:
        artists = getattr(media_item, "artists", None) or ()
        if artists:
            artist = str(getattr(artists[0], "name", "") or "")
    base = f"{str(artist).strip()} - {title}".strip(" -")
    if media_item is None:
        return base

    # a version ("Live", "Remastered") changes what the listener is about to
    # hear, so it belongs next to the title rather than in the trailing clauses
    version = str(getattr(media_item, "version", "") or "").strip()
    if version and version.lower() not in title.lower():
        base = f"{base} ({version})"

    clauses: list[str] = []
    if album_clause := _album_clause(getattr(media_item, "album", None)):
        clauses.append(album_clause)

    metadata = getattr(media_item, "metadata", None)
    raw_genres = getattr(metadata, "genres", None) or ()
    # genres arrive as a set on some providers, so sort before truncating or the
    # same track describes itself differently between runs
    genres = [str(genre).strip() for genre in sorted(raw_genres) if str(genre).strip()]
    if genres:
        clauses.append("genre " + ", ".join(genres[:MAX_DESCRIBED_GENRES]))

    return f"{base} ({'; '.join(clauses)})" if clauses else base


def pick_weighted_choice(choices: list[dict[str, Any]], rng: random.Random) -> str:
    """Pick one ALTERNATIVE section using weighted randomness."""
    valid: list[tuple[str, float]] = []
    for choice in choices:
        section_id = str(choice.get("section", "")).strip()
        weight = float(choice.get("weight", 1))
        if section_id and weight > 0:
            valid.append((section_id, weight))
    if not valid:
        raise MusicAssistantError("ALTERNATIVE has no valid section choices")
    total = sum(weight for _, weight in valid)
    target = rng.random() * total
    cursor = 0.0
    for section_id, weight in valid:
        cursor += weight
        if target <= cursor:
            return section_id
    return valid[-1][0]


def build_slots(tracks: list[dict[str, Any]]) -> list[Slot]:
    """Build insertion slots from a source track list."""
    if not tracks:
        return []

    cumulative_minutes = [0.0]
    total = 0.0
    for track in tracks:
        duration = track.get("duration")
        seconds = float(duration) if isinstance(duration, (int, float)) and duration > 0 else 210.0
        total += seconds / 60.0
        cumulative_minutes.append(total)

    slots: list[Slot] = []
    slots.append(
        Slot(
            when="start_of_playlist",
            at_index=0,
            prev_index=None,
            next_index=0,
            very_next_index=1 if len(tracks) > 1 else None,
            minute_mark=0.0,
        )
    )
    for index in range(len(tracks) - 1):
        slots.append(
            Slot(
                when="between_songs",
                at_index=index + 1,
                prev_index=index,
                next_index=index + 1,
                very_next_index=index + 2 if index + 2 < len(tracks) else None,
                minute_mark=cumulative_minutes[index + 1],
            )
        )
    slots.append(
        Slot(
            when="end_of_playlist",
            at_index=len(tracks),
            prev_index=len(tracks) - 1,
            next_index=None,
            very_next_index=None,
            minute_mark=cumulative_minutes[-1],
        )
    )
    return slots


def soft_limit_text(text: str, max_chars: int, tolerance_ratio: float = 0.15) -> str:
    """Trim generated text softly near sentence boundaries."""
    if max_chars <= 0:
        return text.strip()
    slack = max(30, int(max_chars * tolerance_ratio))
    hard_limit = max_chars + slack
    cleaned = text.strip()
    if len(cleaned) <= hard_limit:
        return cleaned

    candidate = cleaned[:hard_limit].rstrip()
    sentence_ends = [match.end() for match in re.finditer(r"[.!?](?:\s|$)", candidate)]
    if sentence_ends:
        after_target = [index for index in sentence_ends if index >= max_chars]
        if after_target:
            return candidate[: after_target[0]].strip()
        return candidate[: sentence_ends[-1]].strip()

    last_space = candidate.rfind(" ")
    if last_space > 0:
        return candidate[:last_space].rstrip()
    return candidate


def coerce_float(value: Any, default: float) -> float:
    """Convert arbitrary value to float with a safe fallback."""
    try:
        return float(value)
    except TypeError, ValueError:
        return default


def coerce_int(value: Any, default: int) -> int:
    """Convert arbitrary value to int with a safe fallback."""
    try:
        return int(value)
    except TypeError, ValueError:
        return default
