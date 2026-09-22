"""Tests for splitting an AI Radio break into a post, from planning it to airing it."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientError
from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.errors import ProviderUnavailableError
from music_assistant_models.media_items import AudioFormat, ProviderMapping, SoundEffect, Track
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.streams.constants import (
    ATTR_POST_CLIP_ID,
    ATTR_POST_CLIP_OFFSET,
    ATTR_POST_END,
    ATTR_POST_GAIN_DB,
    ATTR_POST_START,
    ATTR_POST_URL,
    POST_ATTRS,
)
from music_assistant.providers.ai_radio.constants import (
    ATTR_ALLOW_POST,
    POST_CLIP_MAX_AGE,
    POST_CLIP_PREFIX,
    POST_TAIL_GAP,
)
from music_assistant.providers.ai_radio.rendering import AIRadioRenderMixin, _ClipAudio, _PostPlan

_QUEUE_ID = "player_a"
_CLIP_ID = "sess_1"
_MEDIA_PATH = "http://ha.invalid/api/tts_proxy/1.mp3"
_MEDIA = cast("Any", SimpleNamespace(path=_MEDIA_PATH))
_CLIP_FORMAT = AudioFormat(content_type=ContentType.MP3)
_BREAK_SECONDS = 20.0
_VOCAL_ONSET = 12.0
_OVERLAP = _VOCAL_ONSET - POST_TAIL_GAP
_HEAD = _BREAK_SECONDS - _OVERLAP
_RENDERING = "music_assistant.providers.ai_radio.rendering"


class PostRenderer(AIRadioRenderMixin):
    """Minimal harness exposing the post planning, with its lookups stubbed and counted."""

    domain = "ai_radio"
    instance_id = "ai_radio--test"

    def __init__(self, staged: Path, order: list[QueueItem]) -> None:
        """Initialize the harness around one queue played in the given order."""
        self.logger = logging.getLogger("tests.ai_radio.post")
        self.staged = staged
        self.order = order
        self.onset: float | None = _VOCAL_ONSET
        self.onset_lookups = 0
        self.stagings = 0
        cast("Any", self).mass = SimpleNamespace(
            player_queues=SimpleNamespace(get_next_item=self._next_item, get_item=self._item)
        )

    def _item(self, queue_id: str, item_id: str) -> QueueItem | None:
        assert queue_id == _QUEUE_ID
        return next((item for item in self.order if item.queue_item_id == item_id), None)

    def _next_item(self, queue_id: str, item_id: str) -> QueueItem | None:
        assert queue_id == _QUEUE_ID
        ids = [item.queue_item_id for item in self.order]
        index = ids.index(item_id) + 1
        return self.order[index] if index < len(self.order) else None

    async def _resolve_vocal_onset(self, queue_item: QueueItem) -> tuple[float | None, str]:
        self.onset_lookups += 1
        return self.onset, "" if self.onset is not None else "no lyrics found"

    async def _stage_post_clip(self, path: str) -> str | None:
        self.stagings += 1
        self.staged.write_bytes(b"voice")
        return str(self.staged)

    async def _precise_duration(self, path: str) -> float | None:
        return _BREAK_SECONDS


class BareRenderer(AIRadioRenderMixin):
    """Harness for the post helpers that reach outside the provider."""

    domain = "ai_radio"
    instance_id = "ai_radio--test"

    def __init__(self, **mass: Any) -> None:
        """Initialize the harness with the given stand-ins on mass."""
        self.logger = logging.getLogger("tests.ai_radio.post")
        cast("Any", self).mass = SimpleNamespace(**mass)


def _break_item(*, allow_post: bool = True) -> QueueItem:
    media_item = SoundEffect(
        item_id=_CLIP_ID,
        provider="ai_radio--test",
        name="Back announce",
        provider_mappings={
            ProviderMapping(
                item_id=_CLIP_ID, provider_domain="ai_radio", provider_instance="ai_radio--test"
            )
        },
    )
    return QueueItem(
        queue_id=_QUEUE_ID,
        queue_item_id="qi_break",
        name="Back announce",
        duration=None,
        media_item=media_item,
        extra_attributes={ATTR_ALLOW_POST: allow_post},
    )


def _track_item(name: str) -> QueueItem:
    media_item = Track(
        item_id=name,
        provider="library",
        name=name,
        provider_mappings={
            ProviderMapping(item_id=name, provider_domain="filesystem", provider_instance="fs")
        },
    )
    return QueueItem(
        queue_id=_QUEUE_ID,
        queue_item_id=f"qi_{name}",
        name=name,
        duration=200,
        media_item=media_item,
        extra_attributes={"playback_speed": 1.0},
    )


def _post_attributes(queue_item: QueueItem) -> dict[str, Any]:
    return {key: value for key, value in queue_item.extra_attributes.items() if key in POST_ATTRS}


@pytest.fixture
def staged(tmp_path: Path) -> Path:
    """Return the path the harness stages its clip at."""
    return tmp_path / "ma_ai_radio_post_staged.mp3"


# --- planning the split ---


async def test_post_is_armed_with_the_break_it_is_the_tail_of(staged: Path) -> None:
    """The record carries everything the streams side needs, including whose tail it is."""
    clip, track = _break_item(), _track_item("song")
    renderer = PostRenderer(staged, [clip, track])
    plan = await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=-2.0)
    assert plan is not None
    assert plan.head == pytest.approx(_HEAD)
    assert _post_attributes(track) == {
        ATTR_POST_URL: str(staged),
        ATTR_POST_CLIP_ID: "qi_break",
        ATTR_POST_CLIP_OFFSET: pytest.approx(_HEAD),
        ATTR_POST_START: 0.0,
        ATTR_POST_END: pytest.approx(_OVERLAP),
        ATTR_POST_GAIN_DB: -2.0,
    }


async def test_break_that_is_not_postable_is_left_alone(staged: Path) -> None:
    """Without the opt-in nothing is looked up and nothing is armed."""
    clip, track = _break_item(allow_post=False), _track_item("song")
    renderer = PostRenderer(staged, [clip, track])
    assert await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=0.0) is None
    assert renderer.onset_lookups == 0
    assert _post_attributes(track) == {}


async def test_repeat_request_gets_the_same_split(staged: Path) -> None:
    """A clip resolved more than once is planned, looked up and staged only once."""
    clip, track = _break_item(), _track_item("song")
    renderer = PostRenderer(staged, [clip, track])
    first = await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=0.0)
    second = await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=0.0)
    assert second is first
    assert renderer.onset_lookups == 1
    assert renderer.stagings == 1


async def test_break_that_cannot_post_stays_whole_on_a_repeat_request(staged: Path) -> None:
    """A clip handed out whole must not turn into a split one on a later request."""
    clip, track = _break_item(), _track_item("song")
    renderer = PostRenderer(staged, [clip, track])
    renderer.onset = None
    assert await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=0.0) is None
    renderer.onset = _VOCAL_ONSET
    assert await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=0.0) is None
    assert renderer.onset_lookups == 1
    assert _post_attributes(track) == {}


async def test_break_airing_again_arms_its_record_again(staged: Path) -> None:
    """The streams side takes an aired post off the record, so a replayed break re-arms it."""
    clip, track = _break_item(), _track_item("song")
    renderer = PostRenderer(staged, [clip, track])
    await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=-2.0)
    armed = _post_attributes(track)
    for key in POST_ATTRS:
        track.extra_attributes.pop(key, None)

    await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=-2.0)
    assert _post_attributes(track) == armed
    assert renderer.stagings == 1


async def test_plan_is_redone_when_another_record_follows_the_break(staged: Path) -> None:
    """The tail moves to the record that now follows, and comes off the one that did."""
    clip, first, second = _break_item(), _track_item("first"), _track_item("second")
    renderer = PostRenderer(staged, [clip, first, second])
    await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=0.0)
    assert _post_attributes(first)

    renderer.order = [clip, second, first]
    plan = await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=0.0)
    assert plan is not None
    assert plan.track_item_id == "qi_second"
    assert _post_attributes(second)[ATTR_POST_CLIP_ID] == "qi_break"
    assert _post_attributes(first) == {}
    assert first.extra_attributes == {"playback_speed": 1.0}


async def test_plan_is_redone_when_the_staged_clip_is_gone(staged: Path) -> None:
    """A pruned staged clip is fetched again rather than armed as a dead path."""
    clip, track = _break_item(), _track_item("song")
    renderer = PostRenderer(staged, [clip, track])
    await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=0.0)
    staged.unlink()

    plan = await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=0.0)
    assert plan is not None
    assert renderer.stagings == 2
    assert staged.is_file()
    assert _post_attributes(track)[ATTR_POST_URL] == str(staged)


async def test_redone_plan_leaves_a_post_armed_by_another_break(staged: Path) -> None:
    """Only the break that armed a record may take its post off again."""
    clip, first, second = _break_item(), _track_item("first"), _track_item("second")
    renderer = PostRenderer(staged, [clip, first, second])
    await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=0.0)
    first.extra_attributes[ATTR_POST_CLIP_ID] = "qi_other_break"

    renderer.order = [clip, second, first]
    await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=0.0)
    assert _post_attributes(first)[ATTR_POST_CLIP_ID] == "qi_other_break"


# --- settling the split when the break's audio is produced ---


def _break_streamdetails(plan: _PostPlan | None) -> StreamDetails:
    """Build the StreamDetails get_stream_details hands out for a break with this plan."""
    return StreamDetails(
        provider="ai_radio--test",
        item_id=_CLIP_ID,
        audio_format=_CLIP_FORMAT,
        media_type=MediaType.SOUND_EFFECT,
        stream_type=StreamType.CUSTOM,
        path=_MEDIA_PATH,
        data=_ClipAudio(_MEDIA_PATH, _CLIP_FORMAT, -2.0, plan),
    )


@pytest.fixture
def ffmpeg_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stand in for the ffmpeg run that produces a break's audio, recording each request."""
    calls: list[dict[str, Any]] = []

    async def _fake_ffmpeg_stream(**kwargs: Any) -> AsyncGenerator[bytes]:
        calls.append(kwargs)
        yield b"pcm"

    monkeypatch.setattr(f"{_RENDERING}.get_ffmpeg_stream", _fake_ffmpeg_stream)
    return calls


async def _produce(renderer: PostRenderer, plan: _PostPlan | None) -> None:
    """Produce the break's audio the way the streams side asks for it ahead of the airing."""
    chunks = [chunk async for chunk in renderer.get_audio_stream(_break_streamdetails(plan))]
    assert chunks == [b"pcm"]


def _is_cut(ffmpeg_call: dict[str, Any]) -> bool:
    return any(str(param).startswith("atrim=") for param in ffmpeg_call["filter_params"])


async def test_break_is_cut_where_its_record_comes_in(
    staged: Path, ffmpeg_calls: list[dict[str, Any]]
) -> None:
    """With its record still next, the break stops where the record takes over its voice."""
    clip, track = _break_item(), _track_item("song")
    renderer = PostRenderer(staged, [clip, track])
    plan = await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=-2.0)

    await _produce(renderer, plan)

    (call,) = ffmpeg_calls
    assert call["audio_input"] == str(staged)
    assert call["filter_params"][0] == f"atrim=end={_HEAD:.3f}"
    assert _post_attributes(track)[ATTR_POST_CLIP_ID] == "qi_break"


async def test_break_airs_whole_once_another_record_follows_it(
    staged: Path, ffmpeg_calls: list[dict[str, Any]]
) -> None:
    """A queue change after the plan leaves the break whole, and its tail off the old record."""
    clip, first, second = _break_item(), _track_item("first"), _track_item("second")
    renderer = PostRenderer(staged, [clip, first, second])
    plan = await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=-2.0)
    renderer.order = [clip, second, first]

    await _produce(renderer, plan)

    (call,) = ffmpeg_calls
    assert call["audio_input"] == _MEDIA_PATH
    assert not _is_cut(call)
    assert _post_attributes(first) == {}
    assert _post_attributes(second) == {}


async def test_break_airs_whole_once_its_record_left_the_queue(
    staged: Path, ffmpeg_calls: list[dict[str, Any]]
) -> None:
    """With nothing after it any more, the break has nowhere to carry its tail."""
    clip, track = _break_item(), _track_item("song")
    renderer = PostRenderer(staged, [clip, track])
    plan = await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=-2.0)
    renderer.order = [clip]

    await _produce(renderer, plan)

    (call,) = ffmpeg_calls
    assert call["audio_input"] == _MEDIA_PATH
    assert not _is_cut(call)


async def test_break_airs_whole_when_its_staged_audio_is_gone(
    staged: Path, ffmpeg_calls: list[dict[str, Any]]
) -> None:
    """Its tail could no longer be mixed in, so the break keeps it."""
    clip, track = _break_item(), _track_item("song")
    renderer = PostRenderer(staged, [clip, track])
    plan = await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=-2.0)
    staged.unlink()

    await _produce(renderer, plan)

    (call,) = ffmpeg_calls
    assert call["audio_input"] == _MEDIA_PATH
    assert not _is_cut(call)
    assert _post_attributes(track) == {}


async def test_break_that_airs_again_is_cut_and_arms_its_record_again(
    staged: Path, ffmpeg_calls: list[dict[str, Any]]
) -> None:
    """After its post aired and came off the record, a replayed break sets it up again."""
    clip, track = _break_item(), _track_item("song")
    renderer = PostRenderer(staged, [clip, track])
    plan = await renderer._plan_post(clip, _MEDIA, _CLIP_ID, gain_db=-2.0)
    armed = _post_attributes(track)
    for key in POST_ATTRS:
        track.extra_attributes.pop(key, None)

    await _produce(renderer, plan)

    assert _is_cut(ffmpeg_calls[0])
    assert _post_attributes(track) == armed


# --- staging the rendered break ---


def _http_session(
    status: int = 200, body: bytes = b"voice", error: Exception | None = None, delay: float = 0.0
) -> SimpleNamespace:
    """Build a stand-in for the shared HTTP session that records the URLs it is asked for."""
    requested: list[str] = []

    @asynccontextmanager
    async def get(url: str) -> AsyncGenerator[SimpleNamespace]:
        requested.append(url)
        await asyncio.sleep(delay)
        if error is not None:
            raise error
        yield SimpleNamespace(status=status, read=AsyncMock(return_value=body))

    return SimpleNamespace(get=get, requested=requested)


@pytest.fixture
def temp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the temp dir that staged clips go to at this test's own directory."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def _staged_clips(directory: Path) -> list[Path]:
    return sorted(directory.glob(f"{POST_CLIP_PREFIX}*"))


async def test_local_clip_is_used_where_it_is(temp_dir: Path) -> None:
    """A clip the engine already wrote locally needs neither a fetch nor a copy."""
    session = _http_session()
    renderer = BareRenderer(http_session=session)
    assert await renderer._stage_post_clip("/media/tts/clip.mp3") == "/media/tts/clip.mp3"
    assert session.requested == []
    assert _staged_clips(temp_dir) == []


async def test_remote_clip_is_fetched_once_into_a_local_copy(temp_dir: Path) -> None:
    """The URL is read once, when the post is armed, so playback never depends on it."""
    session = _http_session(body=b"the whole break")
    renderer = BareRenderer(http_session=session)

    staged = await renderer._stage_post_clip(_MEDIA_PATH)

    assert session.requested == [_MEDIA_PATH]
    assert staged is not None
    assert _staged_clips(temp_dir) == [Path(staged)]
    assert Path(staged).read_bytes() == b"the whole break"


@pytest.mark.parametrize(
    "session_kwargs",
    [
        pytest.param({"status": 500}, id="server error"),
        pytest.param({"error": ClientError("connection refused")}, id="connection error"),
        pytest.param({"body": b""}, id="empty body"),
    ],
)
async def test_clip_that_cannot_be_fetched_arms_nothing(
    temp_dir: Path, session_kwargs: dict[str, Any]
) -> None:
    """Without the audio in hand there is no post, and nothing is left behind."""
    renderer = BareRenderer(http_session=_http_session(**session_kwargs))
    assert await renderer._stage_post_clip(_MEDIA_PATH) is None
    assert _staged_clips(temp_dir) == []


async def test_wedged_fetch_gives_up_instead_of_holding_up_the_break(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fetch that never answers ends in no post, long before the break is due."""
    monkeypatch.setattr(f"{_RENDERING}.POST_CLIP_FETCH_TIMEOUT", 0.05)
    renderer = BareRenderer(http_session=_http_session(delay=60))
    async with asyncio.timeout(5):
        assert await renderer._stage_post_clip(_MEDIA_PATH) is None
    assert _staged_clips(temp_dir) == []


async def test_only_clips_left_by_posts_that_never_aired_are_pruned(temp_dir: Path) -> None:
    """Staged clips past their age go; fresh ones and other files in the temp dir stay."""
    prefix = POST_CLIP_PREFIX
    long_ago = time.time() - POST_CLIP_MAX_AGE - 60
    stale, fresh = temp_dir / f"{prefix}stale.mp3", temp_dir / f"{prefix}fresh.mp3"
    foreign = temp_dir / "someone_elses.mp3"
    for path in (stale, fresh, foreign):
        path.write_bytes(b"x")
    for path in (stale, foreign):
        os.utime(path, (long_ago, long_ago))

    BareRenderer()._prune_post_clips()

    assert not stale.exists()
    assert fresh.exists()
    assert foreign.exists()


# --- finding where the singing starts ---


def _lyrics_renderer(lookup: AsyncMock) -> BareRenderer:
    return BareRenderer(metadata=SimpleNamespace(get_track_lyrics=lookup))


def _track_with_lyrics(lrc_lyrics: str | None) -> QueueItem:
    track = _track_item("song")
    cast("Track", track.media_item).metadata.lrc_lyrics = lrc_lyrics
    return track


async def test_stored_synced_lyrics_are_used_without_a_lookup() -> None:
    """Lyrics already on the track cost nothing, so no provider is asked."""
    lookup = AsyncMock()
    renderer = _lyrics_renderer(lookup)
    track = _track_with_lyrics("[00:00.00]♪\n[00:09.50]First line")
    assert await renderer._resolve_vocal_onset(track) == (9.5, "")
    lookup.assert_not_awaited()


async def test_lyrics_are_looked_up_when_none_are_stored() -> None:
    """A track without stored lyrics gets them from Music Assistant's own lookup."""
    lookup = AsyncMock(return_value=(None, "[00:07.00]First line"))
    renderer = _lyrics_renderer(lookup)
    assert await renderer._resolve_vocal_onset(_track_with_lyrics(None)) == (7.0, "")
    lookup.assert_awaited_once()


@pytest.mark.parametrize(
    ("found", "reason"),
    [
        pytest.param(
            ("Plain words", None),
            "only unsynced lyrics available, so no vocal timing",
            id="plain lyrics only",
        ),
        pytest.param(
            (None, "[00:00.00][Intro]\n[00:20.00](Instrumental)"),
            "synced lyrics have no sung line",
            id="no sung line",
        ),
        pytest.param((None, None), "no lyrics found", id="nothing found"),
    ],
)
async def test_lyrics_without_vocal_timing_say_why(
    found: tuple[str | None, str | None], reason: str
) -> None:
    """Each way of coming up empty is logged with its own reason."""
    renderer = _lyrics_renderer(AsyncMock(return_value=found))
    assert await renderer._resolve_vocal_onset(_track_with_lyrics(None)) == (None, reason)


async def test_failing_lyrics_lookup_says_why() -> None:
    """A lyrics failure costs the post, never the clip."""
    renderer = _lyrics_renderer(AsyncMock(side_effect=ProviderUnavailableError("provider down")))
    assert await renderer._resolve_vocal_onset(_track_with_lyrics(None)) == (
        None,
        "lyrics lookup failed (provider down)",
    )


async def test_slow_lyrics_lookup_is_abandoned(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lookup that walks every metadata provider must not hold up a break about to air."""
    monkeypatch.setattr(f"{_RENDERING}.POST_LYRICS_TIMEOUT", 0.05)

    async def _slow_lookup(_track: Track) -> tuple[str | None, str | None]:
        await asyncio.sleep(60)
        return None, "[00:07.00]Too late"

    renderer = _lyrics_renderer(AsyncMock(side_effect=_slow_lookup))
    async with asyncio.timeout(5):
        onset, reason = await renderer._resolve_vocal_onset(_track_with_lyrics(None))
    assert onset is None
    assert reason.startswith("lyrics lookup took longer than")


async def test_item_that_is_not_a_track_has_no_vocal_onset() -> None:
    """Only a track has lyrics to read the vocal entry from."""
    renderer = _lyrics_renderer(AsyncMock())
    assert await renderer._resolve_vocal_onset(_break_item()) == (None, "no track details")
