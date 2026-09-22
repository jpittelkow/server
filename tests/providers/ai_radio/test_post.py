"""Tests for arming an AI Radio post on the record that follows a break."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from music_assistant_models.media_items import ProviderMapping, SoundEffect, Track
from music_assistant_models.queue_item import QueueItem

from music_assistant.providers.ai_radio.constants import (
    ATTR_ALLOW_POST,
    ATTR_POST_CLIP_ID,
    ATTR_POST_CLIP_OFFSET,
    ATTR_POST_END,
    ATTR_POST_GAIN_DB,
    ATTR_POST_START,
    ATTR_POST_URL,
    POST_ATTRS,
    POST_TAIL_GAP,
)
from music_assistant.providers.ai_radio.rendering import AIRadioRenderMixin

_QUEUE_ID = "player_a"
_CLIP_ID = "sess_1"
_MEDIA = cast("Any", SimpleNamespace(path="http://ha.invalid/api/tts_proxy/1.mp3"))
_BREAK_SECONDS = 20.0
_VOCAL_ONSET = 12.0
_OVERLAP = _VOCAL_ONSET - POST_TAIL_GAP
_HEAD = _BREAK_SECONDS - _OVERLAP


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
