"""Tests for mixing an AI Radio post into a track's stream in StreamsAudio."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from music_assistant_models.errors import AudioError
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.queue_item import QueueItem

from music_assistant.controllers.streams.audio import StreamsAudio

_PCM_FORMAT = AudioFormat(sample_rate=44100, bit_depth=16, channels=2)
_MUSIC_CHUNKS = [b"chunk1", b"chunk2"]
_BREAK_ID = "break_item"
_TRACK_ID = "track_item"
_MIXER = "music_assistant.controllers.streams.audio.get_ffmpeg_post_stream"


def _make_streams_audio(last_served: str | None) -> tuple[StreamsAudio, MagicMock]:
    """Build a StreamsAudio whose queue last put out audio for the given item."""
    mass = MagicMock()
    mass.player_queues.queue_data_or_none = MagicMock(
        return_value=SimpleNamespace(last_served_item_id=last_served)
    )
    return StreamsAudio(mass), mass.player_queues


@pytest.fixture
def clip_file(tmp_path: Path) -> str:
    """Return the path of a staged clip that exists on disk."""
    clip_path = tmp_path / "ma_ai_radio_post_clip.mp3"
    clip_path.write_bytes(b"voice")
    return str(clip_path)


def _armed_track(clip_url: str, **overrides: Any) -> QueueItem:
    """Build a track carrying a post the way the AI Radio provider arms it."""
    attributes: dict[str, Any] = {
        StreamsAudio.POST_URL_ATTR: clip_url,
        StreamsAudio.POST_CLIP_ID_ATTR: _BREAK_ID,
        StreamsAudio.POST_CLIP_OFFSET_ATTR: 7.5,
        StreamsAudio.POST_START_ATTR: 0.0,
        StreamsAudio.POST_END_ATTR: 11.6,
        StreamsAudio.POST_GAIN_ATTR: -2.0,
        "playback_speed": 1.0,
    }
    attributes.update(overrides)
    return QueueItem(
        queue_id="queue",
        queue_item_id=_TRACK_ID,
        name="Song",
        duration=200,
        extra_attributes=attributes,
    )


def _post_attributes(queue_item: QueueItem) -> dict[str, Any]:
    return {
        key: value
        for key, value in queue_item.extra_attributes.items()
        if key in StreamsAudio.POST_ATTRS
    }


async def _music_stream() -> AsyncGenerator[bytes]:
    for chunk in _MUSIC_CHUNKS:
        yield chunk


def _fake_mixer(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the ffmpeg mixer with one that tags every chunk, and return its arguments."""
    mixer_kwargs: dict[str, Any] = {}

    async def _mixer(**kwargs: Any) -> AsyncGenerator[bytes]:
        mixer_kwargs.update(kwargs)
        async for chunk in kwargs["audio_input"]:
            yield b"mixed:" + chunk

    monkeypatch.setattr(_MIXER, _mixer)
    return mixer_kwargs


def _failing_mixer(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _mixer(**kwargs: Any) -> AsyncGenerator[bytes]:
        if kwargs:
            raise AudioError("clip could not be opened")
        yield b""

    monkeypatch.setattr(_MIXER, _mixer)


async def _collect(stream: AsyncGenerator[bytes]) -> list[bytes]:
    return [chunk async for chunk in stream]


async def test_track_without_a_post_passes_through() -> None:
    """An ordinary track is streamed untouched and nothing about the queue is signalled."""
    audio, player_queues = _make_streams_audio(last_served=None)
    track = QueueItem(queue_id="queue", queue_item_id=_TRACK_ID, name="Song", duration=200)
    result = await _collect(audio.get_post_mixed_stream(track, _music_stream(), _PCM_FORMAT))
    assert result == _MUSIC_CHUNKS
    player_queues.signal_update.assert_not_called()


async def test_passthrough_closes_the_stream_it_wraps() -> None:
    """Closing the wrapper early closes the track's own stream along with it."""
    closed: list[bool] = []

    async def _music() -> AsyncGenerator[bytes]:
        try:
            for chunk in _MUSIC_CHUNKS:
                yield chunk
        finally:
            closed.append(True)

    audio, _ = _make_streams_audio(last_served=None)
    track = QueueItem(queue_id="queue", queue_item_id=_TRACK_ID, name="Song", duration=200)
    stream = audio.get_post_mixed_stream(track, _music(), _PCM_FORMAT)
    assert await anext(stream) == _MUSIC_CHUNKS[0]
    await stream.aclose()
    assert closed == [True]


async def test_post_airs_when_its_break_played_right_before(
    monkeypatch: pytest.MonkeyPatch, clip_file: str
) -> None:
    """A track served straight after the break that armed it gets the post mixed in."""
    mixer_kwargs = _fake_mixer(monkeypatch)
    audio, _ = _make_streams_audio(last_served=_BREAK_ID)
    track = _armed_track(clip_file)
    result = await _collect(audio.get_post_mixed_stream(track, _music_stream(), _PCM_FORMAT))
    assert result == [b"mixed:" + chunk for chunk in _MUSIC_CHUNKS]
    assert mixer_kwargs["clip_input"] == clip_file
    assert mixer_kwargs["clip_offset"] == 7.5
    assert mixer_kwargs["voice_start"] == 0.0
    assert mixer_kwargs["voice_end"] == 11.6
    assert mixer_kwargs["gain_db"] == -2.0


async def test_post_comes_off_the_track_once_it_has_aired(
    monkeypatch: pytest.MonkeyPatch, clip_file: str
) -> None:
    """A stream that ran to its end removes the post, persists that, and keeps the rest."""
    _fake_mixer(monkeypatch)
    audio, player_queues = _make_streams_audio(last_served=_BREAK_ID)
    track = _armed_track(clip_file)
    await _collect(audio.get_post_mixed_stream(track, _music_stream(), _PCM_FORMAT))
    assert _post_attributes(track) == {}
    assert track.extra_attributes == {"playback_speed": 1.0}
    player_queues.signal_update.assert_called_once_with("queue", items_changed=True)


async def test_post_survives_a_stream_that_is_cut_short(
    monkeypatch: pytest.MonkeyPatch, clip_file: str
) -> None:
    """A player that drops its first request still finds the post on the second one."""
    _fake_mixer(monkeypatch)
    audio, player_queues = _make_streams_audio(last_served=_BREAK_ID)
    track = _armed_track(clip_file)
    stream = audio.get_post_mixed_stream(track, _music_stream(), _PCM_FORMAT)
    assert await anext(stream) == b"mixed:" + _MUSIC_CHUNKS[0]
    await stream.aclose()
    assert StreamsAudio.POST_URL_ATTR in track.extra_attributes

    # by now the track itself is the item whose audio last went out
    player_queues.queue_data_or_none.return_value = SimpleNamespace(last_served_item_id=_TRACK_ID)
    result = await _collect(audio.get_post_mixed_stream(track, _music_stream(), _PCM_FORMAT))
    assert result == [b"mixed:" + chunk for chunk in _MUSIC_CHUNKS]


@pytest.mark.parametrize(
    "last_served",
    [
        pytest.param(None, id="explicit play, skipped break or restored queue"),
        pytest.param("another_item", id="something came between the break and the track"),
    ],
)
async def test_post_is_dropped_when_its_break_did_not_play_right_before(
    monkeypatch: pytest.MonkeyPatch, clip_file: str, last_served: str | None
) -> None:
    """A post never airs on a playback its break did not lead into, and comes off the track."""
    mixer_kwargs = _fake_mixer(monkeypatch)
    audio, player_queues = _make_streams_audio(last_served=last_served)
    track = _armed_track(clip_file)
    result = await _collect(audio.get_post_mixed_stream(track, _music_stream(), _PCM_FORMAT))
    assert result == _MUSIC_CHUNKS
    assert mixer_kwargs == {}
    assert _post_attributes(track) == {}
    player_queues.signal_update.assert_called_once_with("queue", items_changed=True)


async def test_post_without_a_break_id_is_dropped(
    monkeypatch: pytest.MonkeyPatch, clip_file: str
) -> None:
    """A post that does not say which break it belongs to cannot be vouched for."""
    mixer_kwargs = _fake_mixer(monkeypatch)
    audio, _ = _make_streams_audio(last_served=_TRACK_ID)
    track = _armed_track(clip_file)
    del track.extra_attributes[StreamsAudio.POST_CLIP_ID_ATTR]
    result = await _collect(audio.get_post_mixed_stream(track, _music_stream(), _PCM_FORMAT))
    assert result == _MUSIC_CHUNKS
    assert mixer_kwargs == {}
    assert _post_attributes(track) == {}


async def test_post_is_dropped_on_a_seeked_track(
    monkeypatch: pytest.MonkeyPatch, clip_file: str
) -> None:
    """Every offset is measured from the start of the track, so a seek plays it clean."""
    mixer_kwargs = _fake_mixer(monkeypatch)
    audio, _ = _make_streams_audio(last_served=_BREAK_ID)
    track = _armed_track(clip_file)
    track.streamdetails = cast("Any", SimpleNamespace(seek_position=30))
    result = await _collect(audio.get_post_mixed_stream(track, _music_stream(), _PCM_FORMAT))
    assert result == _MUSIC_CHUNKS
    assert mixer_kwargs == {}
    assert _post_attributes(track) == {}


async def test_post_is_dropped_when_its_clip_is_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A staged clip that no longer exists must not reach the mixer, which it would kill."""
    mixer_kwargs = _fake_mixer(monkeypatch)
    audio, _ = _make_streams_audio(last_served=_BREAK_ID)
    track = _armed_track(str(tmp_path / "pruned.mp3"))
    result = await _collect(audio.get_post_mixed_stream(track, _music_stream(), _PCM_FORMAT))
    assert result == _MUSIC_CHUNKS
    assert mixer_kwargs == {}
    assert _post_attributes(track) == {}


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({StreamsAudio.POST_END_ATTR: None}, id="no end"),
        pytest.param({StreamsAudio.POST_START_ATTR: "soon"}, id="unparsable start"),
        pytest.param({StreamsAudio.POST_GAIN_ATTR: "loud"}, id="unparsable gain"),
        pytest.param({StreamsAudio.POST_END_ATTR: 0.0}, id="empty window"),
        pytest.param({StreamsAudio.POST_START_ATTR: -1.0}, id="negative start"),
    ],
)
async def test_post_with_an_unusable_window_is_dropped(
    monkeypatch: pytest.MonkeyPatch, clip_file: str, overrides: dict[str, Any]
) -> None:
    """Malformed post values play the track clean instead of raising inside the stream."""
    mixer_kwargs = _fake_mixer(monkeypatch)
    audio, _ = _make_streams_audio(last_served=_BREAK_ID)
    track = _armed_track(clip_file, **overrides)
    result = await _collect(audio.get_post_mixed_stream(track, _music_stream(), _PCM_FORMAT))
    assert result == _MUSIC_CHUNKS
    assert mixer_kwargs == {}
    assert _post_attributes(track) == {}


async def test_mixer_failure_raises_and_takes_the_post_off(
    monkeypatch: pytest.MonkeyPatch, clip_file: str
) -> None:
    """The per-item route gets the failure to report, and the broken post is not kept."""
    _failing_mixer(monkeypatch)
    audio, _ = _make_streams_audio(last_served=_BREAK_ID)
    track = _armed_track(clip_file)
    with pytest.raises(AudioError):
        await _collect(audio.get_post_mixed_stream(track, _music_stream(), _PCM_FORMAT))
    assert _post_attributes(track) == {}


async def test_mixer_failure_ends_the_track_quietly_in_flow_mode(
    monkeypatch: pytest.MonkeyPatch, clip_file: str
) -> None:
    """A flow stream loses this one track to a broken post rather than the whole flow."""
    _failing_mixer(monkeypatch)
    audio, _ = _make_streams_audio(last_served=_BREAK_ID)
    track = _armed_track(clip_file)
    result = await _collect(
        audio.get_post_mixed_stream(track, _music_stream(), _PCM_FORMAT, raise_on_error=False)
    )
    assert result == []
    assert _post_attributes(track) == {}
