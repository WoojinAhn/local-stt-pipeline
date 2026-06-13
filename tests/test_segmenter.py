import numpy as np
from mlx_audio.realtime_vad import TurnEvent, TurnEventKind
from segmenter import UtteranceSegmenter, VAD_SAMPLE_RATE


class FakeVad:
    """Scripted VAD: emits the events queued for each process() call index."""

    def __init__(self, script):
        self._script = script  # dict: call_index -> list[TurnEvent]
        self._i = 0

    def process(self, samples):
        events = self._script.get(self._i, [])
        self._i += 1
        return events


def _started():
    return TurnEvent(kind=TurnEventKind.SPEECH_STARTED, audio_ms=0)


def _stopped():
    return TurnEvent(kind=TurnEventKind.SPEECH_STOPPED, audio_ms=0)


def test_emits_one_utterance_between_start_and_stop():
    chunk = np.ones(1600, dtype=np.float32)  # 100 ms
    vad = FakeVad({1: [_started()], 5: [_stopped()]})
    seg = UtteranceSegmenter(vad, pre_roll_ms=0, min_utterance_ms=0)

    completed = []
    for _ in range(7):
        completed.extend(seg.feed(chunk))

    assert len(completed) == 1
    # chunks 1,2,3,4 collected before stop fires on chunk 5 -> ~4 chunks
    assert completed[0].shape[0] == 1600 * 4


def test_min_utterance_guard_drops_short_blip():
    chunk = np.ones(1600, dtype=np.float32)
    vad = FakeVad({0: [_started()], 1: [_stopped()]})
    seg = UtteranceSegmenter(vad, pre_roll_ms=0, min_utterance_ms=500)

    completed = []
    for _ in range(3):
        completed.extend(seg.feed(chunk))

    assert completed == []  # only ~1 chunk (100ms) < 500ms guard


def test_no_events_yields_nothing():
    seg = UtteranceSegmenter(FakeVad({}), pre_roll_ms=0, min_utterance_ms=0)
    assert seg.feed(np.ones(1600, dtype=np.float32)) == []


def test_pre_roll_prepends_audio_before_speech_start():
    # 16000 samples/sec; 1600-sample chunks = 100 ms each.
    chunk = np.ones(1600, dtype=np.float32)
    # pre_roll_ms=100 keeps the most recent 1600 samples as pre-roll.
    vad = FakeVad({2: [_started()], 4: [_stopped()]})
    seg = UtteranceSegmenter(vad, pre_roll_ms=100, min_utterance_ms=0)

    completed = []
    for _ in range(6):
        completed.extend(seg.feed(chunk))

    assert len(completed) == 1
    # chunks 0,1 build the tail; on chunk 2 STARTED seeds utterance with the
    # last 100ms (1 chunk) of pre-roll, then chunks 2,3 are appended before
    # STOPPED fires on chunk 4 -> 1 (pre-roll) + 2 = 3 chunks.
    assert completed[0].shape[0] == 1600 * 3
