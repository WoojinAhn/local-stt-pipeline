"""Turn a stream of mic samples into completed utterance buffers via VAD."""

from __future__ import annotations

from typing import List

import numpy as np

from mlx_audio.realtime_vad import TurnEventKind

VAD_SAMPLE_RATE = 16000


class UtteranceSegmenter:
    """Drive utterance segmentation from a StreamingVad's turn events.

    ``vad`` must expose ``process(samples) -> list[TurnEvent]``. While in speech,
    incoming samples are accumulated; on SPEECH_STOPPED the utterance is emitted
    (unless shorter than ``min_utterance_ms``). A pre-roll keeps the start of
    speech from being clipped.
    """

    def __init__(self, vad, pre_roll_ms: int = 300, min_utterance_ms: int = 300):
        self._vad = vad
        self._pre_roll = int(pre_roll_ms * VAD_SAMPLE_RATE / 1000)
        self._min_samples = int(min_utterance_ms * VAD_SAMPLE_RATE / 1000)
        self._tail = np.zeros(0, dtype=np.float32)
        self._collecting = False
        self._utterance = np.zeros(0, dtype=np.float32)

    def feed(self, samples: np.ndarray) -> List[np.ndarray]:
        samples = samples.astype(np.float32)
        events = self._vad.process(samples)
        completed: List[np.ndarray] = []

        for event in events:
            if event.kind == TurnEventKind.SPEECH_STARTED and not self._collecting:
                self._collecting = True
                self._utterance = self._tail.copy()  # pre-roll captured before this chunk
            elif event.kind == TurnEventKind.SPEECH_STOPPED and self._collecting:
                self._collecting = False
                utterance = self._utterance
                self._utterance = np.zeros(0, dtype=np.float32)
                if utterance.shape[0] >= self._min_samples:
                    completed.append(utterance)

        if self._collecting:
            self._utterance = np.concatenate([self._utterance, samples])
        if self._pre_roll:
            self._tail = np.concatenate([self._tail, samples])[-self._pre_roll:]

        return completed
