"""Append captured 16 kHz mono audio to a session WAV (PCM16).

16 kHz mono PCM WAV is the de-facto STT baseline and is exactly what the mic
pipeline captures, so saving it needs no resampling and no extra dependencies
(stdlib ``wave``). The file is created lazily on the first chunk; the WAV header
is finalized on ``close()``.
"""

from __future__ import annotations

import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


class AudioWriter:
    def __init__(
        self, output_dir: str, sample_rate: int, enabled: bool = True, stamp: Optional[str] = None
    ):
        self._output_dir = output_dir
        self._sample_rate = sample_rate
        self._enabled = enabled
        self._stamp = stamp
        self._wav: Optional[wave.Wave_write] = None
        self._path: Optional[Path] = None

    def write(self, samples: np.ndarray) -> None:
        if not self._enabled:
            return
        if self._wav is None:
            out_dir = Path(self._output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = self._stamp or datetime.now().strftime("%Y-%m-%d-%H%M%S")
            self._path = out_dir / f"{stamp}.wav"
            self._wav = wave.open(str(self._path), "wb")
            self._wav.setnchannels(1)
            self._wav.setsampwidth(2)  # PCM16
            self._wav.setframerate(self._sample_rate)
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        self._wav.writeframes(pcm16.tobytes())

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None

    @property
    def path(self) -> Optional[Path]:
        return self._path
