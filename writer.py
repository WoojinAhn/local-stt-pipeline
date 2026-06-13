"""Append transcript lines to a session file as they are produced.

Each line is flushed immediately, so a crash or hard kill loses at most the
utterance currently being transcribed — everything already printed is on disk.
The file is created lazily on the first line, so a session with no speech leaves
no empty file behind.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


class TranscriptWriter:
    def __init__(self, output_dir: str, enabled: bool = True):
        self._output_dir = output_dir
        self._enabled = enabled
        self._file = None
        self._path: Optional[Path] = None

    def write_line(self, timestamp: str, text: str) -> None:
        if not self._enabled:
            return
        if self._file is None:
            out_dir = Path(self._output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            self._path = out_dir / f"{stamp}.txt"
            self._file = self._path.open("a", encoding="utf-8")
        self._file.write(f"[{timestamp}] {text}\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def path(self) -> Optional[Path]:
        return self._path
