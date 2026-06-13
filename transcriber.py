"""Load an STT model once and transcribe 16 kHz float32 buffers.

Uses ``model.generate`` directly rather than ``generate_transcription`` because
the latter writes a transcript file (``save_as_txt``) on every call. Handles both
mlx-audio model families used by the engine tiers: Whisper (needs a separately
loaded WhisperProcessor) and Qwen3-ASR (loads standalone).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx_audio.stt.utils import load_model

from engines import DEFAULT_ENGINE, EngineSpec, resolve_engine


class Transcriber:
    def __init__(self, spec: EngineSpec | None = None):
        self._spec = spec or resolve_engine(DEFAULT_ENGINE)
        self._model = load_model(self._spec.model_id)
        if self._spec.family == "whisper" and getattr(self._model, "_processor", None) is None:
            from transformers import WhisperProcessor

            self._model._processor = WhisperProcessor.from_pretrained(
                self._spec.processor_source
            )

    def transcribe(self, buffer: np.ndarray) -> str:
        result = self._model.generate(
            mx.array(buffer.astype(np.float32)), language=self._spec.language
        )
        text = getattr(result, "text", "") or ""
        return text.strip()
