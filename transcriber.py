"""Load a Whisper STT model once and transcribe 16 kHz float32 buffers.

Uses ``model.generate`` directly rather than ``generate_transcription`` because
the latter writes a transcript file (``save_as_txt``) on every call.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx_audio.stt.utils import load_model
from transformers import WhisperProcessor

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
# mlx-community Whisper repos ship weights but no preprocessor_config.json, so the
# processor must be loaded separately and injected after load_model().
_PROCESSOR_FALLBACK = "openai/whisper-large-v3-turbo"


class Transcriber:
    def __init__(self, model_id: str = DEFAULT_MODEL, language: str = "ko"):
        self._model = load_model(model_id)
        if getattr(self._model, "_processor", None) is None:
            self._model._processor = WhisperProcessor.from_pretrained(_PROCESSOR_FALLBACK)
        self._language = language

    def transcribe(self, buffer: np.ndarray) -> str:
        result = self._model.generate(
            mx.array(buffer.astype(np.float32)), language=self._language
        )
        text = getattr(result, "text", "") or ""
        return text.strip()
