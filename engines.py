"""STT engine tiers selectable at runtime.

Accuracy ladder (high -> low). All tiers support Korean and run through mlx-audio.
ASR accuracy saturates at small model sizes, so tiers differ by model family and
quality, not by parameter count.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class EngineSpec:
    tier: str
    model_id: str
    family: str  # "whisper" | "qwen3_asr"
    language: str  # value passed to model.generate(language=...)
    # whisper repos on mlx-community ship no preprocessor_config.json, so a
    # WhisperProcessor must be loaded from here and injected. None for qwen3_asr.
    processor_source: Optional[str] = None


ENGINES = {
    "high": EngineSpec(
        tier="high",
        model_id="mlx-community/Qwen3-ASR-1.7B-8bit",
        family="qwen3_asr",
        language="Korean",
    ),
    "mid": EngineSpec(
        tier="mid",
        model_id="mlx-community/whisper-large-v3-mlx",
        family="whisper",
        language="ko",
        processor_source="openai/whisper-large-v3",
    ),
    "low": EngineSpec(
        tier="low",
        model_id="mlx-community/whisper-large-v3-turbo",
        family="whisper",
        language="ko",
        processor_source="openai/whisper-large-v3-turbo",
    ),
}

DEFAULT_ENGINE = "low"


def resolve_engine(
    tier: str = DEFAULT_ENGINE,
    model_override: Optional[str] = None,
    language_override: Optional[str] = None,
) -> EngineSpec:
    """Look up a tier and apply optional --model / --language overrides.

    An overridden model keeps the tier's family (loader/processor handling).
    """
    if tier not in ENGINES:
        raise ValueError(f"unknown engine tier {tier!r}; choose one of {sorted(ENGINES)}")
    spec = ENGINES[tier]
    if model_override:
        spec = replace(spec, model_id=model_override)
    if language_override:
        spec = replace(spec, language=language_override)
    return spec
