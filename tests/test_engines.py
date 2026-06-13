import pytest

from engines import DEFAULT_ENGINE, ENGINES, resolve_engine


def test_default_engine_is_low_turbo():
    assert DEFAULT_ENGINE == "low"
    spec = resolve_engine()
    assert spec.tier == "low"
    assert spec.model_id == "mlx-community/whisper-large-v3-turbo"
    assert spec.family == "whisper"


def test_high_tier_is_qwen3_asr_korean():
    spec = resolve_engine("high")
    assert spec.family == "qwen3_asr"
    assert spec.language == "Korean"
    assert spec.processor_source is None  # qwen3_asr needs no WhisperProcessor


def test_whisper_tiers_have_processor_source():
    for tier in ("mid", "low"):
        spec = resolve_engine(tier)
        assert spec.family == "whisper"
        assert spec.language == "ko"
        assert spec.processor_source is not None


def test_model_override_keeps_family():
    spec = resolve_engine("high", model_override="mlx-community/Qwen3-ASR-0.6B-8bit")
    assert spec.model_id == "mlx-community/Qwen3-ASR-0.6B-8bit"
    assert spec.family == "qwen3_asr"  # family preserved from the tier


def test_language_override():
    spec = resolve_engine("low", language_override="en")
    assert spec.language == "en"


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        resolve_engine("ultra")
