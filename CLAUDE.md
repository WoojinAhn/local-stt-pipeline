# CLAUDE.md

## Overview

Local real-time Korean speech-to-text. Microphone input is segmented by
voice-activity detection, each utterance is transcribed with mlx-whisper (Metal
GPU on Apple Silicon), printed live to the terminal, and the full session is
saved to a timestamped text file. Transcription only — no LLM analysis chain.

Sibling project: `local-llm-pipeline` (text/multimodal LLM pipelines). STT was
split into its own repo to keep the heavy audio/torch dependencies isolated from
that project's clean MLX stack.

## Architecture

### stt-pipeline.py (mlx-whisper direct inference)

- **mlx-whisper** `whisper-large-v3-mlx` (4bit, ~1.5GB): Korean STT, Metal GPU
  accelerated. Native 16kHz mono input.
- **silero-vad** (PyTorch): utterance-boundary detection. Whisper is not a true
  streaming model (30s window), so "real-time" = VAD segments speech, each
  completed utterance is transcribed (1–3s latency, stable text).
- **sounddevice**: microphone capture into a thread-safe frame queue.
- Two concurrent units: a capture thread fills the queue; the main loop runs VAD
  + transcription so audio keeps buffering while the model is busy.
- Output: Rich live terminal lines `[HH:MM:SS] text`, saved to
  `outputs/stt/YYYY-MM-DD-HHMMSS.txt` on Ctrl+C.

## Key Files

- `stt-pipeline.py`: real-time STT pipeline. mlx-whisper direct inference.
- `docs/superpowers/specs/`: design specs.

## README Convention

- README.md keeps Korean and English sections equivalent.
- Editing one half requires the same edit to the other.
- Keep the `[English](#english) | [한국어](#한국어)` anchor links at the top.

## Issue-First Workflow

For non-trivial code changes (features, behavior changes), register a GitHub
issue (English) before code modification. Root-cause analysis and exploration may
precede the issue; implementation follows it.

## Development Guidelines

- Dependencies: `pip install -r requirements.txt` (venv recommended). Note
  silero-vad pulls PyTorch (~2GB) — accepted trade-off for VAD accuracy.
- Model downloads from HuggingFace cache (`~/.cache/huggingface/hub/`); HF token
  recommended to avoid rate limits on first download.
- Whisper is chunk-based, not streaming: never claim token-level real-time output.
- Microphone requires macOS input permission for the terminal app.

## Known Issues

- (none yet — pre-implementation)
