# CLAUDE.md

## Overview

Local real-time Korean speech-to-text. A thin wrapper over the `mlx-audio`
library: microphone input is segmented by voice-activity detection, each
utterance is transcribed with Whisper (Metal GPU on Apple Silicon), printed live
to the terminal, and the full session is saved to a timestamped text file.
Transcription only — no LLM analysis chain.

Sibling project: `local-llm-pipeline` (text/multimodal LLM pipelines). STT was
split into its own repo to keep its dependencies isolated from that project's
LLM stack.

## Don't reinvent the wheel

OSS research (see spec) confirmed the engine + VAD + streaming layers already
exist in `mlx-audio` (Blaizzy, ~7k★, maintained, MIT). We build only the thin
glue: mic capture (`sounddevice`, which mlx-audio lacks) + VAD segmentation loop
+ Rich output + session save. We do NOT reimplement the engine or VAD.

## Architecture

### stt-pipeline.py (thin wrapper over mlx-audio)

- **mlx-audio** provides the Whisper STT engine
  (`mlx-community/whisper-large-v3-turbo-asr-fp16`, Metal GPU) and an
  **MLX-native silero-vad** — so there is NO PyTorch dependency.
- **sounddevice** captures the microphone (16kHz mono) into a thread-safe queue;
  mlx-audio has no mic input API.
- Whisper is batch (30s window), not streaming. "Real-time" = silero-vad segments
  speech, each completed utterance is transcribed (1–3s latency, stable text).
- Two concurrent units: a capture thread fills the queue; the main loop runs VAD
  + transcription so audio keeps buffering while the model is busy.
- Output: Rich live lines `[HH:MM:SS] text`, saved to
  `outputs/stt/YYYY-MM-DD-HHMMSS.txt` on Ctrl+C.

## Key Files

- `stt-pipeline.py`: real-time STT pipeline. Thin wrapper over mlx-audio.
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

- Dependencies: `pip install -r requirements.txt` (venv recommended). Clean MLX
  stack — no PyTorch (mlx-audio's silero-vad is MLX-native).
- Model downloads from HuggingFace cache (`~/.cache/huggingface/hub/`); HF token
  recommended to avoid rate limits on first download.
- Whisper is chunk-based, not streaming: never claim token-level real-time output.
- Microphone requires macOS input permission for the terminal app.
- The exact mlx-audio silero-vad streaming API must be confirmed from source
  before implementing the segmenter (see spec "Open implementation question").

## Known Issues

- (none yet — pre-implementation)
