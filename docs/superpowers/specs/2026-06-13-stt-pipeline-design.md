# STT Pipeline Design (`stt-pipeline.py`)

**Date:** 2026-06-13
**Status:** Approved design (revised after OSS research), pending implementation plan

## Goal

A thin, programmable wrapper for real-time Korean speech-to-text. Microphone
input is segmented by voice-activity detection, each utterance is transcribed
with Whisper via the `mlx-audio` library, printed live to the terminal, and the
full session is saved to a timestamped text file on exit. Transcription only —
no LLM analysis chain.

## "Don't reinvent the wheel" — what we reuse vs. build

OSS research (2026-06-13) confirmed the engine + VAD + streaming layers already
exist. We build a **thin wrapper**, not the engine.

**Reused from `mlx-audio`** (Blaizzy, 7.3k★, actively maintained, MIT):
- Whisper transcription engine (`generate_transcription`), Metal GPU accelerated.
- **MLX-native silero-vad** (`mlx-community/silero-vad`) — eliminates the PyTorch
  (~2GB) dependency the original from-scratch design required.

**We build (thin):**
- Microphone capture (mlx-audio has no mic input API) via `sounddevice`.
- The glue loop: mic frames → VAD utterance segmentation → transcription → Rich
  live output → session save.

Alternatives considered and rejected: RealtimeSTT (9.9k★ but no MLX/Metal on Mac,
faster-whisper CPU only); Realtime_mlx_STT (exact match but 7★ / ~1yr stale);
finished apps Handy/VoiceInk (different UX — dictation into apps, not a
session-transcript pipeline); from-scratch (reinvents mlx-audio).

## Scope decisions (locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Purpose | Pure transcription (no LLM chain) | Standalone dictation/transcription tool |
| Base library | **mlx-audio** | Maintained MLX-native STT + VAD; avoids reinvention; torch-free VAD |
| Engine | **Whisper large-v3-turbo** (`mlx-community/whisper-large-v3-turbo-asr-fp16`) | Proven Korean accuracy. (Voxtral Realtime considered for native streaming but less Korean track record.) |
| Streaming strategy | **VAD utterance-segmented** | Whisper is batch (30s window), not streaming. VAD segments speech; each completed utterance transcribed. 1–3s latency, stable text. |
| VAD | **mlx-audio's MLX silero-vad** | MLX-native, no PyTorch dependency |
| Mic capture | **sounddevice** | mlx-audio provides no microphone input |
| Output | Terminal (live) + file save | `.txt` with timestamps to `outputs/stt/` |
| Target HW | Apple Silicon (M5 Max / 128GB assumed) | mlx-audio is Metal-accelerated on Apple Silicon |

## Architecture

```
[mic] --sounddevice (16kHz mono)--> [frame queue]
                                          |
                          mlx-audio silero-vad (utterance boundaries)
                                          |
                                  utterance audio buffer (float32)
                                          |
                  mlx_audio.stt.generate.generate_transcription(buf, language="ko")
                                          |
                          Rich live output  +  in-memory accumulation
                                          |
                          Ctrl+C  ->  outputs/stt/YYYY-MM-DD-HHMMSS.txt
```

Two concurrent units, communicating through a thread-safe queue:

1. **Capture thread** — `sounddevice.InputStream` callback pushes 16 kHz mono
   float32 frames into a `queue.Queue`. Never touches the model.
2. **Main loop** — pulls frames, drives mlx-audio's silero-vad to detect
   speech start/end, assembles each completed utterance, calls mlx-audio
   transcription, then renders + accumulates.

Decoupling capture from transcription keeps audio buffering while the model is
busy on the previous utterance (no dropped speech).

## Components (single responsibility each)

| Function | Input → Output | Depends on |
|----------|----------------|-----------|
| `mic_stream(queue)` | mic → 16 kHz float32 frames into queue | sounddevice |
| `vad_segmenter(frames)` | frames → completed utterance buffers | mlx-audio silero-vad |
| `transcribe(buffer)` | utterance buffer → Korean text | mlx-audio (`generate_transcription`) |
| `render_and_accumulate(text)` | text → live Rich line + history list | rich |
| `save_session(history, path)` | history → `.txt` file | stdlib |

`vad_segmenter` is the testable core. Mic and Rich are I/O boundaries.

## Open implementation question (verify, don't guess)

The exact API for driving mlx-audio's MLX silero-vad over **streaming mic frames**
(stateful start/end events vs. file-oriented) is not fully documented in the
README. The implementation plan's first technical step is a spike: confirm the
VAD streaming API from mlx-audio source/examples. Fallbacks if no frame-streaming
API exists:
- Accumulate a rolling buffer and run VAD on short windows to find boundaries, or
- Run silero-vad on fixed ~1s windows and merge contiguous speech windows.

Likewise confirm whether `generate_transcription` accepts an in-memory float32
array directly or requires a temp `.wav`; if the latter, write the utterance to a
tmp file before transcription.

## Data flow detail

- **Sample rate:** 16 kHz mono (Whisper native; no resampling).
- **Min-utterance guard:** skip buffers shorter than ~0.3 s (avoid hallucination
  on near-silent blips).
- **Transcription:** `generate_transcription(model=MODEL, audio=buf,
  language="ko")`; use the text, dropped if empty.
- **Timestamp:** wall-clock `[HH:MM:SS]` at utterance start, prepended per line.

## Output format

Live terminal: `[HH:MM:SS] <text>` via Rich.

Saved `outputs/stt/YYYY-MM-DD-HHMMSS.txt`:
```
[14:03:12] 안녕하세요, 회의를 시작하겠습니다.
[14:03:18] 오늘 안건은 세 가지입니다.
```
`outputs/` is gitignored, so sessions are not committed.

## CLI

Minimal flags (YAGNI):
- `--language` (default `ko`)
- `--model` (default `mlx-community/whisper-large-v3-turbo-asr-fp16`)
- `--output-dir` (default `outputs/stt`)
- `--no-save` (terminal only)

Startup: print "loading model…", load + warm up, print "listening… (Ctrl+C to
stop)", enter the loop.

## Error handling

- **No microphone / sounddevice error:** catch on stream open; clear message
  (no input device / permission); exit non-zero.
- **Model / VAD fetch:** HuggingFace cache; surface download errors plainly.
- **Empty / hallucinated transcription:** min-utterance guard + drop empty text.
- **Ctrl+C:** normal stop — flush in-progress utterance, save, exit 0.

## Testing

Per project convention (test critical business logic only):

- **Unit — `vad_segmenter`:** feed a known sample `.wav` (speech + silence gaps)
  as frames; assert the expected number of utterance buffers. Deterministic core.
- **Smoke — `transcribe`:** short Korean sample `.wav` → non-empty text.
  (Slow / model-gated; skippable without model cached.)
- **Manual verification:** run, speak Korean, confirm live terminal output and
  saved `.txt`. Capture terminal output as evidence.

## Files

New:
- `stt-pipeline.py`
- `outputs/stt/` (gitignored, created on first run)
- test sample `.wav` + test file (location per writing-plans)

Modified (already scaffolded in repo, updated for this design):
- `requirements.txt` — `mlx-audio`, `sounddevice`, `rich` (no torch/silero-vad)
- `README.md` — both Korean and English halves kept equivalent
- `CLAUDE.md` — architecture + dependencies

## Process notes

- **Issue-first:** GitHub issue #1 tracks implementation (English). Update it to
  the mlx-audio-based approach.
- **Dependency win:** PyTorch dependency eliminated — mlx-audio's silero-vad is
  MLX-native. The repo stays a clean MLX stack.
