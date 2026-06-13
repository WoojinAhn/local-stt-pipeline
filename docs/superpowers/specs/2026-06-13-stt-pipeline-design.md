# STT Pipeline Design (`stt-pipeline.py`)

**Date:** 2026-06-13
**Status:** Approved design, pending implementation plan

## Goal

Add a third pipeline to the local LLM project: real-time Korean speech-to-text
(transcription only, no LLM analysis chain). Microphone input is segmented by
voice-activity detection, each utterance is transcribed with mlx-whisper, printed
live to the terminal, and the full session is saved to a text file on exit.

## Scope decisions (locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Purpose | Pure transcription (no LLM chain) | Standalone receiving/dictation tool |
| Engine | **mlx-whisper** | MLX-native, Metal GPU accel on Apple Silicon; ecosystem-consistent with mlx-lm/mlx-vlm. faster-whisper (CTranslate2) runs CPU-only on Mac. |
| Input | Real-time microphone streaming | |
| Streaming strategy | **VAD utterance-segmented** | Whisper is not a true streaming model (30s window). VAD detects speech boundaries; each completed utterance is transcribed. Stable text, 1–3s latency. |
| VAD library | **silero-vad** | Higher Korean/noise accuracy. Accepts PyTorch (~2GB) dependency as a deliberate trade-off over lightweight webrtcvad. |
| Output | Terminal (live) + file save | `.txt` with timestamps to `outputs/stt/` |
| Model | `mlx-community/whisper-large-v3-mlx` (4bit, ~1.5GB) | Korean accuracy/size balance |
| Target HW | Apple Silicon (M5 Max / 128GB assumed) | mlx-whisper install is uniform across Apple Silicon |

## Architecture

```
[mic] --sounddevice (16kHz mono)--> [frame queue]
                                          |
                                  VADIterator (silero-vad)
                                          |
                                  utterance audio buffer (float32)
                                          |
                          mlx_whisper.transcribe(buf, language="ko")
                                          |
                          Rich live output  +  in-memory accumulation
                                          |
                          Ctrl+C  ->  outputs/stt/YYYY-MM-DD-HHMMSS.txt
```

Two concurrent units, communicating through a thread-safe queue:

1. **Capture thread** — `sounddevice.InputStream` callback pushes audio frames
   (16 kHz, mono, float32) into a `queue.Queue`. Non-blocking; never touches the
   model.
2. **Main loop** — pulls frames, feeds them to silero-vad's `VADIterator`. On a
   speech-end event, assembles the buffered speech samples into one utterance
   array, calls `mlx_whisper.transcribe`, then renders + accumulates the result.

Decoupling capture from transcription means audio keeps buffering while the model
is busy on the previous utterance (no dropped speech).

## Components (single responsibility each)

| Function | Input → Output | Depends on |
|----------|----------------|-----------|
| `mic_stream(queue)` | mic → 16 kHz float32 frames into queue | sounddevice |
| `vad_segmenter(frames)` | frames → completed utterance buffers | silero-vad |
| `transcribe(buffer)` | utterance buffer → Korean text + segments | mlx-whisper |
| `render_and_accumulate(text)` | text → live Rich line + history list | rich |
| `save_session(history, path)` | history → `.txt` file | stdlib |

`vad_segmenter` is the testable core (deterministic given audio frames). Mic and
Rich are I/O boundaries.

## Data flow detail

- **Sample rate:** 16 kHz mono (Whisper's native rate; no resampling needed).
- **silero-vad:** use `VADIterator` which emits `{'start': ...}` / `{'end': ...}`
  events on 512-sample (32 ms) chunks at 16 kHz. Buffer speech samples between
  start and end; emit the buffer as one utterance on `end`.
- **Min-utterance guard:** skip buffers shorter than ~0.3 s to avoid Whisper
  hallucinating text on near-silent blips.
- **Transcription:** `mlx_whisper.transcribe(audio_array, path_or_hf_repo=MODEL,
  language="ko")`. Use `.text` (stripped). Drop empty results.
- **Timestamp:** wall-clock `[HH:MM:SS]` at utterance start, prepended to each line.

## Output format

Live terminal: each finalized utterance printed as `[HH:MM:SS] <text>` via Rich.

Saved file `outputs/stt/YYYY-MM-DD-HHMMSS.txt`:
```
[14:03:12] 안녕하세요, 회의를 시작하겠습니다.
[14:03:18] 오늘 안건은 세 가지입니다.
```
`outputs/` is already gitignored, so sessions are not committed.

## CLI

Minimal flags (YAGNI):
- `--language` (default `ko`)
- `--model` (default `mlx-community/whisper-large-v3-mlx`)
- `--output-dir` (default `outputs/stt`)
- `--no-save` (terminal only)

Startup sequence: print "loading model…", load + warm up (first inference is slow),
print "listening… (Ctrl+C to stop)", then enter the loop.

## Error handling

- **No microphone / sounddevice error:** catch on stream open, print a clear
  message naming the likely cause (no input device / permission), exit non-zero.
- **silero-vad model fetch:** small (~2 MB), torch.hub cache; surface download
  errors plainly.
- **Empty / hallucinated transcription:** min-utterance guard + drop empty text;
  rely on VAD gating instead of feeding silence to Whisper.
- **Ctrl+C:** treated as normal stop — flush any in-progress utterance, save, exit 0.

## Testing

Per project convention (test critical business logic only):

- **Unit — `vad_segmenter`:** feed a known sample `.wav` (speech + silence gaps)
  as frames; assert it emits the expected number of utterance buffers with
  plausible lengths. This is the deterministic core.
- **Smoke — `transcribe`:** run a short Korean sample `.wav` through
  `mlx_whisper.transcribe`; assert non-empty text. (Marked slow / model-gated;
  may be skipped in CI without the model cached.)
- **Manual verification:** run the tool, speak a few Korean sentences, confirm
  live terminal output and the saved `.txt` content. Capture terminal output as
  evidence.

## Files touched

New:
- `stt-pipeline.py`
- `outputs/stt/` (gitignored, created on first run)
- test sample `.wav` + test file (location per writing-plans)

Modified (coordinator-owned, batch-updated):
- `requirements.txt` — add `mlx-whisper`, `sounddevice`, `silero-vad` (pulls torch)
- `README.md` — new pipeline section, both Korean and English halves kept equivalent
- `CLAUDE.md` — new pipeline entry under Architecture + Key Files

## Process notes

- **Issue-first:** per project workflow, register an English GitHub issue before
  any code modification. The writing-plans phase folds issue creation in as the
  first step.
- **Dependency cost:** silero-vad adds PyTorch (~2 GB on disk). Acknowledged and
  accepted for Korean VAD accuracy on a 128 GB machine.
