# STT Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `stt-pipeline.py`, a thin wrapper over `mlx-audio` that turns live microphone speech into a real-time Korean transcript printed to the terminal and saved to a timestamped file.

**Architecture:** A `sounddevice` capture thread feeds 16 kHz float32 audio into a queue. The main loop drives `mlx-audio`'s `StreamingVad` (MLX-native silero-vad) to detect utterance boundaries via `UtteranceSegmenter`, transcribes each completed utterance with a once-loaded Whisper model (`model.generate`, NOT `generate_transcription` — the latter writes a file every call), renders it live with Rich, and saves the session on Ctrl+C.

**Tech Stack:** Python 3, `mlx-audio` (Whisper engine + silero-vad), `sounddevice`, `rich`, `mlx`, `numpy`, `pytest` (dev).

---

## Confirmed mlx-audio API (from source, 2026-06-13)

These were read directly from `Blaizzy/mlx-audio@main` and are the basis for the code below:

- **VAD load:** `from mlx_audio.vad import load as load_vad` → `vad_model = load_vad("mlx-community/silero-vad")`. The silero `Model` exposes `initial_state(sample_rate=)` and `feed(chunk, state, sample_rate=) -> (probability, state)`.
- **Streaming VAD:** `from mlx_audio.realtime_vad import StreamingVad, ServerVadConfig, TurnEvent, TurnEventKind`.
  - `StreamingVad(vad_model, ServerVadConfig(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500))`.
  - `.process(samples: np.ndarray) -> List[TurnEvent]` — feed 16 kHz float32 samples; returns turn events. Rebuffers to 512-sample (32 ms) frames internally, so any input chunk size works.
  - `TurnEvent.kind ∈ {TurnEventKind.SPEECH_STARTED, TurnEventKind.SPEECH_STOPPED}`; `.audio_ms` is the session-relative offset.
  - `VAD_SAMPLE_RATE = 16000`.
- **STT model load:** `from mlx_audio.stt.utils import load_model` → `model = load_model("mlx-community/whisper-large-v3-turbo")` (loads once).
- **Transcribe:** call `model.generate(mx.array(buffer), language="ko")` directly. Returns an `STTOutput` with `.text`. **Do NOT use `generate_transcription` in the loop** — it always calls `save_as_txt(...)` and `os.makedirs(...)`, writing a file on every call.
- The one line to confirm empirically in Task 1 is the exact `model.generate(...)` kwargs (whether `language=` alone is accepted, or `generation_stream=` is also required).

## File structure

| File | Responsibility | Tested |
|------|----------------|--------|
| `segmenter.py` | `UtteranceSegmenter`: turn-event stream → completed utterance buffers. Pure glue, model-free, VAD injected. | Unit (fake VAD) |
| `transcriber.py` | `Transcriber`: load Whisper once, `.transcribe(buffer) -> str` via `model.generate`. | Manual smoke |
| `stt-pipeline.py` | CLI entry: args, wire VAD+segmenter+transcriber, mic capture thread, main loop, Rich render, session save. | Manual e2e |
| `tests/test_segmenter.py` | Unit tests for `UtteranceSegmenter` with a scripted fake VAD. | — |
| `requirements-dev.txt` | `pytest`. | — |

---

### Task 1: Environment + API spike

**Files:**
- Create: `requirements-dev.txt`

- [ ] **Step 1: Create venv and install runtime deps**

Run:
```bash
cd ~/home/local-stt-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Expected: `mlx-audio`, `sounddevice`, `rich` (and their deps incl. `mlx`, `numpy`) install. No `torch` in the tree — verify:
```bash
pip list | grep -i torch || echo "no torch — good"
```
Expected: `no torch — good`.

- [ ] **Step 2: Create dev requirements and install pytest**

Create `requirements-dev.txt`:
```
pytest
```
Run:
```bash
pip install -r requirements-dev.txt
```

- [ ] **Step 3: Confirm the VAD + transcription API empirically**

Run this spike script (records the exact working call):
```bash
python3 - <<'PY'
import numpy as np, mlx.core as mx
from mlx_audio.vad import load as load_vad
from mlx_audio.realtime_vad import StreamingVad, ServerVadConfig, TurnEventKind
from mlx_audio.stt.utils import load_model

# 1) VAD: feed 2s of noise then silence, expect it constructs and runs
vad = StreamingVad(load_vad("mlx-community/silero-vad"), ServerVadConfig())
events = vad.process(np.random.randn(16000).astype(np.float32) * 0.1)
print("VAD ok, event kinds:", [e.kind for e in events])

# 2) STT: load once, transcribe 1s of silence via model.generate (NOT generate_transcription)
model = load_model("mlx-community/whisper-large-v3-turbo")
try:
    out = model.generate(mx.array(np.zeros(16000, dtype=np.float32)), language="ko")
    print("generate(language=ko) OK; text repr:", repr(getattr(out, "text", out)))
except TypeError as e:
    print("language-only failed:", e)
PY
```
Expected: prints "VAD ok" and "generate(language=ko) OK" with a `.text` attribute (likely empty string for silence). **If the `model.generate` call raises**, record the corrected signature and use it in Task 3 (e.g. add `generation_stream=mx.new_stream(mx.default_device())`).

- [ ] **Step 4: Commit**

```bash
git add requirements-dev.txt
git commit -m "[#1] chore: add dev deps, confirm mlx-audio VAD/STT API"
```

---

### Task 2: `UtteranceSegmenter` (testable core)

**Files:**
- Create: `segmenter.py`
- Test: `tests/test_segmenter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_segmenter.py`:
```python
import numpy as np
from mlx_audio.realtime_vad import TurnEvent, TurnEventKind
from segmenter import UtteranceSegmenter, VAD_SAMPLE_RATE


class FakeVad:
    """Scripted VAD: emits the events queued for each process() call index."""

    def __init__(self, script):
        self._script = script  # dict: call_index -> list[TurnEvent]
        self._i = 0

    def process(self, samples):
        events = self._script.get(self._i, [])
        self._i += 1
        return events


def _started():
    return TurnEvent(kind=TurnEventKind.SPEECH_STARTED, audio_ms=0)


def _stopped():
    return TurnEvent(kind=TurnEventKind.SPEECH_STOPPED, audio_ms=0)


def test_emits_one_utterance_between_start_and_stop():
    chunk = np.ones(1600, dtype=np.float32)  # 100 ms
    vad = FakeVad({1: [_started()], 5: [_stopped()]})
    seg = UtteranceSegmenter(vad, pre_roll_ms=0, min_utterance_ms=0)

    completed = []
    for _ in range(7):
        completed.extend(seg.feed(chunk))

    assert len(completed) == 1
    # chunks 1,2,3,4 collected before stop fires on chunk 5 -> ~4 chunks
    assert completed[0].shape[0] == 1600 * 4


def test_min_utterance_guard_drops_short_blip():
    chunk = np.ones(1600, dtype=np.float32)
    vad = FakeVad({0: [_started()], 1: [_stopped()]})
    seg = UtteranceSegmenter(vad, pre_roll_ms=0, min_utterance_ms=500)

    completed = []
    for _ in range(3):
        completed.extend(seg.feed(chunk))

    assert completed == []  # only ~1 chunk (100ms) < 500ms guard


def test_no_events_yields_nothing():
    seg = UtteranceSegmenter(FakeVad({}), pre_roll_ms=0, min_utterance_ms=0)
    assert seg.feed(np.ones(1600, dtype=np.float32)) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_segmenter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'segmenter'`.

- [ ] **Step 3: Write the minimal implementation**

Create `segmenter.py`:
```python
"""Turn a stream of mic samples into completed utterance buffers via VAD."""

from __future__ import annotations

from typing import List

import numpy as np

from mlx_audio.realtime_vad import TurnEventKind

VAD_SAMPLE_RATE = 16000


class UtteranceSegmenter:
    """Drive utterance segmentation from a StreamingVad's turn events.

    ``vad`` must expose ``process(samples) -> list[TurnEvent]``. While in speech,
    incoming samples are accumulated; on SPEECH_STOPPED the utterance is emitted
    (unless shorter than ``min_utterance_ms``). A pre-roll keeps the start of
    speech from being clipped.
    """

    def __init__(self, vad, pre_roll_ms: int = 300, min_utterance_ms: int = 300):
        self._vad = vad
        self._pre_roll = int(pre_roll_ms * VAD_SAMPLE_RATE / 1000)
        self._min_samples = int(min_utterance_ms * VAD_SAMPLE_RATE / 1000)
        self._tail = np.zeros(0, dtype=np.float32)
        self._collecting = False
        self._utterance = np.zeros(0, dtype=np.float32)

    def feed(self, samples: np.ndarray) -> List[np.ndarray]:
        samples = samples.astype(np.float32)
        events = self._vad.process(samples)
        completed: List[np.ndarray] = []

        for event in events:
            if event.kind == TurnEventKind.SPEECH_STARTED and not self._collecting:
                self._collecting = True
                self._utterance = self._tail.copy()  # pre-roll captured before this chunk
            elif event.kind == TurnEventKind.SPEECH_STOPPED and self._collecting:
                self._collecting = False
                utterance = self._utterance
                self._utterance = np.zeros(0, dtype=np.float32)
                if utterance.shape[0] >= self._min_samples:
                    completed.append(utterance)

        if self._collecting:
            self._utterance = np.concatenate([self._utterance, samples])
        if self._pre_roll:
            self._tail = np.concatenate([self._tail, samples])[-self._pre_roll:]

        return completed
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_segmenter.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add segmenter.py tests/test_segmenter.py
git commit -m "[#1] feat: utterance segmenter over StreamingVad turn events"
```

---

### Task 3: `Transcriber` (load model once, transcribe buffers)

**Files:**
- Create: `transcriber.py`

- [ ] **Step 1: Write the implementation**

Create `transcriber.py` (use the `model.generate` call confirmed in Task 1 Step 3; the version below assumes `language=` is accepted — adjust if the spike showed otherwise):
```python
"""Load a Whisper STT model once and transcribe 16 kHz float32 buffers.

Uses ``model.generate`` directly rather than ``generate_transcription`` because
the latter writes a transcript file (``save_as_txt``) on every call.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import load_model

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


class Transcriber:
    def __init__(self, model_id: str = DEFAULT_MODEL, language: str = "ko"):
        self._model = load_model(model_id)
        self._language = language

    def transcribe(self, buffer: np.ndarray) -> str:
        result = self._model.generate(
            mx.array(buffer.astype(np.float32)), language=self._language
        )
        text = getattr(result, "text", "") or ""
        return text.strip()
```

- [ ] **Step 2: Manual smoke test**

Run:
```bash
python3 - <<'PY'
import numpy as np
from transcriber import Transcriber
t = Transcriber()
print("empty-on-silence:", repr(t.transcribe(np.zeros(16000, dtype=np.float32))))
print("loaded once, reusable")
PY
```
Expected: loads the model once, prints a (likely empty) string for silence without raising and without writing any `transcript.txt`. Verify no file was written:
```bash
ls transcript.txt 2>/dev/null && echo "BUG: file written" || echo "no file written — good"
```
Expected: `no file written — good`.

- [ ] **Step 3: Commit**

```bash
git add transcriber.py
git commit -m "[#1] feat: Whisper transcriber (load once, no file side-effect)"
```

---

### Task 4: `stt-pipeline.py` entry — mic capture + main loop

**Files:**
- Create: `stt-pipeline.py`

- [ ] **Step 1: Write the entry script**

Create `stt-pipeline.py`:
```python
"""Real-time Korean speech-to-text. Thin wrapper over mlx-audio.

Mic -> sounddevice -> StreamingVad utterance segmentation -> Whisper -> Rich.
"""

from __future__ import annotations

import argparse
import queue
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console

from mlx_audio.realtime_vad import ServerVadConfig, StreamingVad
from mlx_audio.vad import load as load_vad
from segmenter import VAD_SAMPLE_RATE, UtteranceSegmenter
from transcriber import DEFAULT_MODEL, Transcriber

VAD_MODEL = "mlx-community/silero-vad"
BLOCK_SIZE = 1600  # 100 ms chunks at 16 kHz


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Real-time Korean speech-to-text")
    parser.add_argument("--language", default="ko", help="Language code (default: ko)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="STT model id")
    parser.add_argument("--output-dir", default="outputs/stt", help="Where to save transcripts")
    parser.add_argument("--no-save", action="store_true", help="Do not save a transcript file")
    return parser.parse_args(argv)


def save_session(history, output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    path = out_dir / f"{stamp}.txt"
    path.write_text("".join(f"[{ts}] {text}\n" for ts, text in history), encoding="utf-8")
    return path


def main():
    args = parse_args()
    console = Console()

    console.print("[cyan]loading model…[/cyan]")
    vad = StreamingVad(load_vad(VAD_MODEL), ServerVadConfig())
    segmenter = UtteranceSegmenter(vad)
    transcriber = Transcriber(args.model, args.language)

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            console.print(f"[yellow]{status}[/yellow]")
        audio_q.put(indata[:, 0].copy())

    history = []
    console.print("[green]listening… (Ctrl+C to stop)[/green]")
    try:
        with sd.InputStream(
            samplerate=VAD_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=callback,
        ):
            while True:
                samples = audio_q.get()
                for utterance in segmenter.feed(samples):
                    text = transcriber.transcribe(utterance)
                    if text:
                        ts = datetime.now().strftime("%H:%M:%S")
                        console.print(f"[dim]\\[{ts}][/dim] {text}")
                        history.append((ts, text))
    except KeyboardInterrupt:
        console.print("\n[cyan]stopped.[/cyan]")
    except sd.PortAudioError as exc:
        console.print(f"[red]microphone error:[/red] {exc}")
        console.print("[red]check the input device and macOS microphone permission.[/red]")
        sys.exit(1)

    if not args.no_save and history:
        path = save_session(history, args.output_dir)
        console.print(f"[green]saved:[/green] {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports and parses args without a mic**

Run: `python3 stt-pipeline.py --help`
Expected: argparse help text listing `--language`, `--model`, `--output-dir`, `--no-save`. No exceptions.

- [ ] **Step 3: Commit**

```bash
git add stt-pipeline.py
git commit -m "[#1] feat: real-time STT entry — mic capture, loop, render, save"
```

---

### Task 5: End-to-end manual verification

**Files:** none (verification only)

- [ ] **Step 1: Run and speak Korean**

Run:
```bash
source .venv/bin/activate
python3 stt-pipeline.py
```
Speak 2–3 Korean sentences with short pauses, then press Ctrl+C.
Expected: each sentence appears live as `[HH:MM:SS] <korean text>`; on Ctrl+C it prints `saved: outputs/stt/<timestamp>.txt`.

- [ ] **Step 2: Verify the saved transcript**

Run: `cat outputs/stt/*.txt | tail`
Expected: timestamped Korean lines matching what was spoken. Capture this output as the completion evidence.

- [ ] **Step 3: Verify --no-save**

Run: `python3 stt-pipeline.py --no-save`, speak one sentence, Ctrl+C.
Expected: live output appears; no new file under `outputs/stt/`.

---

### Task 6: Finalize docs

**Files:**
- Modify: `README.md` (both 한국어 and English halves), `CLAUDE.md`

- [ ] **Step 1: Flip status from "in development" to working**

In `README.md`, remove the `🚧 In development / 개발 중` blockquote near the top and change the two `사용 (예정)` / `Usage (planned)` headings to `사용` / `Usage`. Keep the Korean and English halves equivalent.

- [ ] **Step 2: Note the working state in CLAUDE.md**

In `CLAUDE.md` under `## Known Issues`, replace `- (none yet — pre-implementation)` with any real issues found during Task 5 (e.g. latency notes, mic-permission gotchas), or `- (none)` if clean.

- [ ] **Step 3: Commit and push**

```bash
git add README.md CLAUDE.md
git commit -m "[#1] docs: mark STT pipeline working; usage notes"
git push origin main
```

- [ ] **Step 4: Close the issue**

```bash
gh issue close 1 --comment "Implemented in $(git rev-parse --short HEAD). Real-time Korean STT working: mic -> StreamingVad -> Whisper -> Rich + saved transcript."
```

---

## Self-review notes

- **Spec coverage:** mic capture (Task 4), VAD segmentation (Task 2), transcription without file side-effect (Task 3), Rich live output + save (Task 4), CLI flags (Task 4), min-utterance guard (Task 2), mic error handling (Task 4), VAD-segmenter unit test (Task 2), no-PyTorch check (Task 1). The spec's "open implementation question" is resolved by the Task 1 spike + confirmed-API section.
- **Single unverified line:** the exact `model.generate(...)` kwargs — Task 1 Step 3 confirms it before Task 3 depends on it.
- **Naming consistency:** `UtteranceSegmenter.feed`, `Transcriber.transcribe`, `save_session`, `VAD_SAMPLE_RATE`, `DEFAULT_MODEL` used consistently across `segmenter.py`, `transcriber.py`, `stt-pipeline.py`, and tests.
