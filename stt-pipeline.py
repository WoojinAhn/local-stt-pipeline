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
from rich.markup import escape

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
                        console.print(f"[dim]\\[{ts}][/dim] {escape(text)}")
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
