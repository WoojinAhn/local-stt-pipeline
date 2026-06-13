"""Real-time Korean speech-to-text. Thin wrapper over mlx-audio.

Mic -> sounddevice -> StreamingVad utterance segmentation -> Whisper -> Rich.
"""

from __future__ import annotations

import argparse
import queue
import sys
from datetime import datetime

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.markup import escape

from mlx_audio.realtime_vad import ServerVadConfig, StreamingVad
from mlx_audio.vad import load as load_vad
from segmenter import VAD_SAMPLE_RATE, UtteranceSegmenter
from transcriber import DEFAULT_MODEL, Transcriber
from writer import TranscriptWriter

VAD_MODEL = "mlx-community/silero-vad"
BLOCK_SIZE = 1600  # 100 ms chunks at 16 kHz


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Real-time Korean speech-to-text")
    parser.add_argument("--language", default="ko", help="Language code (default: ko)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="STT model id")
    parser.add_argument("--output-dir", default="outputs/stt", help="Where to save transcripts")
    parser.add_argument("--no-save", action="store_true", help="Do not save a transcript file")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    console = Console()

    console.print("[cyan]loading model…[/cyan]")
    vad = StreamingVad(load_vad(VAD_MODEL), ServerVadConfig())
    segmenter = UtteranceSegmenter(vad)
    transcriber = Transcriber(args.model, args.language)
    writer = TranscriptWriter(args.output_dir, enabled=not args.no_save)

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            console.print(f"[yellow]{status}[/yellow]")
        audio_q.put(indata[:, 0].copy())

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
                        writer.write_line(ts, text)
    except KeyboardInterrupt:
        console.print("\n[cyan]stopped.[/cyan]")
    except sd.PortAudioError as exc:
        console.print(f"[red]microphone error:[/red] {exc}")
        console.print("[red]check the input device and macOS microphone permission.[/red]")
        sys.exit(1)
    finally:
        writer.close()

    if writer.path:
        console.print(f"[green]saved:[/green] {writer.path}")


if __name__ == "__main__":
    main()
