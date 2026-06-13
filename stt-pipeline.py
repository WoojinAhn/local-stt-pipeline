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
from engines import DEFAULT_ENGINE, ENGINES, resolve_engine
from segmenter import VAD_SAMPLE_RATE, UtteranceSegmenter
from transcriber import Transcriber
from writer import TranscriptWriter

VAD_MODEL = "mlx-community/silero-vad"
BLOCK_SIZE = 1600  # 100 ms chunks at 16 kHz


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Real-time Korean speech-to-text")
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        choices=sorted(ENGINES),
        help=(
            f"STT tier (default: {DEFAULT_ENGINE}). "
            "high=Qwen3-ASR-1.7B, mid=whisper-large-v3, low=whisper-large-v3-turbo"
        ),
    )
    parser.add_argument(
        "--model", default=None, help="Override the tier's model id (keeps the tier's family)"
    )
    parser.add_argument(
        "--language", default=None, help="Override the language passed to the model"
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=900,
        help="Silence (ms) that ends an utterance (default: 900; higher = fuller sentences, more latency)",
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=0.5, help="VAD speech probability threshold (default: 0.5)"
    )
    parser.add_argument(
        "--min-utterance-ms",
        type=int,
        default=300,
        help="Drop utterances shorter than this (default: 300)",
    )
    parser.add_argument("--output-dir", default="outputs/stt", help="Where to save transcripts")
    parser.add_argument("--no-save", action="store_true", help="Do not save a transcript file")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    console = Console()

    spec = resolve_engine(args.engine, args.model, args.language)
    console.print(f"[cyan]loading[/cyan] {spec.tier} engine: {spec.model_id} …")
    vad = StreamingVad(
        load_vad(VAD_MODEL),
        ServerVadConfig(threshold=args.vad_threshold, silence_duration_ms=args.silence_ms),
    )
    segmenter = UtteranceSegmenter(vad, min_utterance_ms=args.min_utterance_ms)
    transcriber = Transcriber(spec)
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
