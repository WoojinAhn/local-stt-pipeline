# local-stt-pipeline

[English](#english) | [한국어](#한국어)

> 🚧 **In development / 개발 중** — design approved, implementation pending.
> See [`docs/superpowers/specs/`](docs/superpowers/specs/) for the design.

---

## 한국어

Apple Silicon Mac에서 동작하는 **로컬 실시간 한국어 받아쓰기** 도구입니다.
[`mlx-audio`](https://github.com/Blaizzy/mlx-audio) 라이브러리 위의 얇은
래퍼로, 마이크 입력을 음성 활동 감지(VAD)로 발화 단위로 끊어 Whisper(Metal GPU
가속)로 전사하고 터미널에 실시간 출력합니다. 세션 종료 시 타임스탬프가 붙은
텍스트 파일로 저장합니다. **전사 전용**으로, LLM 분석 체인은 없습니다.

### 바퀴를 재발명하지 않습니다

엔진·VAD·스트리밍은 검증된 `mlx-audio`(~7k★, 유지보수 활발, MIT)에 위임하고,
이 저장소는 **얇은 글루 코드만** 구현합니다 — 마이크 캡처(mlx-audio에 없는
부분, `sounddevice`) + VAD 분절 루프 + Rich 출력 + 세션 저장.

### 특징

- **mlx-audio** — Whisper STT 엔진(`whisper-large-v3-turbo`, Metal GPU)과
  **MLX 네이티브 silero-vad** 제공 → **PyTorch 의존성 없음.**
- **sounddevice** — 마이크 캡처(16kHz 모노).
- 캡처 스레드 + 전사 루프 분리 — 모델이 바쁜 동안에도 음성 유실 없음.
- Rich 실시간 출력 + `outputs/stt/*.txt` 자동 저장.

### 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> 깔끔한 MLX 스택 — PyTorch 없음.

### 사용 (예정)

```bash
python3 stt-pipeline.py            # 실시간 받아쓰기 시작, Ctrl+C로 종료
python3 stt-pipeline.py --no-save  # 파일 저장 없이 터미널만
```

### 동작 원리

Whisper 계열은 30초 윈도우 단위로 완성된 오디오를 전사하는 구조라, 말하는 즉시
글자가 흐르는 진짜 토큰 스트리밍은 설계상 불가능합니다. 이 도구는 VAD로 발화
경계를 감지해, 한 발화가 끝나면 그 구간을 전사하는 방식으로 "실시간"을
구현합니다(지연 1~3초, 텍스트가 흔들리지 않음).

---

## English

A **local real-time Korean speech-to-text** tool for Apple Silicon Macs. A thin
wrapper over the [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) library:
microphone input is segmented into utterances by voice-activity detection (VAD),
each utterance is transcribed with Whisper (Metal GPU accelerated) and printed
live to the terminal. On exit the full session is saved to a timestamped text
file. **Transcription only** — no LLM analysis chain.

### Don't reinvent the wheel

Engine, VAD, and streaming are delegated to the maintained `mlx-audio` (~7k★,
MIT). This repo implements only the thin glue: microphone capture (which
mlx-audio lacks, via `sounddevice`) + a VAD segmentation loop + Rich output +
session save.

### Features

- **mlx-audio** — provides the Whisper STT engine (`whisper-large-v3-turbo`,
  Metal GPU) and an **MLX-native silero-vad** → **no PyTorch dependency.**
- **sounddevice** — microphone capture (16kHz mono).
- Capture thread decoupled from the transcription loop — no dropped speech while
  the model is busy.
- Rich live output + automatic `outputs/stt/*.txt` save.

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Clean MLX stack — no PyTorch.

### Usage (planned)

```bash
python3 stt-pipeline.py            # start real-time transcription, Ctrl+C to stop
python3 stt-pipeline.py --no-save  # terminal only, no file save
```

### How it works

Whisper-family models transcribe completed 30-second audio windows, so true
token-level streaming (letters flowing as you speak) is not possible by design.
This tool implements "real-time" by detecting utterance boundaries with VAD and
transcribing each utterance once it ends (1–3s latency, no text flicker).
