# local-stt-pipeline

[English](#english) | [한국어](#한국어)

Local real-time Korean speech-to-text for Apple Silicon Macs. Microphone input is
segmented by voice-activity detection and transcribed with Whisper (Metal GPU),
printed live to the terminal and appended to a timestamped text file.

---

## 한국어

### 기능

- 실시간 마이크 받아쓰기 (한국어 기본, Whisper 99개 언어 지원)
- `mlx-whisper` (`whisper-large-v3-turbo`) — Apple Silicon Metal GPU 가속
- silero-VAD 발화 단위 분절 (PyTorch 불필요, MLX 네이티브)
- 전사 결과를 발화마다 `outputs/stt/<날짜>.txt`에 즉시 기록 (강제 종료 시에도 유실 최소)

### 요구 사항

- **칩**: Apple Silicon (M1/M2/M3/M4/M5 계열). MLX는 Apple Silicon 전용이라 Intel Mac에서는 동작하지 않습니다.
- **OS**: macOS 13.5 (Ventura) 이상 (MLX/Metal 요구).
- **Python**: 3.10 이상 (MLX 요구사항).
- **메모리**: 추론 피크 약 2.5GB(통합 메모리). 8GB RAM이면 충분합니다.
- **디스크**: 모델 약 1.5GB(Whisper) + VAD ~2MB. `~/.cache/huggingface`에 캐시되며 venv 의존성은 별도입니다.
- **마이크**: 입력 장치 + 터미널 앱의 macOS 마이크 접근 권한.
- **네트워크**: 첫 실행 시 모델 다운로드용 인터넷 연결 (`HF_TOKEN` 환경변수 권장 — rate limit 회피).

검증 환경: Apple M5 Max / macOS 26 / Python 3.14 / mlx 0.31.

### 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 사용

```bash
python3 stt-pipeline.py              # 실시간 받아쓰기 시작, Ctrl+C로 종료
python3 stt-pipeline.py --engine high  # 최고 정확도(Qwen3-ASR)
python3 stt-pipeline.py --no-save    # 파일 저장 없이 터미널만
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--engine` | `low` | STT 티어 (`high`/`mid`/`low`) |
| `--model` | — | 티어 모델 ID 덮어쓰기 (티어의 패밀리 유지) |
| `--language` | — | 모델에 전달할 언어 덮어쓰기 |
| `--output-dir` | `outputs/stt` | 전사 파일 저장 위치 |
| `--no-save` | — | 파일 저장 안 함 |

### 엔진 티어 (`--engine`)

정확도 사다리입니다. ASR 정확도는 작은 크기에서 포화되므로 티어는 모델 크기가
아니라 **모델 패밀리/품질**로 나뉩니다. 첫 실행 시 해당 모델을 내려받습니다.

| 티어 | 모델 | 성격 |
|------|------|------|
| `high` (상) | Qwen3-ASR-1.7B | 2026 SOTA, 최고 정확도 |
| `mid` (중) | Whisper large-v3 (full) | 검증된 정확도 |
| `low` (하, 기본) | Whisper large-v3-turbo | 가장 빠름 |

### 출력

발화마다 `[HH:MM:SS] 텍스트` 한 줄이 터미널에 표시되고 동시에
`outputs/stt/YYYY-MM-DD-HHMMSS.txt`에 추가됩니다.

### 동작 방식

Whisper는 30초 윈도우 단위로 완성된 오디오를 전사하는 모델이라 글자 단위
실시간 스트리밍은 불가능합니다. 대신 VAD로 발화 경계(침묵)를 감지해, 한 발화가
끝나면 그 구간을 전사합니다. 지연은 한 문장 길이(약 1~3초)입니다.

### 한계

- 빠른 발화나 한국어/영어 혼용 구간에서 오인식·환각이 발생할 수 있습니다 (Whisper 모델 특성).

---

## English

### Features

- Real-time microphone transcription (Korean by default; Whisper supports 99 languages)
- `mlx-whisper` (`whisper-large-v3-turbo`) — Metal GPU accelerated on Apple Silicon
- silero-VAD utterance segmentation (MLX-native, no PyTorch)
- Each utterance is appended to `outputs/stt/<timestamp>.txt` immediately (minimal loss on hard kill)

### Requirements

- **Chip**: Apple Silicon (M1/M2/M3/M4/M5). MLX is Apple-Silicon only — Intel Macs are not supported.
- **OS**: macOS 13.5 (Ventura) or later (required by MLX/Metal).
- **Python**: 3.10 or later (required by MLX).
- **Memory**: inference peaks at ~2.5 GB (unified memory); 8 GB RAM is enough.
- **Disk**: ~1.5 GB for the Whisper model + ~2 MB for VAD, cached under `~/.cache/huggingface`; venv dependencies are separate.
- **Microphone**: an input device + macOS microphone permission for your terminal app.
- **Network**: internet for the first-run model download (`HF_TOKEN` recommended to avoid rate limits).

Tested on: Apple M5 Max / macOS 26 / Python 3.14 / mlx 0.31.

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
python3 stt-pipeline.py               # start real-time transcription, Ctrl+C to stop
python3 stt-pipeline.py --engine high # highest accuracy (Qwen3-ASR)
python3 stt-pipeline.py --no-save     # terminal only, no file
```

| Option | Default | Description |
|--------|---------|-------------|
| `--engine` | `low` | STT tier (`high`/`mid`/`low`) |
| `--model` | — | Override the tier's model id (keeps the tier's family) |
| `--language` | — | Override the language passed to the model |
| `--output-dir` | `outputs/stt` | Where transcripts are saved |
| `--no-save` | — | Do not save a file |

### Engine tiers (`--engine`)

An accuracy ladder. ASR accuracy saturates at small model sizes, so tiers differ
by model family/quality, not parameter count. The selected model is downloaded on
first use.

| Tier | Model | Profile |
|------|-------|---------|
| `high` | Qwen3-ASR-1.7B | 2026 SOTA, highest accuracy |
| `mid` | Whisper large-v3 (full) | proven accuracy |
| `low` (default) | Whisper large-v3-turbo | fastest |

### Output

Each utterance is printed as `[HH:MM:SS] text` and appended at the same time to
`outputs/stt/YYYY-MM-DD-HHMMSS.txt`.

### How it works

Whisper transcribes completed 30-second audio windows, so character-level
streaming is not possible. Instead, VAD detects utterance boundaries (silence)
and each utterance is transcribed once it ends. Latency is roughly one sentence
(about 1–3s).

### Limitations

- Fast speech or Korean/English code-switching can cause misrecognition or hallucination (a Whisper model characteristic).
