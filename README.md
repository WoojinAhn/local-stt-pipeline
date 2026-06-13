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

- Apple Silicon Mac (M1 이상), macOS
- Python 3.10+
- 터미널의 마이크 접근 권한

### 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 사용

```bash
python3 stt-pipeline.py              # 실시간 받아쓰기 시작, Ctrl+C로 종료
python3 stt-pipeline.py --no-save    # 파일 저장 없이 터미널만
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--language` | `ko` | 언어 코드 |
| `--model` | `mlx-community/whisper-large-v3-turbo` | STT 모델 ID |
| `--output-dir` | `outputs/stt` | 전사 파일 저장 위치 |
| `--no-save` | — | 파일 저장 안 함 |

### 출력

발화마다 `[HH:MM:SS] 텍스트` 한 줄이 터미널에 표시되고 동시에
`outputs/stt/YYYY-MM-DD-HHMMSS.txt`에 추가됩니다.

### 동작 방식

Whisper는 30초 윈도우 단위로 완성된 오디오를 전사하는 모델이라 글자 단위
실시간 스트리밍은 불가능합니다. 대신 VAD로 발화 경계(침묵)를 감지해, 한 발화가
끝나면 그 구간을 전사합니다. 지연은 한 문장 길이(약 1~3초)입니다.

### 한계

- 빠른 발화나 한국어/영어 혼용 구간에서 오인식·환각이 발생할 수 있습니다.
- 첫 실행 시 Whisper/VAD 모델을 HuggingFace에서 내려받습니다 (`HF_TOKEN` 권장).

---

## English

### Features

- Real-time microphone transcription (Korean by default; Whisper supports 99 languages)
- `mlx-whisper` (`whisper-large-v3-turbo`) — Metal GPU accelerated on Apple Silicon
- silero-VAD utterance segmentation (MLX-native, no PyTorch)
- Each utterance is appended to `outputs/stt/<timestamp>.txt` immediately (minimal loss on hard kill)

### Requirements

- Apple Silicon Mac (M1 or later), macOS
- Python 3.10+
- Microphone permission for your terminal

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
python3 stt-pipeline.py              # start real-time transcription, Ctrl+C to stop
python3 stt-pipeline.py --no-save    # terminal only, no file
```

| Option | Default | Description |
|--------|---------|-------------|
| `--language` | `ko` | Language code |
| `--model` | `mlx-community/whisper-large-v3-turbo` | STT model id |
| `--output-dir` | `outputs/stt` | Where transcripts are saved |
| `--no-save` | — | Do not save a file |

### Output

Each utterance is printed as `[HH:MM:SS] text` and appended at the same time to
`outputs/stt/YYYY-MM-DD-HHMMSS.txt`.

### How it works

Whisper transcribes completed 30-second audio windows, so character-level
streaming is not possible. Instead, VAD detects utterance boundaries (silence)
and each utterance is transcribed once it ends. Latency is roughly one sentence
(about 1–3s).

### Limitations

- Fast speech or Korean/English code-switching can cause misrecognition or hallucination.
- The first run downloads the Whisper/VAD models from HuggingFace (`HF_TOKEN` recommended).
