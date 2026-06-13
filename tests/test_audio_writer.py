import wave

import numpy as np

from audio_writer import AudioWriter

SR = 16000


def test_no_file_until_first_chunk(tmp_path):
    w = AudioWriter(str(tmp_path), SR)
    assert list(tmp_path.glob("*.wav")) == []
    w.close()
    assert list(tmp_path.glob("*.wav")) == []  # empty session leaves no file


def test_writes_valid_wav_with_expected_frames(tmp_path):
    w = AudioWriter(str(tmp_path), SR, stamp="2026-06-13-120000")
    w.write(np.zeros(1600, dtype=np.float32))   # 100 ms
    w.write(np.ones(800, dtype=np.float32))     # 50 ms (clipped to full-scale)
    w.close()

    path = tmp_path / "2026-06-13-120000.wav"
    assert path.exists()
    with wave.open(str(path), "rb") as r:
        assert r.getnchannels() == 1
        assert r.getsampwidth() == 2
        assert r.getframerate() == SR
        assert r.getnframes() == 2400  # 1600 + 800


def test_disabled_writes_nothing(tmp_path):
    w = AudioWriter(str(tmp_path), SR, enabled=False)
    w.write(np.ones(1600, dtype=np.float32))
    w.close()
    assert list(tmp_path.glob("*.wav")) == []
    assert w.path is None
