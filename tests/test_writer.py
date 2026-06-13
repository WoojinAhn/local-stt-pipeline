from writer import TranscriptWriter


def test_no_file_until_first_line(tmp_path):
    writer = TranscriptWriter(str(tmp_path))
    assert list(tmp_path.glob("*.txt")) == []
    writer.close()
    assert list(tmp_path.glob("*.txt")) == []  # empty session leaves no file


def test_lines_are_flushed_before_close(tmp_path):
    writer = TranscriptWriter(str(tmp_path))
    writer.write_line("13:00:01", "안녕하세요")

    files = list(tmp_path.glob("*.txt"))
    assert len(files) == 1
    # readable before close() -> proves each line is flushed (crash-resilient)
    assert files[0].read_text(encoding="utf-8") == "[13:00:01] 안녕하세요\n"

    writer.write_line("13:00:05", "두 번째")
    assert files[0].read_text(encoding="utf-8") == (
        "[13:00:01] 안녕하세요\n[13:00:05] 두 번째\n"
    )
    writer.close()


def test_disabled_writer_writes_nothing(tmp_path):
    writer = TranscriptWriter(str(tmp_path), enabled=False)
    writer.write_line("13:00:01", "x")
    writer.close()
    assert list(tmp_path.glob("*.txt")) == []
    assert writer.path is None
