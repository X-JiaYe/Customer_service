"""测试 knowledge/backup.py 的快照/恢复（纯文件 I/O，不触发 ChromaDB/模型）。"""
from knowledge import backup


def test_snapshot_copies_contents(tmp_path):
    src = tmp_path / "chroma_db"
    src.mkdir()
    (src / "data.bin").write_text("payload")
    (src / "sub").mkdir()
    (src / "sub" / "meta.txt").write_text("x")

    dest = backup.snapshot(src, tmp_path / "backup")
    assert dest.exists()
    assert (dest / "data.bin").read_text() == "payload"
    assert (dest / "sub" / "meta.txt").read_text() == "x"


def test_snapshot_missing_src_raises(tmp_path):
    try:
        backup.snapshot(tmp_path / "nope", tmp_path / "backup")
        assert False, "应抛 FileNotFoundError"
    except FileNotFoundError:
        pass


def test_restore_overwrites_dst(tmp_path):
    src = tmp_path / "chroma_db"
    src.mkdir()
    (src / "data.txt").write_text("v1")
    snap = backup.snapshot(src, tmp_path / "backup")

    # 模拟数据损坏 / 误删
    (src / "data.txt").write_text("corrupted")
    (src / "junk.txt").write_text("x")

    backup.restore(snap, src)
    assert (src / "data.txt").read_text() == "v1"
    assert not (src / "junk.txt").exists()  # 恢复是覆盖，多余文件被清掉


def test_restore_missing_snapshot_raises(tmp_path):
    try:
        backup.restore(tmp_path / "nope", tmp_path / "dst")
        assert False, "应抛 FileNotFoundError"
    except FileNotFoundError:
        pass
