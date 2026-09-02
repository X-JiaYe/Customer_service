"""测试 knowledge/ingest.py 的分块逻辑（纯函数，不触发 embedding / ChromaDB）。"""
from knowledge.ingest import split_text


def test_split_short_text():
    assert split_text("短文本", chunk_size=500) == ["短文本"]


def test_split_empty():
    assert split_text("   ", chunk_size=500) == []


def test_split_long_text_no_overlap():
    chunks = split_text("a" * 1200, chunk_size=500, chunk_overlap=0)
    assert len(chunks) == 3
    assert all(len(c) <= 500 for c in chunks)
    assert sum(len(c) for c in chunks) == 1200  # 无重叠时不丢字


def test_split_long_text_with_overlap():
    chunks = split_text("a" * 1200, chunk_size=500, chunk_overlap=50)
    # 相邻块共享尾部 overlap，故总量 > 原文
    assert chunks[1].startswith(chunks[0][-50:])
