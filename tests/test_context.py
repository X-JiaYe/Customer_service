"""测试 knowledge/context.py 的上下文组装（纯函数，不触发 ChromaDB/模型）。"""
from knowledge import context


def test_truncate_to_sentence_cuts_at_boundary():
    text = "第一句。第二句。第三句。"
    piece = context.truncate_to_sentence(text, 5)
    assert piece == "第一句。"
    assert len(piece) <= 5


def test_truncate_short_text_untouched():
    assert context.truncate_to_sentence("短文本", 10) == "短文本"


def test_budgeted_truncate_keeps_all_when_fits():
    texts = ["aaa", "bbb"]
    assert context.budgeted_truncate(texts, 100) == ["aaa", "bbb"]


def test_budgeted_truncate_drops_tail_when_exhausted():
    texts = ["第一句。", "第二句。"]
    out = context.budgeted_truncate(texts, 6)  # 只够第一句 + 第二句部分
    assert out == ["第一句。"]
    assert sum(len(t) for t in out) <= 6


def test_budgeted_truncate_never_exceeds_budget():
    texts = ["a" * 100, "b" * 100, "c" * 100]
    out = context.budgeted_truncate(texts, 150)
    assert sum(len(t) for t in out) <= 150
    assert len(out) >= 1  # top-1 至少保留


def test_dedupe_removes_near_duplicates():
    items = [
        {"text": "这是完全相同的两段内容，用于测试去重。", "label": "A"},
        {"text": "这是完全相同的两段内容，用于测试去重。", "label": "A"},
        {"text": "完全不同的一段内容。", "label": "B"},
    ]
    kept = context.dedupe(items, threshold=0.85)
    assert len(kept) == 2
    assert kept[0]["label"] == "A"
    assert kept[1]["label"] == "B"


def test_assemble_context_expands_parent_and_labels():
    ranked = [
        {
            "document": "child片段",
            "source": "doc.md",
            "source_label": "doc.md 第0段",
            "parent_id": "p0",
            "parent_text": "完整的父块上下文内容。",
        },
        {
            "document": "child片段2",
            "source": "doc.md",
            "source_label": "doc.md 第1段",
            "parent_id": "p0",  # 同一父块，应被去重合并
            "parent_text": "完整的父块上下文内容。",
        },
    ]
    out = context.assemble_context(ranked, max_chars=1000)
    assert "完整的父块上下文内容" in out
    assert "【来源：doc.md 第0段】" in out
    # 同一 parent 只出现一次
    assert out.count("完整的父块上下文内容") == 1


def test_assemble_context_falls_back_to_child_without_parent():
    ranked = [{"document": "普通片段", "source": "doc.md", "source_label": "doc.md 第0段"}]
    out = context.assemble_context(ranked, max_chars=1000)
    assert "【来源：doc.md 第0段】" in out
    assert "普通片段" in out
