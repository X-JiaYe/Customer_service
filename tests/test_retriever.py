"""测试 tools/rag_retriever.py 的轻量分词（纯函数，不触发 RAG 重依赖）。"""
from tools.rag_retriever import tokenize


def test_tokenize_chinese_bigrams():
    assert tokenize("你好世界") == ["你好", "好世", "世界"]


def test_tokenize_english_and_chinese():
    tokens = tokenize("ACoS 红线")
    assert "acos" in tokens
    assert "红线" in tokens


def test_tokenize_empty_falls_back():
    assert tokenize("") == [""]
