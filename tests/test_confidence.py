"""测试 tools/rag_retriever.py 的置信度归一化（纯函数，不触发 RAG 重依赖）。"""
from tools.rag_retriever import compute_confidence


def test_confidence_sigmoid_positive():
    conf = compute_confidence([2.0], None)
    assert 0.8 < conf < 0.95


def test_confidence_sigmoid_negative():
    conf = compute_confidence([-2.0], None)
    assert conf < 0.2


def test_confidence_from_distance():
    assert compute_confidence(None, 0.1) == 0.9
    assert compute_confidence(None, 1.5) == 0.0
    assert compute_confidence(None, -0.5) == 1.0  # clamp 到 [0,1]


def test_confidence_fallback():
    # 无 reranker 分数、无距离（如纯 BM25 命中）→ 不误判为低置信度
    assert compute_confidence(None, None) == 1.0
