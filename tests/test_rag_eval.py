"""测试 eval/rag_eval.py 的评测指标（纯函数，不加载 ChromaDB/模型）。"""
from eval.rag_eval import evaluate, hit_rank, recall_at_k, reciprocal_rank


def test_hit_rank_finds_first_match():
    ranked = ["A", "B", "A"]
    assert hit_rank(ranked, {"A"}) == 1
    assert hit_rank(ranked, {"B"}) == 2
    assert hit_rank(ranked, {"C"}) is None


def test_recall_at_k():
    ranked = ["A", "B", "C", "D"]
    assert recall_at_k(ranked, {"C"}, 2) == 0
    assert recall_at_k(ranked, {"C"}, 3) == 1
    assert recall_at_k(ranked, {"D"}, 5) == 1


def test_reciprocal_rank():
    ranked = ["A", "B", "C"]
    assert reciprocal_rank(ranked, {"A"}) == 1.0
    assert reciprocal_rank(ranked, {"B"}) == 0.5
    assert reciprocal_rank(ranked, {"C"}) == 1.0 / 3
    assert reciprocal_rank(ranked, {"Z"}) == 0.0


def test_evaluate_aggregates_recall_and_mrr():
    # 用假 search_fn 模拟检索，验证指标聚合与未命中收集
    dataset = [
        {"query": "q1", "expected_source": "A"},
        {"query": "q2", "expected_source": "C"},
    ]

    def search_fn(query: str):
        return {"q1": [{"source": "A"}, {"source": "B"}], "q2": [{"source": "D"}]}[query]

    report = evaluate(dataset, search_fn, ks=(1, 5))
    assert report["total"] == 2
    assert report["recall@1"] == 0.5  # 仅 q1 在 top1 命中
    assert report["recall@5"] == 0.5  # q2 完全未命中
    assert report["mrr"] == 0.5  # q1 rank1=1.0，q2 未命中=0.0 → (1.0+0.0)/2
    assert len(report["misses"]) == 1
    assert report["misses"][0]["query"] == "q2"


def test_evaluate_supports_multi_source():
    dataset = [{"query": "q", "expected_sources": ["A", "B"]}]

    def search_fn(query: str):
        return [{"source": "B"}]

    report = evaluate(dataset, search_fn, ks=(1,))
    assert report["recall@1"] == 1.0
    assert report["mrr"] == 1.0
