"""RAG 评测闭环 §5.4：用种子数据集离线评测检索质量（Recall@K / MRR）。

不依赖模型推理：直接调用 RagRetrieverTool.search() 取 chunk 级排序结果，
与种子数据集中的 expected_source（文档名）比对。

用法：
    python -m eval.rag_eval                          # 用默认 eval/seed_dataset.json
    python -m eval.rag_eval --dataset path.json      # 指定数据集
    python -m eval.rag_eval --k 5 10                 # 指定 Recall@K 的 K 值

输出示例：
    recall@5=0.92  recall@10=1.00  mrr=0.97
    未命中 1/13（列出 query → expected → 实际 top5 来源）
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_DATASET = Path(__file__).resolve().parent / "seed_dataset.json"


def _expected_sources(item: dict) -> set[str]:
    """兼容 expected_source（单文档）与 expected_sources（多文档）两种标注。"""
    if "expected_sources" in item:
        return set(item["expected_sources"])
    return {item["expected_source"]}


def hit_rank(ranked_sources: list[str], expected: set[str]) -> int | None:
    """返回首个命中 expected 的排名（1 起），未命中返回 None。"""
    for rank, src in enumerate(ranked_sources, start=1):
        if src in expected:
            return rank
    return None


def recall_at_k(ranked_sources: list[str], expected: set[str], k: int) -> int:
    """单条样本：top-K 内是否命中（0/1），供平均得到 Recall@K。"""
    return int(hit_rank(ranked_sources[:k], expected) is not None)


def reciprocal_rank(ranked_sources: list[str], expected: set[str]) -> float:
    """单条样本的倒数排名（MRR 的分子）。"""
    rank = hit_rank(ranked_sources, expected)
    return 1.0 / rank if rank else 0.0


def evaluate(dataset: list[dict], search_fn, ks: tuple[int, ...] = (5, 10)) -> dict:
    """对数据集跑检索，聚合 Recall@K / MRR，并收集未命中样本。

    search_fn(query) -> list[dict]，每项含 "source" 字段（文档名）。
    """
    recall_sums = {k: 0.0 for k in ks}
    rr_sum = 0.0
    misses: list[dict] = []
    for item in dataset:
        expected = _expected_sources(item)
        results = search_fn(item["query"]) or []
        ranked = [r.get("source") or "" for r in results]
        for k in ks:
            recall_sums[k] += recall_at_k(ranked, expected, k)
        rr_sum += reciprocal_rank(ranked, expected)
        if hit_rank(ranked, expected) is None:
            misses.append({"query": item["query"], "expected": sorted(expected), "got": ranked[:5]})

    n = len(dataset) or 1
    report = {f"recall@{k}": round(recall_sums[k] / n, 4) for k in ks}
    report["mrr"] = round(rr_sum / n, 4)
    report["total"] = len(dataset)
    report["misses"] = misses
    return report


def load_dataset(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # 兼容 {"queries": [...]} 与顶层 list 两种写法
    return data["queries"] if isinstance(data, dict) and "queries" in data else data


def _build_search_fn():
    from tools.rag_retriever import RagRetrieverTool

    tool = RagRetrieverTool()
    return tool.search


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK，强制 UTF-8，避免中文输出乱码（与 config.py 保持一致）
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="RAG 检索质量离线评测（Recall@K / MRR）")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="种子数据集 JSON 路径")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10], help="Recall@K 的 K 值列表")
    args = parser.parse_args(argv)

    dataset = load_dataset(args.dataset)
    print(f"[eval] 数据集 {args.dataset.name}：{len(dataset)} 条")
    print("[eval] 加载知识库 + 检索器（含 embedding/reranker，首次较慢）…")
    search_fn = _build_search_fn()

    report = evaluate(dataset, search_fn, ks=tuple(args.k))
    print("[eval] " + "  ".join(f"{k}={v}" for k, v in report.items() if k.startswith("recall") or k == "mrr"))
    if report["misses"]:
        print(f"[eval] 未命中 {len(report['misses'])}/{report['total']}：")
        for m in report["misses"]:
            print(f"  - 问：{m['query']}\n    期望：{', '.join(m['expected'])}\n    实际：{', '.join(m['got']) or '(空)'}")
    else:
        print("[eval] 全部命中")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
