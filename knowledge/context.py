"""上下文组装 §5.7：父子块扩展 + 近重复去重 + 按预算裁剪（替换「简单拼接 + 硬截断」）。

纯函数，不依赖 ChromaDB / 模型，便于单测。

- assemble_context(ranked, max_chars)：把 search() 的 chunk 级结果组装成注入 LLM 的文本。
- dedupe(items, threshold)：按字符 bigram Jaccard 相似度去除近重复 chunk。
- budgeted_truncate(texts, max_chars)：按排名顺序贪婪填充预算，句子边界截断，避免硬切半句。
"""
import re

_SENT_BOUNDARY = re.compile(r"[。！？；.!?;\n]")
_WS = re.compile(r"\s+")


def _shingles(text: str) -> set[str]:
    """字符 bigram 集合（用于近重复判断）。"""
    t = _WS.sub("", text)
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe(items: list[dict], threshold: float = 0.85) -> list[dict]:
    """按顺序去除与已保留项「近重复」的 chunk（items 须已按相关性排序）。"""
    kept: list[dict] = []
    kept_shingles: list[set[str]] = []
    for it in items:
        s = _shingles(it["text"])
        if any(_jaccard(s, ks) >= threshold for ks in kept_shingles):
            continue
        kept.append(it)
        kept_shingles.append(s)
    return kept


def truncate_to_sentence(text: str, limit: int) -> str:
    """把 text 截到 limit 内的最后一个句子边界；若放不下完整一句则返回空串（不切半句）。"""
    if len(text) <= limit:
        return text
    prefix = text[:limit]
    matches = list(_SENT_BOUNDARY.finditer(prefix))
    return prefix[: matches[-1].end()] if matches else ""


def budgeted_truncate(texts: list[str], max_chars: int) -> list[str]:
    """按排名顺序贪婪填充预算：靠前的 chunk 完整保留，预算耗尽时在句子边界截断后丢弃后续。

    - 从不超 max_chars；后续 chunk 只放「至少一句完整句子」，不产生碎片。
    - 首名优先：top-1 若首句超预算，硬切兜底保留其前缀，避免首条被整段丢弃。
    """
    out: list[str] = []
    remaining = max_chars
    for i, t in enumerate(texts):
        if remaining <= 0:
            break
        if len(t) <= remaining:
            out.append(t)
            remaining -= len(t)
            continue
        piece = truncate_to_sentence(t, remaining)
        if piece:
            out.append(piece)
        elif i == 0:
            out.append(t[:remaining])  # top-1 硬切兜底
        break  # 预算耗尽，后续丢弃
    return out


def assemble_context(ranked: list[dict], max_chars: int, dedupe_threshold: float = 0.85) -> str:
    """把 search() 结果组装为注入 LLM 的文本（含【来源】标注）。

    ranked 每项字段：document / source_label / source / parent_id / parent_text（均可缺省）。
    流程：父子块扩展（同一 parent 的多个 child 合并）→ 近重复去重 → 预算裁剪 → 渲染。
    """
    # 1) 父子块扩展：有 parent_text 时用 parent 完整上下文替代 child（按 parent_id 去重）
    expanded: list[dict] = []
    seen_parents: set[str] = set()
    for r in ranked:
        pid = r.get("parent_id")
        ptext = r.get("parent_text")
        label = r.get("source_label") or r.get("source") or "未知文档"
        if ptext and pid:
            if pid in seen_parents:
                continue
            seen_parents.add(pid)
            expanded.append({"text": ptext, "label": label, "source": r.get("source") or ""})
        else:
            expanded.append({"text": r.get("document") or "", "label": label, "source": r.get("source") or ""})

    # 2) 近重复去重
    kept = dedupe(expanded, dedupe_threshold)

    # 3) 按预算裁剪
    texts = [k["text"] for k in kept]
    budgeted = budgeted_truncate(texts, max_chars)

    # 4) 渲染（budgeted 与 kept 前缀对齐）
    parts = [f"【来源：{k['label']}】\n{t}" for k, t in zip(kept, budgeted)]
    return "\n\n".join(parts)
