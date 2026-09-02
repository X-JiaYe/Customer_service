"""混合检索工具：BM25（词法）+ ChromaDB（语义）→ 合并去重 → Reranker 重排。

重依赖（chromadb / rank_bm25 / sentence-transformers / FlagEmbedding）采用懒加载：
未安装时工具仍可注册，被调用时优雅降级返回提示，不阻塞 Agent 启动。
"""
import math
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smolagents import Tool

import config
from audit import current_context
from knowledge import context, lifecycle
from metrics import observe_knowledge_hit, observe_knowledge_miss


def tokenize(text: str) -> list[str]:
    """轻量分词：英文/数字整词 + 中文二字组，供 BM25 词法检索使用。"""
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9_]+", text)
    chinese = re.findall(r"[一-鿿]", text)
    tokens += [chinese[i] + chinese[i + 1] for i in range(len(chinese) - 1)]
    return tokens or [text]


def compute_confidence(scores: list[float] | None, top_distance: float | None) -> float:
    """把检索置信度归一化到 0..1（供低置信度判定与单测）。

    - 有 reranker 分数：首名 logit 走 sigmoid（0.5 为中性）。
    - 无 reranker：用向量距离，1 - distance（归一化向量的距离近似）。
    - 都缺失（如纯 BM25 命中）：返回 1.0，不误判为低置信度。
    """
    if scores:
        return 1.0 / (1.0 + math.exp(-scores[0]))
    if top_distance is not None:
        return max(0.0, min(1.0, 1.0 - top_distance))
    return 1.0


class RagRetrieverTool(Tool):
    name = "knowledge_retriever"
    description = (
        "从产品知识库中检索相关文档，用于回答产品功能、使用方法、常见问题等知识类问题。"
        "输入 query 为用户的问题，返回最相关的文档片段。"
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "用户的检索问题，一句话描述想查询的内容",
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._available = False
        self._reason = ""
        self._meta_by_id = {}  # chunk_id -> metadata（source/chunk_index，用于来源标注）

        try:
            import chromadb
            from rank_bm25 import BM25Okapi

            from knowledge.embedding import get_embedding_function
        except ImportError as e:
            self._reason = f"缺少 RAG 依赖（{getattr(e, 'name', str(e))}），请安装 chromadb / rank-bm25 / sentence-transformers 后重试"
            return

        try:
            client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
            self.collection = client.get_or_create_collection(
                name=config.CHROMA_COLLECTION,
                embedding_function=get_embedding_function(),
            )
            data = self.collection.get(include=["documents", "metadatas"])
            docs = data["documents"] or []
            ids = data["ids"] or []
            metas = data["metadatas"] or []
            self._meta_by_id = dict(zip(ids, metas)) if metas else {}

            # 父子块 §5.7：分离 child（检索）与 parent（注入 LLM 的完整上下文）
            self._child_ids, self._child_docs = [], []
            self._parent_by_id = {}
            for cid, doc, meta in zip(ids, docs, metas):
                if (meta or {}).get("chunk_type") == "parent":
                    self._parent_by_id[cid] = doc
                else:
                    self._child_ids.append(cid)
                    self._child_docs.append(doc)

            # 检索/BM25 只针对 child（精准）；parent 仅用于上下文扩展
            self._docs = self._child_docs
            self._ids = self._child_ids
            self._bm25 = BM25Okapi([tokenize(d) for d in self._child_docs]) if self._child_docs else None
            # 多租户隔离：按 tenant 懒构建 BM25（key=tenant → (ids, docs, bm25)）
            self._bm25_cache: dict = {}

            # Reranker 懒加载（首次检索时才下载模型，见 _ensure_reranker）
            self._reranker = None
            self._reranker_loaded = False

            self._available = True
        except Exception as e:  # noqa: BLE001
            self._reason = f"知识库初始化失败：{e}"
            self._docs = []
            self._ids = []
            self._meta_by_id = {}
            self._parent_by_id = {}
            self._bm25 = None
            self._bm25_cache = {}
            self._reranker = None

    def _ensure_reranker(self) -> bool:
        """懒加载 Reranker 模型：首次调用才下载/加载，失败则降级为不重排。"""
        if self._reranker_loaded:
            return self._reranker is not None
        self._reranker_loaded = True
        try:
            from FlagEmbedding import FlagReranker

            self._reranker = FlagReranker(config.RERANKER_MODEL, use_fp16=True)
        except Exception as e:  # noqa: BLE001
            print(f"[RAG] Reranker 加载失败，跳过重排：{e}")
            self._reranker = None
        return self._reranker is not None

    def _source_label(self, cid: str) -> str:
        """从 chunk metadata 生成来源标注（文档名 + 分块序号）。"""
        meta = self._meta_by_id.get(cid) or {}
        src = meta.get("source") or "未知文档"
        idx = meta.get("chunk_index")
        return f"{src} 第{idx}段" if idx is not None else src

    def _tenant(self) -> str:
        """当前请求租户（供指标打标），无上下文时回落 default。"""
        return current_context().get("tenant_id") or "default"

    def _tenant_children(self, tenant: str) -> tuple[list, list, object]:
        """按租户返回 child 的 (ids, docs, bm25)，懒构建并缓存。

        多租户隔离关键点：BM25 索引按 tenant 拆分，检索只在本租户范围内召回，
        从根上杜绝跨租户串扰（§4.1 租户级隔离）。
        """
        from rank_bm25 import BM25Okapi

        cached = self._bm25_cache.get(tenant)
        if cached is not None:
            return cached
        ids, docs = [], []
        for cid, doc in zip(self._child_ids, self._child_docs):
            if (self._meta_by_id.get(cid) or {}).get("tenant_id", "default") == tenant:
                ids.append(cid)
                docs.append(doc)
        bm25 = BM25Okapi([tokenize(d) for d in docs]) if docs else None
        self._bm25_cache[tenant] = (ids, docs, bm25)
        return self._bm25_cache[tenant]

    def search(self, query: str) -> list[dict]:
        """纯检索：返回按相关性降序的 chunk 结果，不渲染文本、不打指标。

        每项字段：id / document / source / chunk_index / score / distance / parent_id / parent_text / source_label
        - score：reranker 分数（启用且重排成功时），否则 None
        - distance：ChromaDB 向量距离（命中向量检索时），否则 None
        - parent_id / parent_text：父子块扩展（§5.7），供 assemble_context 注入完整上下文

        供 forward() 渲染与 eval/rag_eval.py 评测（Recall@K / MRR）复用。
        不可用或未导入时返回 []。
        """
        if not self._available or not self._docs:
            return []

        top_k = config.RETRIEVE_TOP_K
        candidates = {}  # chunk_id -> text
        distances = {}   # chunk_id -> 向量距离（仅无 reranker 时用于置信度）

        # 多租户隔离：只在本租户范围内召回（BM25 与向量都按 tenant 过滤）
        tenant = self._tenant()
        t_ids, t_docs, t_bm25 = self._tenant_children(tenant)

        # 1) BM25 词法检索（仅当前租户的 child）
        if t_bm25:
            scores = t_bm25.get_scores(tokenize(query))
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            for i in ranked:
                candidates[t_ids[i]] = t_docs[i]

        # 2) 向量语义检索（仅 child chunk，且按 tenant 隔离）
        try:
            res = self.collection.query(
                query_texts=[query], n_results=top_k, include=["documents", "distances"],
                where={"chunk_type": "child", "tenant_id": tenant},
            )
            for cid, doc, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
                candidates.setdefault(cid, doc)
                distances[cid] = dist
        except Exception:
            pass

        if not candidates:
            return []

        # 生命周期治理 §5.1：剔除未发布（draft/pending/deprecated）或已过期的知识
        today = date.today()
        candidates = {
            cid: doc
            for cid, doc in candidates.items()
            if lifecycle.is_retrievable(lifecycle.normalize_meta(self._meta_by_id.get(cid) or {}), today)
        }
        if not candidates:
            return []

        ids = list(candidates.keys())
        docs = [candidates[i] for i in ids]
        reranker_scores = [None] * len(docs)

        # 3) Reranker 重排（懒加载，可通过 ENABLE_RERANKER 关闭）
        if config.ENABLE_RERANKER and len(docs) > 1 and self._ensure_reranker():
            try:
                pairs = [[query, d] for d in docs]
                scores = self._reranker.compute_score(pairs)
                if not isinstance(scores, list):  # 单对时返回标量
                    scores = [scores]
                order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
                ids = [ids[i] for i in order]
                docs = [docs[i] for i in order]
                reranker_scores = [scores[i] for i in order]
            except Exception:
                reranker_scores = [None] * len(ids)

        results = []
        for cid, doc, sc in zip(ids, docs, reranker_scores):
            meta = self._meta_by_id.get(cid) or {}
            parent_id = meta.get("parent_id")
            results.append(
                {
                    "id": cid,
                    "document": doc,
                    "source": meta.get("source") or "未知文档",
                    "chunk_index": meta.get("chunk_index"),
                    "score": sc,
                    "distance": distances.get(cid),
                    "parent_id": parent_id,
                    "parent_text": self._parent_by_id.get(parent_id),
                    "source_label": self._source_label(cid),
                }
            )
        return results

    def forward(self, query: str) -> str:
        if not self._available:
            observe_knowledge_miss(self._tenant())
            return f"知识库检索功能未启用。{self._reason}"

        if not self._docs:
            observe_knowledge_miss(self._tenant())
            return "知识库尚未导入文档，请先运行 python -m knowledge.ingest 导入。"

        results = self.search(query)
        if not results:
            observe_knowledge_miss(self._tenant())
            return "未在知识库中检索到相关内容。"

        # 置信度：默认门槛 0.0（关闭）；开启后低于门槛 → 低置信度，建议转人工
        scores = [r["score"] for r in results if r["score"] is not None]
        top_distance = results[0].get("distance")
        confidence = compute_confidence(scores or None, top_distance)
        low_confidence = confidence < config.RAG_CONFIDENCE_THRESHOLD

        # 上下文组装 §5.7：父子块扩展 + 近重复去重 + 预算裁剪（替换硬截断）
        top = results[: config.RERANK_TOP_N]
        result = context.assemble_context(top, config.MAX_CONTEXT_CHARS)

        if low_confidence:
            result = "【低置信度】以下内容与问题可能不匹配，仅供参考，建议转人工确认。\n\n" + result

        observe_knowledge_hit(self._tenant())
        return result
