"""测试 tools/rag_retriever.py 的轻量分词（纯函数，不触发 RAG 重依赖）。"""
from tools.rag_retriever import tokenize


def test_tokenize_chinese_bigrams():
    assert tokenize("你好世界") == ["你好", "好世", "世界"]


def test_tokenize_english_and_chinese():
    tokens = tokenize("GW200 网关")
    assert "gw200" in tokens
    assert "网关" in tokens


def test_tokenize_empty_falls_back():
    assert tokenize("") == [""]


def test_tenant_children_filters_by_tenant():
    """§4.1 多租户隔离：BM25 召回只返回当前租户的 child，缺 tenant 标签回落 default。"""
    from tools.rag_retriever import RagRetrieverTool

    tool = object.__new__(RagRetrieverTool)  # 绕过 __init__，不加载 ChromaDB/embedding
    tool._child_ids = ["a0", "a1", "b0", "c0"]
    tool._child_docs = ["A 文档一", "A 文档二", "B 文档", "C 文档"]
    tool._meta_by_id = {
        "a0": {"tenant_id": "tenant_a"},
        "a1": {"tenant_id": "tenant_a"},
        "b0": {"tenant_id": "tenant_b"},
        "c0": {},  # 缺 tenant 标签 → 回落 default
    }
    tool._bm25_cache = {}

    ids_a, docs_a, bm25_a = tool._tenant_children("tenant_a")
    assert ids_a == ["a0", "a1"]
    assert docs_a == ["A 文档一", "A 文档二"]
    assert bm25_a is not None

    assert tool._tenant_children("tenant_b")[0] == ["b0"]
    assert tool._tenant_children("default")[0] == ["c0"]

    # 无匹配租户 → 空结果 + bm25 None
    ids_empty, _, bm25_empty = tool._tenant_children("tenant_x")
    assert ids_empty == []
    assert bm25_empty is None

    # 缓存生效：同租户二次调用返回同一对象
    assert tool._tenant_children("tenant_a")[0] is ids_a
