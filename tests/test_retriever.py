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
    """§4.1 多租户隔离：召回 = 本租户专属 + shared 共享，隔离其他租户；缺标签回落 shared。"""
    from tools.rag_retriever import RagRetrieverTool

    tool = object.__new__(RagRetrieverTool)  # 绕过 __init__，不加载 ChromaDB/embedding
    tool._child_ids = ["a0", "a1", "b0", "c0", "d0"]
    tool._child_docs = ["A 文档一", "A 文档二", "B 文档", "共享文档", "显式共享"]
    tool._meta_by_id = {
        "a0": {"tenant_id": "tenant_a"},
        "a1": {"tenant_id": "tenant_a"},
        "b0": {"tenant_id": "tenant_b"},
        "c0": {},  # 缺 tenant 标签 → 回落 shared
        "d0": {"tenant_id": "shared"},
    }
    tool._bm25_cache = {}

    # 本租户专属 + shared
    assert tool._tenant_children("tenant_a")[0] == ["a0", "a1", "c0", "d0"]
    assert tool._tenant_children("tenant_b")[0] == ["b0", "c0", "d0"]

    # 无专属文档的租户只看到 shared
    assert tool._tenant_children("tenant_x")[0] == ["c0", "d0"]

    # 缓存生效：同租户二次调用返回同一对象
    ids_a, _, bm25_a = tool._tenant_children("tenant_a")
    assert tool._tenant_children("tenant_a")[0] is ids_a
    assert bm25_a is not None
