"""共享的 Embedding 函数：保证 ChromaDB 导入与检索两端用同一套向量。"""
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer


class BGEEmbeddingFunction(EmbeddingFunction):
    """基于 sentence-transformers 的 BGE 中文 Embedding，返回归一化向量。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), normalize_embeddings=True).tolist()

    def name(self) -> str:
        return self.model_name.replace("/", "-")


_embedding_fn: BGEEmbeddingFunction | None = None


def get_embedding_function() -> BGEEmbeddingFunction:
    """返回单例 Embedding 函数，避免 ingest 与 retriever 各自加载一遍模型。"""
    global _embedding_fn
    if _embedding_fn is None:
        import config  # 延迟导入，避免循环依赖

        _embedding_fn = BGEEmbeddingFunction(config.EMBEDDING_MODEL)
    return _embedding_fn
