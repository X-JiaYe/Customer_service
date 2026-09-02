"""知识库导入：扫描 docs/ → 递归分块 → Embedding → ChromaDB（支持增量导入）。

用法：
    python -m knowledge.ingest
"""
import hashlib
import sys
from pathlib import Path

# 允许从项目根目录 import config / knowledge
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb

import config
from knowledge.embedding import get_embedding_function

# 递归切分的分隔符优先级：段落 → 换行 → 句子 → 词 → 字符
# 注意：不含空串 ""（str.split("") 会抛 ValueError）；字符级硬切由下方 sep is None 兜底完成
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", ".", "!", "?", ";", ",", " "]


def split_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> list[str]:
    """递归式分块：优先按更粗粒度分隔符切分，超过 chunk_size 才下探到更细分隔符。"""
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

    text = text.strip()
    if not text:
        return []

    def _recursive(t: str, seps: list[str]) -> list[str]:
        if len(t) <= chunk_size:
            return [t] if t else []
        sep = seps[0] if seps else None
        if sep is None:  # 兜底：硬切
            return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]

        parts = t.split(sep)
        result, buf = [], ""
        for part in parts:
            piece = buf + sep + part if buf else part
            if len(piece) <= chunk_size:
                buf = piece
            else:
                if buf:
                    result.append(buf)
                if len(part) > chunk_size:
                    result.extend(_recursive(part, seps[1:]))
                    buf = ""
                else:
                    buf = part
        if buf:
            result.append(buf)
        return result

    chunks = [c for c in _recursive(text, _SEPARATORS) if c.strip()]

    # 施加 chunk_overlap：相邻块保留上一块尾部的 overlap 字符
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            overlapped.append(overlapped[i - 1][-chunk_overlap:] + chunks[i])
        chunks = overlapped
    return chunks


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("读取 PDF 需要安装 pypdf：pip install pypdf") from e
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return ""


def _iter_docs(docs_dir: Path):
    for suffix in (".txt", ".md", ".pdf"):
        for p in sorted(docs_dir.glob(f"*{suffix}")):
            yield p


def ingest() -> None:
    docs_dir = Path(config.DOCS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        embedding_function=get_embedding_function(),
    )

    files = list(_iter_docs(docs_dir))
    if not files:
        print(f"[ingest] 未在 {docs_dir} 下找到任何 .txt/.md/.pdf 文档")
        return

    total_added = 0
    for path in files:
        mtime = str(path.stat().st_mtime)
        # 文件名 + 修改时间 作为增量判断依据
        source_id = hashlib.md5(f"{path.name}:{mtime}".encode()).hexdigest()[:16]

        existing = collection.get(where={"source_id": source_id}, include=[])
        if existing["ids"]:
            print(f"[ingest] 跳过（未变化）：{path.name}")
            continue

        text = _read_file(path)
        if not text.strip():
            print(f"[ingest] 跳过（空文件）：{path.name}")
            continue

        # 删除该文件旧版本的分块，避免重复
        collection.delete(where={"source": path.name})

        chunks = split_text(text)
        ids = [f"{source_id}:{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": path.name, "source_id": source_id, "mtime": mtime, "chunk_index": i}
            for i in range(len(chunks))
        ]
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        total_added += len(chunks)
        print(f"[ingest] 已导入 {path.name}：{len(chunks)} 个分块")

    print(f"[ingest] 完成，本次新增 {total_added} 个分块，集合共 {collection.count()} 条")


if __name__ == "__main__":
    ingest()
