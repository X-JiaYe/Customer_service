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
from knowledge import chunking, lifecycle
from knowledge.embedding import get_embedding_function

# 分块 / 父子块 schema 版本：变更分块逻辑时递增，强制全量重导（旧 schema 数据不兼容）
SCHEMA_VERSION = "3"

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

    # 迁移：旧 schema（分块无 tenant_id）一次性清空重建，保证多租户隔离字段齐全
    existing = collection.get(include=["metadatas"])
    if existing["ids"] and any((m or {}).get("tenant_id") is None for m in existing["metadatas"]):
        collection.delete(ids=existing["ids"])
        print("[ingest] 检测到旧 schema（缺 tenant_id），已清空，即将按多租户重导")

    files = list(_iter_docs(docs_dir))
    if not files:
        print(f"[ingest] 未在 {docs_dir} 下找到任何 .txt/.md/.pdf 文档")
        return

    # 生命周期治理 §5.1：读 manifest 元数据 + 版本回溯历史（写 knowledge/ 下，避免被当作文档扫描）
    manifest = lifecycle.load_manifest(docs_dir)
    history_path = docs_dir.parent / lifecycle.HISTORY_NAME

    total_added = 0
    for path in files:
        mtime = str(path.stat().st_mtime)
        meta = lifecycle.meta_for_source(manifest, path.name)
        # 文件名 + 修改时间 + 生命周期元数据 + schema 版本 一起作为增量指纹：
        # 内容、元数据（owner/状态/有效期/版本）或分块逻辑任一变化都触发重导
        tenant = meta["tenant"]
        lifecycle_fp = f"{meta['owner']}|{meta['status']}|{meta['valid_until']}|{meta['version']}|{tenant}"
        source_id = hashlib.md5(f"{path.name}:{mtime}:{lifecycle_fp}:{SCHEMA_VERSION}".encode()).hexdigest()[:16]

        existing = collection.get(where={"source_id": source_id}, include=[])
        if existing["ids"]:
            print(f"[ingest] 跳过（未变化）：{path.name}")
            continue

        text = _read_file(path)
        if not text.strip():
            print(f"[ingest] 跳过（空文件）：{path.name}")
            continue

        # 删除该文件旧版本的分块（按 文档名+租户 精确删，跨租户同名文档不误删）
        collection.delete(where={"$and": [{"source": path.name}, {"tenant_id": tenant}]})

        # 父子块 §5.7：child 用于检索（精准），parent 用于注入 LLM（上下文完整）
        child_chunks = split_text(text)
        parents, child_to_parent = chunking.build_parent_chunks(child_chunks, config.CHUNK_PARENT_SIZE)

        valid_until = meta["valid_until"].isoformat() if meta["valid_until"] else ""
        base_meta = {
            "source": path.name,
            "source_id": source_id,
            "mtime": mtime,
            "owner": meta["owner"],
            "status": meta["status"],
            "valid_until": valid_until,
            "version": meta["version"],
            "tenant_id": tenant,
        }

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for i, c in enumerate(child_chunks):
            ids.append(f"{source_id}:{i}")
            documents.append(c)
            metadatas.append({**base_meta, "chunk_index": i, "chunk_type": "child", "parent_id": f"{source_id}:p{child_to_parent[i]}"})
        for pi, p in enumerate(parents):
            ids.append(f"{source_id}:p{pi}")
            documents.append(p)
            metadatas.append({**base_meta, "chunk_type": "parent", "parent_id": f"{source_id}:p{pi}"})

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        total_added += len(child_chunks)
        lifecycle.record_history(history_path, path.name, meta, len(child_chunks), source_id, mtime)
        print(f"[ingest] 已导入 {path.name}：{len(child_chunks)} 子块 / {len(parents)} 父块（status={meta['status']}, v{meta['version']}）")

    print(f"[ingest] 完成，本次新增 {total_added} 个分块，集合共 {collection.count()} 条")


if __name__ == "__main__":
    ingest()
