"""知识库备份 / 恢复 / 重建 §5.8：向量库快照 + 从 docs 重建的容灾路径。

- snapshot：把 ChromaDB 目录快照到 backup/ 下的时间戳目录。
- restore：用快照覆盖当前 ChromaDB（回滚到历史版本）。
- rebuild：清空 ChromaDB 后从 docs 全量重导（灾难恢复最终兜底，docs 是「源」）。
- verify：检查集合存在且条数 > 0（恢复后自检）。

用法：
    python -m knowledge.backup --backup
    python -m knowledge.backup --restore backup/chroma-20260902T120000
    python -m knowledge.backup --rebuild
    python -m knowledge.backup --verify

snapshot / restore 为纯文件 I/O，便于单测（不触发 ChromaDB / 模型）。
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

BACKUP_DIR = Path(config.BASE_DIR) / "backup"


def _timestamp() -> str:
    # 含微秒，避免同一秒内多次快照同名冲突
    return datetime.now().strftime("%Y%m%dT%H%M%S%f")


def snapshot(src_dir: Path | str, backup_dir: Path | str = BACKUP_DIR) -> Path:
    """把 src_dir 整体快照到 backup_dir/chroma-<timestamp>/，返回快照路径。"""
    src = Path(src_dir)
    if not src.exists():
        raise FileNotFoundError(f"ChromaDB 目录不存在：{src}")
    dest = Path(backup_dir) / f"chroma-{_timestamp()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)  # dest 不存在 → copytree 将其创建为 src 的完整副本
    return dest


def restore(snapshot_path: Path | str, dst_dir: Path | str) -> None:
    """用快照覆盖 dst_dir（dst 存在则先清空），实现回滚。"""
    snap = Path(snapshot_path)
    dst = Path(dst_dir)
    if not snap.exists():
        raise FileNotFoundError(f"快照不存在：{snap}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(snap, dst)


def rebuild(chroma_dir: Path | str = config.CHROMA_PERSIST_DIR) -> int:
    """清空 ChromaDB 后从 docs 全量重导，返回最终集合条数。"""
    from knowledge.ingest import ingest  # 懒加载，避免未安装依赖时 import 失败

    chroma_dir = Path(chroma_dir)
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    ingest()
    return verify()


def verify(chroma_dir: Path | str = config.CHROMA_PERSIST_DIR) -> int:
    """返回集合条数；异常抛错（供恢复后自检）。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_or_create_collection(name=config.CHROMA_COLLECTION).count()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="知识库备份/恢复/重建（§5.8 容灾）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--backup", action="store_true", help="快照当前 ChromaDB 到 backup/")
    group.add_argument("--restore", metavar="SNAPSHOT", help="用快照覆盖当前 ChromaDB")
    group.add_argument("--rebuild", action="store_true", help="清空并从 docs 全量重导")
    group.add_argument("--verify", action="store_true", help="检查集合条数")
    args = parser.parse_args(argv)

    if args.backup:
        dest = snapshot(config.CHROMA_PERSIST_DIR, BACKUP_DIR)
        print(f"[backup] 快照完成：{dest}")
    elif args.restore:
        restore(args.restore, config.CHROMA_PERSIST_DIR)
        print(f"[backup] 已从 {args.restore} 恢复，集合条数 = {verify()}")
    elif args.rebuild:
        print(f"[backup] 重建完成，集合条数 = {rebuild()}")
    elif args.verify:
        print(f"[backup] 集合条数 = {verify()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
