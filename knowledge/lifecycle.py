"""知识生命周期治理 §5.1：元数据（有效期/owner/审批状态/版本）+ 到期下架 + 版本回溯。

- 元数据模型：owner（负责人）、status（审批状态）、valid_until（有效期）、version（版本）。
- 到期下架：status != approved 或已过期（valid_until < today）的知识在检索端自动过滤。
- 版本回溯：每次入库追加一条 JSONL 历史，可查任意知识的历次版本。

纯函数不依赖 ChromaDB / 模型，便于单测；文件 I/O 用显式路径参数隔离。
"""
import json
from datetime import date, datetime
from pathlib import Path

# 审批状态机：draft(草稿) → pending(待审) → approved(已发布) → deprecated(已下架)
STATUS_DRAFT = "draft"
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_DEPRECATED = "deprecated"
STATUS_VALUES = {STATUS_DRAFT, STATUS_PENDING, STATUS_APPROVED, STATUS_DEPRECATED}

MANIFEST_NAME = "manifest.json"
HISTORY_NAME = "ingest_history.jsonl"

# 未在 manifest.json 中声明的文档，用安全默认值（已发布、永不过期、无 owner、版本 1.0.0）
DEFAULT_META = {"owner": "", "status": STATUS_APPROVED, "valid_until": None, "version": "1.0.0", "tenant": "default"}


def parse_date(s: str | None) -> date | None:
    """ISO 日期字符串 → date；空/None/非法 → None（视为未设有效期）。"""
    if not s:
        return None
    try:
        return date.fromisoformat(str(s))
    except ValueError:
        return None


def normalize_meta(raw: dict | None) -> dict:
    """把 manifest 中的原始字段补全为规范元数据（含默认值、日期解析、状态校验）。"""
    meta = dict(DEFAULT_META)
    if not raw:
        return meta
    if raw.get("owner"):
        meta["owner"] = str(raw["owner"])
    if raw.get("version"):
        meta["version"] = str(raw["version"])
    if raw.get("status") in STATUS_VALUES:
        meta["status"] = raw["status"]
    if raw.get("valid_until"):
        meta["valid_until"] = parse_date(raw["valid_until"])
    if raw.get("tenant"):
        meta["tenant"] = str(raw["tenant"])
    return meta


def is_expired(meta: dict, today: date | None = None) -> bool:
    """是否已过期：设置了 valid_until 且早于 today。"""
    today = today or date.today()
    vu = meta.get("valid_until")
    return vu is not None and vu < today


def is_retrievable(meta: dict, today: date | None = None) -> bool:
    """是否可被检索：仅已发布（approved）且未过期。"""
    return meta.get("status") == STATUS_APPROVED and not is_expired(meta, today)


def load_manifest(docs_dir: Path) -> dict:
    """读 docs_dir/manifest.json（缺省返回空 dict → 全部走默认元数据）。"""
    path = Path(docs_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def meta_for_source(manifest: dict, source: str) -> dict:
    """取某文档的生命周期元数据（manifest 未声明的用默认值）。"""
    return normalize_meta(manifest.get(source))


def record_history(history_path: Path, source: str, meta: dict, chunk_count: int, source_id: str, mtime: str) -> None:
    """追加一条入库历史（JSONL），供版本回溯。"""
    rec = {
        "source": source,
        "source_id": source_id,
        "mtime": mtime,
        "owner": meta["owner"],
        "status": meta["status"],
        "valid_until": meta["valid_until"].isoformat() if meta["valid_until"] else None,
        "version": meta["version"],
        "chunks": chunk_count,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def list_history(history_path: Path, source: str | None = None) -> list[dict]:
    """查历史版本记录（可按 source 过滤），返回按入库时间升序的列表。"""
    if not history_path.exists():
        return []
    out = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if source is None or rec.get("source") == source:
            out.append(rec)
    return out
