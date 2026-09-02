"""测试 knowledge/lifecycle.py 的知识生命周期治理（纯函数，不触发 ChromaDB/模型）。"""
from datetime import date

from knowledge import lifecycle


def test_normalize_meta_defaults():
    meta = lifecycle.normalize_meta(None)
    assert meta["owner"] == ""
    assert meta["status"] == "approved"
    assert meta["valid_until"] is None
    assert meta["version"] == "1.0.0"


def test_normalize_meta_applies_fields_and_validates_status():
    meta = lifecycle.normalize_meta(
        {"owner": "张三", "status": "pending", "valid_until": "2027-12-31", "version": "2.1.0"}
    )
    assert meta["owner"] == "张三"
    assert meta["status"] == "pending"
    assert meta["valid_until"] == date(2027, 12, 31)
    assert meta["version"] == "2.1.0"
    # 非法状态回退为默认 approved，避免 typo 导致误下架
    assert lifecycle.normalize_meta({"status": "aprovd"})["status"] == "approved"


def test_parse_date_invalid_falls_back_none():
    assert lifecycle.parse_date("not-a-date") is None
    assert lifecycle.parse_date("") is None


def test_is_expired_and_retrievable():
    today = date(2026, 9, 2)
    expired = {"status": "approved", "valid_until": date(2026, 9, 1)}
    active = {"status": "approved", "valid_until": date(2026, 9, 3)}
    no_expiry = {"status": "approved", "valid_until": None}
    assert lifecycle.is_expired(expired, today) is True
    assert lifecycle.is_expired(active, today) is False
    assert lifecycle.is_expired(no_expiry, today) is False

    assert lifecycle.is_retrievable(active, today) is True
    assert lifecycle.is_retrievable(expired, today) is False
    # 非 approved 一律不可检索，即使未过期
    assert lifecycle.is_retrievable({"status": "deprecated", "valid_until": None}, today) is False
    assert lifecycle.is_retrievable({"status": "draft", "valid_until": None}, today) is False


def test_load_manifest_and_meta_for_source(tmp_path):
    (tmp_path / "manifest.json").write_text(
        '{"A.md": {"owner": "op", "status": "approved", "version": "1.0.0"}, "B.md": {"status": "deprecated"}}',
        encoding="utf-8",
    )
    manifest = lifecycle.load_manifest(tmp_path)
    assert lifecycle.meta_for_source(manifest, "A.md")["owner"] == "op"
    assert lifecycle.meta_for_source(manifest, "B.md")["status"] == "deprecated"
    # 未声明走默认
    assert lifecycle.meta_for_source(manifest, "C.md")["status"] == "approved"


def test_load_manifest_missing_returns_empty(tmp_path):
    assert lifecycle.load_manifest(tmp_path) == {}


def test_record_and_list_history(tmp_path):
    hist = tmp_path / "ingest_history.jsonl"
    meta = lifecycle.normalize_meta({"owner": "张三", "status": "approved", "version": "1.0.0"})
    lifecycle.record_history(hist, "A.md", meta, chunk_count=3, source_id="abc123", mtime="1.0")
    lifecycle.record_history(hist, "B.md", meta, chunk_count=1, source_id="def456", mtime="1.0")

    all_records = lifecycle.list_history(hist)
    assert len(all_records) == 2
    assert all_records[0]["source"] == "A.md"
    assert all_records[0]["chunks"] == 3
    assert all_records[0]["status"] == "approved"

    # 按 source 过滤（版本回溯）
    a_records = lifecycle.list_history(hist, source="A.md")
    assert len(a_records) == 1
    assert a_records[0]["source_id"] == "abc123"


def test_list_history_missing_file(tmp_path):
    assert lifecycle.list_history(tmp_path / "nope.jsonl") == []
