"""测试 audit.py 的审计签名（防篡改）纯函数，不依赖 Redis。"""
import config
from audit import _sign, verify_audit


def test_sign_and_verify(monkeypatch):
    monkeypatch.setattr(config, "AUDIT_HMAC_KEY", "secret-key")
    record = {"action": "chat", "tenant_id": "default"}
    sig = _sign(record)
    assert sig is not None and len(sig) == 64  # sha256 hex

    signed = {**record, "sig": sig}
    assert verify_audit(signed) is True

    # 篡改内容 → 校验失败
    tampered = {**record, "action": "query_order", "sig": sig}
    assert verify_audit(tampered) is False

    # 缺 sig → 失败
    assert verify_audit(record) is False


def test_no_key_disables_signing(monkeypatch):
    monkeypatch.setattr(config, "AUDIT_HMAC_KEY", "")
    assert _sign({"a": 1}) is None
    assert verify_audit({"a": 1}) is True
