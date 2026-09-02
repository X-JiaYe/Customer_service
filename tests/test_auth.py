"""测试 auth.py 的鉴权：API Key 解析 + 短期 token 签发/解析（内存 fake redis）。"""
import auth
import config


class _FakeRedis:
    def __init__(self):
        self.d = {}

    def get(self, key):
        return self.d.get(key)

    def set(self, key, val, ex=None):
        self.d[key] = val


def test_resolve_tenant(monkeypatch):
    monkeypatch.setattr(config, "API_KEYS", {"k1": "t1", "k2": "t2"})
    assert auth.resolve_tenant("k1") == "t1"
    assert auth.resolve_tenant("k2") == "t2"
    assert auth.resolve_tenant("nope") is None
    assert auth.resolve_tenant(None) is None
    assert auth.resolve_tenant("") is None


def test_issue_and_resolve_token(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(auth, "get_redis", lambda: r)
    token = auth.issue_token("t1")
    assert auth.resolve_token(token) == "t1"
    assert auth.resolve_token("bogus") is None
    assert auth.resolve_token(None) is None
    assert auth.resolve_token("") is None


def test_tokens_are_unique(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(auth, "get_redis", lambda: r)
    assert auth.issue_token("t1") != auth.issue_token("t1")
