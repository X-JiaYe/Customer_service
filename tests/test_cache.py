"""测试 cache.py 的语义缓存（纯函数 + 内存 fake redis，不触发网络）。"""
import cache
import config


def test_normalize_strips_punct_and_case():
    assert cache.normalize("你们的产品支持哪些支付方式？") == cache.normalize("你们的产品支持哪些支付方式")
    assert cache.normalize("  Hello, World! ") == "helloworld"
    assert cache.normalize("支持哪些付款方式？") == "支持哪些付款方式"
    assert cache.normalize("") == ""


def test_key_deterministic_and_tenant_scoped():
    k1 = cache._key("t1", "支持哪些付款方式？")
    k2 = cache._key("t1", "支持哪些付款方式!")
    k3 = cache._key("t2", "支持哪些付款方式？")
    assert k1 == k2  # 归一化后同 key
    assert k1 != k3  # 不同租户隔离


class _FakeRedis:
    def __init__(self):
        self.d = {}

    def get(self, key):
        return self.d.get(key)

    def set(self, key, val, ex=None):
        self.d[key] = val


def test_cache_roundtrip(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(cache, "get_redis", lambda: r)
    monkeypatch.setattr(config, "CACHE_ENABLED", True)
    cache.set_answer("t1", "支持哪些付款方式？", "我们支持微信/支付宝/对公转账。")
    # 不同标点/问法归一化后命中同一缓存
    assert cache.get_answer("t1", "支持哪些付款方式！") == "我们支持微信/支付宝/对公转账。"
    assert cache.get_answer("t1", "退款多久到账") is None


def test_cache_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(config, "CACHE_ENABLED", False)
    cache.set_answer("t1", "x", "y")
    assert cache.get_answer("t1", "x") is None
