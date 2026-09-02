"""测试 budget.py 的成本记账与预算告警（内存 fake redis，不触发网络）。"""
import budget
import config


def test_hour_key_deterministic():
    k1 = budget._hour_key("t1", 1700000000)
    k2 = budget._hour_key("t1", 1700000000)
    assert k1 == k2
    assert k1.startswith("budget:t1:calls:")


class _FakeRedis:
    def __init__(self):
        self.d = {}

    def get(self, key):
        return self.d.get(key)

    def pipeline(self):
        return _FakePipe(self)


class _FakePipe:
    def __init__(self, r):
        self.r = r
        self.cmds = []

    def incr(self, key):
        self.cmds.append(("incr", key))
        return self

    def expire(self, key, ttl):
        self.cmds.append(("expire", key, ttl))
        return self

    def execute(self):
        out = []
        for op, key in self.cmds:
            if op == "incr":
                self.r.d[key] = int(self.r.d.get(key, 0)) + 1
                out.append(self.r.d[key])
            else:
                out.append(True)
        return out


def test_check_budget_and_record(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(budget, "get_redis", lambda: r)
    monkeypatch.setattr(config, "BUDGET_ENABLED", True)
    monkeypatch.setattr(config, "BUDGET_MAX_CALLS_PER_HOUR", 2)
    assert budget.check_budget("t1") is False
    budget.record_call("t1")
    budget.record_call("t1")
    assert budget.current_usage("t1") == 2
    assert budget.check_budget("t1") is True


def test_budget_disabled(monkeypatch):
    monkeypatch.setattr(config, "BUDGET_ENABLED", False)
    monkeypatch.setattr(config, "BUDGET_MAX_CALLS_PER_HOUR", 0)
    assert budget.check_budget("t1") is False
    assert budget.current_usage("t1") == 0
