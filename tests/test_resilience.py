"""测试 resilience.py 的 CircuitBreaker 状态机（纯逻辑，用假时钟，不真实 sleep）。"""
import time

from resilience import CircuitBreaker


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_breaker_opens_after_threshold(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    b = CircuitBreaker("t", failure_threshold=2, reset_seconds=10)
    assert b.allow() is True
    b.record_failure()
    assert b.allow() is True  # 1 次失败 < 阈值
    b.record_failure()
    assert b.is_open is True
    assert b.allow() is False  # 打开（冷却中）


def test_breaker_half_open_then_reset(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock)
    b = CircuitBreaker("t", failure_threshold=2, reset_seconds=10)
    b.record_failure()
    b.record_failure()
    assert b.allow() is False
    clock.advance(11)  # 冷却结束
    assert b.allow() is True  # 半开放行一次试探
    b.record_success()
    assert b.is_open is False


def test_breaker_success_resets_failure_count():
    b = CircuitBreaker("t", failure_threshold=2, reset_seconds=10)
    b.record_failure()
    b.record_success()
    b.record_failure()
    assert b.allow() is True  # 未达阈值
