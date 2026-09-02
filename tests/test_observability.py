"""测试可观测：metrics 文本渲染 + audit 请求上下文（纯逻辑，不触发 Redis/模型）。"""
import audit
import metrics


def test_metrics_response_contains_request_counter():
    out = metrics.metrics_response()
    assert "cs_requests_total" in out
    assert "cs_chat_latency_seconds" in out


def test_metrics_counters_increment():
    before = metrics.REQUESTS.labels("t_test", "ok")._value.get()
    metrics.observe_request("t_test", "ok")
    assert metrics.REQUESTS.labels("t_test", "ok")._value.get() == before + 1


def test_request_context_roundtrip():
    audit.set_request_context("tenant_x", "req-123", "sess-1")
    ctx = audit.current_context()
    assert ctx["tenant_id"] == "tenant_x"
    assert ctx["request_id"] == "req-123"
    assert ctx["session_id"] == "sess-1"
