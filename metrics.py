"""Prometheus 指标：QPS / 延迟 / 错误 / 转人工 / token。

- 通过 /metrics 端点暴露（Prometheus 文本格式，prometheus-client 生成）。
- 标签以 tenant 维度聚合，便于多租户观测。
"""
from prometheus_client import Counter, Histogram, generate_latest

REQUESTS = Counter(
    "cs_requests_total", "对话请求总数", ["tenant", "status"]
)
LATENCY = Histogram(
    "cs_chat_latency_seconds",
    "对话延迟（秒）",
    ["tenant"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 120.0),
)
ERRORS = Counter("cs_errors_total", "错误总数", ["tenant"])
TRANSFERS = Counter("cs_transfer_to_human_total", "转人工次数", ["tenant"])
# token 消耗：待接 LLM 返回元数据后填充（当前 LLM 调用不直接暴露 token 数，留接口）
TOKENS = Counter("cs_tokens_total", "token 消耗", ["tenant"])
# 服务指标体系 §5.5：知识命中/未命中、未解决
KNOWLEDGE_HITS = Counter("cs_knowledge_hits_total", "知识库检索命中次数", ["tenant"])
KNOWLEDGE_MISSES = Counter("cs_knowledge_misses_total", "知识库检索未命中次数", ["tenant"])
UNRESOLVED = Counter("cs_unresolved_total", "未解决（低置信/拒答/转人工/降级）次数", ["tenant"])
# Token 成本控制 §6.4：语义缓存命中（命中即省一次 LLM 调用）
CACHE_HITS = Counter("cs_cache_hits_total", "语义缓存命中次数", ["tenant"])


def observe_request(tenant: str, status: str) -> None:
    REQUESTS.labels(tenant, status).inc()


def observe_latency(tenant: str, seconds: float) -> None:
    LATENCY.labels(tenant).observe(seconds)


def observe_error(tenant: str) -> None:
    ERRORS.labels(tenant).inc()


def observe_transfer(tenant: str) -> None:
    TRANSFERS.labels(tenant).inc()


def observe_tokens(tenant: str, n: int) -> None:
    TOKENS.labels(tenant).inc(n)


def observe_knowledge_hit(tenant: str) -> None:
    KNOWLEDGE_HITS.labels(tenant).inc()


def observe_knowledge_miss(tenant: str) -> None:
    KNOWLEDGE_MISSES.labels(tenant).inc()


def observe_unresolved(tenant: str) -> None:
    UNRESOLVED.labels(tenant).inc()


def observe_cache_hit(tenant: str) -> None:
    CACHE_HITS.labels(tenant).inc()


def metrics_response() -> str:
    return generate_latest().decode("utf-8")
