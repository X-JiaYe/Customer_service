"""依赖熔断与降级：CircuitBreaker + 优雅降级话术。

目标（升级方案 §5.9）：上游（DeepSeek/LLM）抖动时，不让整个对话直接失败，
而是按「超时重试 → 熔断打开 → 降级话术」逐级兜底，保证服务可响应。

设计：
- 只做进程内熔断（单实例足够；多实例可后续接 Redis 计数）。
- 失败口径：agent.run() 抛异常才计一次失败（LLM/上游故障为主）；
  工具内部已捕获的异常不算。
- 打开后冷却期结束进入「半开」：放行一次试探，成功即复位，失败立即重新打开。
"""
import threading
import time

# 降级话术：熔断打开或上游不可用时，返回给客户的兜底文案（明确不硬编、可转人工）
DEGRADE_MESSAGE = "系统当前繁忙或服务暂时不可用，请稍后重试；如需紧急处理，可转人工客服协助。"


class CircuitBreaker:
    """简单熔断器：连续失败达阈值 → 打开；冷却期后进入半开，允许一次试探。"""

    def __init__(self, name: str, failure_threshold: int = 3, reset_seconds: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """是否放行本次调用。打开（冷却中）返回 False；半开放行一次。"""
        with self._lock:
            if self._opened_at is None:
                return True
            if time.time() - self._opened_at >= self.reset_seconds:
                # 冷却结束，进入半开：复位并放行一次试探
                self._opened_at = None
                self._failures = 0
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.time()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_at is not None and time.time() - self._opened_at < self.reset_seconds


def create_llm_breaker() -> CircuitBreaker:
    """按 config 构建 LLM 熔断器（延迟读 config，避免循环导入）。"""
    import config

    return CircuitBreaker(
        name="llm",
        failure_threshold=config.CIRCUIT_FAILURE_THRESHOLD,
        reset_seconds=config.CIRCUIT_RESET_SECONDS,
    )
