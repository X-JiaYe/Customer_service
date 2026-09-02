"""成本记账 + 预算告警 §6.4：按租户核算 LLM 调用次数，超预算降级。

- 以「每租户每小时 LLM 调用次数」为计量单元（token 精确计量待接 LLM 返回元数据后替换，
  见 metrics.observe_tokens 的待办）。
- `check_budget`：发起 LLM 调用前调用，超预算返回 True，调用方降级（返回话术，不再调用 LLM）。
- `record_call`：LLM 调用成功发起后调用，记账 +1。
- Redis 异常一律 fail-open（不阻断正常对话，仅失去记账/告警能力）。

纯函数 `_hour_key` 可单测。
"""
import time

import config
from store import get_redis

_PREFIX = "budget:"
_HOUR_FMT = "%Y%m%d%H"


def _hour_key(tenant: str, now: float | None = None) -> str:
    """当前小时的计数 key（按租户 + 小时隔离）。"""
    ts = time.localtime(now if now is not None else time.time())
    return f"{_PREFIX}{tenant}:calls:{time.strftime(_HOUR_FMT, ts)}"


def current_usage(tenant: str) -> int:
    """当前小时该租户已记账的 LLM 调用次数。"""
    if not config.BUDGET_ENABLED:
        return 0
    try:
        return int(get_redis().get(_hour_key(tenant)) or 0)
    except Exception:  # noqa: BLE001
        return 0


def check_budget(tenant: str) -> bool:
    """是否已超预算（True=超，应降级不再调用 LLM）。"""
    if not config.BUDGET_ENABLED:
        return False
    return current_usage(tenant) >= config.BUDGET_MAX_CALLS_PER_HOUR


def record_call(tenant: str) -> None:
    """记账一次 LLM 调用（INCR + 设过期，防 key 无限堆积）。"""
    if not config.BUDGET_ENABLED:
        return
    try:
        r = get_redis()
        key = _hour_key(tenant)
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, 7200)  # 2 小时后过期，即保留上一小时的计数窗口
        pipe.execute()
    except Exception:  # noqa: BLE001
        pass
