"""未命中监控 + 转人工回流（升级方案 §5.2 / §5.3）。

把「答不上来 / 转人工」的案例沉淀到 Redis，供运营做知识补全与自动化解率分析：
- record_unanswered：记录低置信/拒答/未命中的问题。
- record_transfer：记录转人工案例（含原因）。
- summarize：对某类反馈做聚合，输出「高频 Top 榜」（供周报/补知识入口）。

存储：Redis LIST（feedback:unanswered / feedback:transfer），LTRIM 上限 + 7 天滑动 TTL。
fail-open：Redis 不可用时仅告警，不阻断主流程。
"""
import json
import time
import uuid

import config
from audit import current_context
from store import get_redis

_PREFIX = "feedback:"
_MAX = 10000
_TTL_SECONDS = 7 * 24 * 3600  # 保留 7 天

# 判定「未解决」的答案信号词（出现即视为低置信/拒答/降级/转人工）
UNRESOLVED_MARKERS = ("不确定", "无法回答", "建议转人工", "转人工", "转接人工", "系统繁忙", "稍后重试", "未找到")


def classify_unresolved(answer: str) -> str | None:
    """纯函数：根据答案文本判断是否「未解决」，命中返回信号词，否则 None。"""
    if not answer:
        return None
    for marker in UNRESOLVED_MARKERS:
        if marker in answer:
            return marker
    return None


def record_unanswered(question: str, reason: str = "") -> None:
    _push("unanswered", {"question": question, "reason": reason})


def record_transfer(reason: str) -> None:
    _push("transfer", {"reason": reason})


def _push(category: str, payload: dict) -> None:
    if not config.AUDIT_ENABLED:
        return
    ctx = current_context()
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "request_id": ctx.get("request_id") or uuid.uuid4().hex,
        "tenant_id": ctx.get("tenant_id") or "default",
        "session_id": ctx.get("session_id") or "",
    }
    record.update(payload)
    try:
        r = get_redis()
        key = _PREFIX + category
        pipe = r.pipeline()
        pipe.lpush(key, json.dumps(record, ensure_ascii=False))
        pipe.ltrim(key, 0, _MAX - 1)
        pipe.expire(key, _TTL_SECONDS)
        pipe.execute()
    except Exception as e:  # noqa: BLE001
        print(f"[feedback] Redis 不可用，反馈记录失败：{e}")


def recent(category: str, n: int = 20) -> list[dict]:
    """返回某类反馈最近 n 条（倒序，最新在前）。"""
    try:
        r = get_redis()
        raw = r.lrange(_PREFIX + category, 0, n - 1)
        out = []
        for x in raw:
            try:
                out.append(json.loads(x))
            except (json.JSONDecodeError, TypeError):
                continue
        return out
    except Exception:  # noqa: BLE001
        return []


def summarize(category: str, top_n: int = 20) -> list[dict]:
    """聚合某类反馈，输出高频 Top 榜：[{text, count, last_ts}, ...]（按次数降序）。"""
    items = recent(category, 1000)
    counter: dict[str, int] = {}
    last_ts: dict[str, str] = {}
    for it in items:
        key = (it.get("question") or it.get("reason") or "").strip()
        if not key:
            continue
        counter[key] = counter.get(key, 0) + 1
        last_ts.setdefault(key, it.get("ts", ""))
    ranked = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [{"text": k, "count": v, "last_ts": last_ts.get(k)} for k, v in ranked]
