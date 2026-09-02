"""审计日志 + 结构化 JSON 日志 + 请求上下文。

- request 上下文：用 contextvars 贯穿 request_id / tenant_id / session_id，
  由 require_auth 设置，工具（query_order/create_ticket/transfer_to_human）与审计读取。
- record_audit：写 Redis LIST（audit:log，带 LTRIM 上限 + TTL），Redis 不可用则 fail-open 仅打日志。
- get_json_logger：结构化 JSON 日志（单行、request_id 贯穿），本地开发无 Redis 也能观察。
"""
import contextvars
import json
import logging
import time
import uuid

import config
from store import get_redis

# 请求上下文（线程/异步安全，随 run_in_threadpool 传播到工作线程）
_request_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("cs_request_ctx", default=None)


def set_request_context(tenant_id: str, request_id: str, session_id: str = "") -> None:
    _request_ctx.set({"tenant_id": tenant_id, "request_id": request_id, "session_id": session_id})


def current_context() -> dict:
    return _request_ctx.get() or {}


# ---------- 结构化 JSON 日志 ----------
_LOG_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "message", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        ctx = current_context()
        payload["request_id"] = ctx.get("request_id")
        payload["tenant_id"] = ctx.get("tenant_id")
        for key, value in record.__dict__.items():
            if key not in _LOG_RESERVED:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def get_json_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ---------- 审计写 Redis ----------
_AUDIT_KEY = "audit:log"
_AUDIT_MAX = 10000  # 保留最近 1 万条，避免无限膨胀


def record_audit(action: str, **fields) -> None:
    """写一条审计记录到 Redis（audit:log），Redis 不可用则降级为仅结构化日志。"""
    if not config.AUDIT_ENABLED:
        return

    ctx = current_context()
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "request_id": ctx.get("request_id") or uuid.uuid4().hex,
        "tenant_id": ctx.get("tenant_id") or "default",
        "session_id": ctx.get("session_id") or "",
        "action": action,
    }
    record.update(fields)

    try:
        r = get_redis()
        pipe = r.pipeline()
        pipe.lpush(_AUDIT_KEY, json.dumps(record, ensure_ascii=False))
        pipe.ltrim(_AUDIT_KEY, 0, _AUDIT_MAX - 1)
        pipe.expire(_AUDIT_KEY, config.SESSION_TTL_SECONDS * 24)
        pipe.execute()
    except Exception as e:  # noqa: BLE001
        print(f"[audit] Redis 不可用，审计仅输出日志：{e}")

    # 始终输出结构化日志，本地开发（无 Redis）也能观察全链路
    get_json_logger("audit").info(action, extra={"record": record})
