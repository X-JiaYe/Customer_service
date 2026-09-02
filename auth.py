"""API Key 鉴权 + Redis 限流 + request_id / tenant 注入。

- `require_auth` 是 FastAPI 依赖：校验 `X-API-Key` → 解析 tenant_id，注入 request.state，
  并做按 租户+IP+接口 的 Redis 限流。
- `config.API_KEYS` 为空时进入开发模式（不鉴权，tenant 默认 "default"），便于本地调试；
  配置了 API_KEYS 即强制鉴权，未知/缺失 key 一律 401。
"""
import uuid

from fastapi import HTTPException, Request

import config
from audit import set_request_context
from store import get_redis


def resolve_tenant(api_key: str | None) -> str | None:
    """根据 API Key 解析 tenant_id；未知/缺失返回 None。"""
    if not api_key:
        return None
    return config.API_KEYS.get(api_key)


def _check_rate_limit(request: Request, tenant: str) -> None:
    """Redis 固定窗口限流：超限抛 429；Redis 不可用则 fail-open（仅记录，不阻断）。"""
    client_ip = request.client.host if request.client else "unknown"
    key = f"ratelimit:{tenant}:{client_ip}:{request.url.path}"
    try:
        r = get_redis()
        n = r.incr(key)
        if n == 1:
            r.expire(key, config.RATE_LIMIT_WINDOW_SECONDS)
        if n > config.RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[auth] 限流依赖 Redis 不可用，跳过限流（fail-open）：{e}")


def require_auth(request: Request) -> str:
    """FastAPI 依赖：鉴权 + 注入 request_id/tenant + 限流。返回 tenant_id。"""
    if config.API_KEYS:
        api_key = request.headers.get("X-API-Key")
        tenant = resolve_tenant(api_key)
        if tenant is None:
            raise HTTPException(status_code=401, detail="无效或缺失的 API Key")
    else:
        # 开发模式：未配置 API_KEYS 时不鉴权
        tenant = "default"

    request.state.tenant_id = tenant
    request.state.request_id = uuid.uuid4().hex
    # 写入请求上下文（contextvar），供工具与审计读取 tenant/request_id
    set_request_context(tenant, request.state.request_id)
    _check_rate_limit(request, tenant)
    return tenant
