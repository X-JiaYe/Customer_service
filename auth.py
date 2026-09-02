"""鉴权 + Redis 限流 + request_id / tenant 注入。

- `require_auth` 是 FastAPI 依赖：解析 tenant_id，注入 request.state，
  并做按 租户+IP+接口 的 Redis 限流。
- 鉴权凭据（二选一，优先级从高到低）：
  1. `Authorization: Bearer <token>`：前端登录后签发的短期访问 token（Redis 存 token→tenant）；
  2. `X-API-Key`：长期 API Key（仅网关/服务端持有，不进浏览器）。
- `config.API_KEYS` 为空时进入开发模式（不鉴权，tenant 默认 "default"）；配置了即强制鉴权。
"""
import secrets
import uuid

from fastapi import HTTPException, Request

import config
from audit import set_request_context
from store import get_redis

_TOKEN_PREFIX = "auth:token:"


def resolve_tenant(api_key: str | None) -> str | None:
    """根据 API Key 解析 tenant_id；未知/缺失返回 None。"""
    if not api_key:
        return None
    return config.API_KEYS.get(api_key)


def issue_token(tenant: str) -> str:
    """为租户签发短期访问 token（Redis 存 token→tenant，带 TTL）。"""
    token = secrets.token_urlsafe(32)
    try:
        get_redis().set(f"{_TOKEN_PREFIX}{token}", tenant, ex=config.TOKEN_TTL_SECONDS)
    except Exception as e:  # noqa: BLE001
        print(f"[auth] 签发 token 依赖 Redis 失败（fail-open，token 可能无效）：{e}")
    return token


def resolve_token(token: str | None) -> str | None:
    """根据短期 token 解析 tenant；无效/过期/Redis 不可用返回 None。"""
    if not token:
        return None
    try:
        return get_redis().get(f"{_TOKEN_PREFIX}{token}")
    except Exception:  # noqa: BLE001
        return None


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
    """FastAPI 依赖：鉴权 + 注入 request_id/tenant + 限流。返回 tenant_id。

    优先 Bearer token（短期），其次 X-API-Key（长期）。
    """
    if config.API_KEYS:
        auth_header = request.headers.get("Authorization", "")
        token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
        tenant = resolve_token(token)
        if tenant is None:
            tenant = resolve_tenant(request.headers.get("X-API-Key"))
        if tenant is None:
            raise HTTPException(status_code=401, detail="无效或缺失的 API Key 或 token")
    else:
        # 开发模式：未配置 API_KEYS 时不鉴权
        tenant = "default"

    request.state.tenant_id = tenant
    request.state.request_id = uuid.uuid4().hex
    # 写入请求上下文（contextvar），供工具与审计读取 tenant/request_id
    set_request_context(tenant, request.state.request_id)
    _check_rate_limit(request, tenant)
    return tenant
