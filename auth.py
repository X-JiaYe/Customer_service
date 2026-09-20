"""鉴权：单 API Key 校验。

- `require_auth` 是 FastAPI 依赖：`config.API_KEY` 非空时校验 `X-API-Key` 头，
  不匹配返回 401；为空 = 开发模式（免鉴权）。
- 多租户已移除，不再注入 tenant；仅注入 request_id。
"""
import uuid

from fastapi import HTTPException, Request

import config


def require_auth(request: Request) -> None:
    """FastAPI 依赖：校验 X-API-Key（配置了 API_KEY 时），并注入 request_id。"""
    if config.API_KEY:
        provided = request.headers.get("X-API-Key", "")
        if provided != config.API_KEY:
            raise HTTPException(status_code=401, detail="无效或缺失的 API Key")
    request.state.request_id = uuid.uuid4().hex
