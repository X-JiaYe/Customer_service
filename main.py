"""FastAPI 入口：SSE 流式接口 + Gradio 调试界面。

用法：
    python main.py --mode api      # 启动 FastAPI（默认）
    python main.py --mode gradio   # 启动 Gradio ChatInterface
"""
import argparse
import asyncio
import contextvars
import json
import queue
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import Depends, FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import budget
import cache
import channels
import config  # noqa: F401
import sse
from agent import chat, chat_stream, warm_up
from audit import record_audit, set_request_context
from auth import issue_token, require_auth, resolve_tenant, resolve_token
from feedback import classify_unresolved, record_satisfaction, record_unanswered, recent, summarize
from metrics import (
    observe_cache_hit,
    observe_error,
    observe_latency,
    observe_request,
    observe_unresolved,
    metrics_response,
)
from security import check_output_safety, mask_pii
from store import RedisSessionStore


def _warm_up_in_background() -> None:
    """后台预热模型（embedding + ChromaDB + reranker），避免首条消息冷启动。"""

    def _run():
        try:
            warm_up()
            print("[warmup] 模型预热完成")
        except Exception as e:  # noqa: BLE001
            print(f"[warmup] 预热失败（不影响服务，首条消息将冷启动）：{e}")

    threading.Thread(target=_run, daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时自动导入知识库（幂等），保证 Docker 一键启动后即可用。"""
    if config.AUTO_INGEST:
        try:
            from knowledge.ingest import ingest  # 懒加载，避免 RAG 依赖缺失时启动失败

            ingest()
        except Exception as e:  # noqa: BLE001
            print(f"[startup] 自动导入知识库失败（可稍后调用 /knowledge/ingest 重试）：{e}")
    # 后台预热模型，避免首条消息冷启动（embedding + ChromaDB + reranker）
    _warm_up_in_background()
    yield


app = FastAPI(title="智能客服 Agent", lifespan=lifespan)

# 前后端分离：前端独立部署后跨域访问，此处按来源白名单放行
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 会话管理：Redis 持久化（多 worker 共享、重启不丢），key = session:{tenant}:{session_id}
_session_store = RedisSessionStore()

# 流式断点续传缓冲 §6.3：按 (tenant:session_id) 缓存已产出的 SSE 事件，断线重连后重放尾部
_stream_buffer = sse.StreamBuffer()


def _get_history(tenant_id: str, session_id: str) -> list[dict]:
    """取会话历史（Redis）。"""
    return _session_store.get_history(tenant_id, session_id)


def _append_turn(tenant_id: str, session_id: str, role: str, content: str) -> None:
    """追加一条对话（Redis）。"""
    _session_store.append_turn(tenant_id, session_id, role, content)


def _postprocess_answer(text: str) -> str:
    """答案后处理：PII 脱敏 + 输出安全兜底（命中敏感承诺时附加官方口径提示）。"""
    if not text:
        return text
    if config.ENABLE_PII_MASK:
        text = mask_pii(text)
    if check_output_safety(text):
        text = text.rstrip() + "\n\n（注：以上信息请以官方渠道最新说明为准。）"
    return text


def _maybe_record_unresolved(tenant: str, question: str, answer: str) -> None:
    """未命中监控 §5.2：答案判定为「未解决」时记录，供运营补知识分析。"""
    reason = classify_unresolved(answer)
    if reason:
        record_unanswered(question, reason=reason)
        observe_unresolved(tenant)


def _try_fast_path(tenant: str, session_id: str, message: str, history: list[dict]) -> tuple[str, str] | None:
    """快路径 §6.4：命中语义缓存或超预算时直接返回 (答案, kind)，否则 None 走 LLM。

    kind ∈ {"cached", "budget"}；命中时同步落地会话/审计/指标，不调用 LLM。
    """
    if not history:
        cached = cache.get_answer(tenant, message)
        if cached is not None:
            observe_cache_hit(tenant)
            observe_request(tenant, "ok")
            _append_turn(tenant, session_id, "user", message)
            _append_turn(tenant, session_id, "assistant", cached)
            record_audit("chat", question=message, answer=cached, status="ok", cached=True)
            return cached, "cached"
    if budget.check_budget(tenant):
        msg = "本时段调用量已达上限，请稍后再试，或转人工客服协助处理。"
        _append_turn(tenant, session_id, "user", message)
        _append_turn(tenant, session_id, "assistant", msg)
        observe_request(tenant, "budget_exceeded")
        record_audit("chat", question=message, answer=msg, status="budget_exceeded")
        return msg, "budget"
    return None


def _process_message(tenant: str, session_id: str, message: str, history: list[dict]) -> str:
    """非流式处理单条消息：LLM 生成 + 后处理 + 记账/缓存/审计/指标。返回最终答案。

    同步函数（内部直接调 chat），异步端点用 run_in_threadpool 包裹，避免阻塞事件循环。
    """
    start = time.perf_counter()
    _append_turn(tenant, session_id, "user", message)
    status = "ok"
    try:
        answer = chat(message, history)
    except Exception:  # noqa: BLE001
        answer = "系统繁忙，请稍后重试"
        status = "error"
        observe_error(tenant)
    budget.record_call(tenant)
    answer = _postprocess_answer(answer)
    if status == "ok" and not history:
        cache.set_answer(tenant, message, answer)
    _append_turn(tenant, session_id, "assistant", answer)
    _maybe_record_unresolved(tenant, message, answer)
    observe_request(tenant, status)
    observe_latency(tenant, time.perf_counter() - start)
    record_audit("chat", question=message, answer=answer, status=status)
    return answer


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    stream: bool = True  # false = 只返回最终答案，不走 SSE 逐 token 推送（客户端减 token 输出）


@app.get("/health")
def health():
    return {"status": "ok"}


class TokenRequest(BaseModel):
    api_key: str = ""


@app.post("/auth/token")
def auth_token(req: TokenRequest, request: Request):
    """登录换取短期访问 token §前后端分离：前端只存 token，长期 API Key 不进浏览器。"""
    if config.API_KEYS:
        api_key = req.api_key or request.headers.get("X-API-Key")
        tenant = resolve_tenant(api_key)
        if tenant is None:
            return JSONResponse({"status": "error", "message": "无效 API Key"}, status_code=401)
    else:
        # 开发模式：无 API_KEYS 时签发 default 租户的 token
        tenant = "default"
    token = issue_token(tenant)
    return JSONResponse({"token": token, "tenant": tenant, "expires_in": config.TOKEN_TTL_SECONDS})


@app.get("/metrics")
def metrics():
    """Prometheus 指标端点（供抓取，与 /health 同级别不鉴权）。"""
    return Response(metrics_response(), media_type=CONTENT_TYPE_LATEST)


@app.get("/admin/feedback")
def admin_feedback(
    category: str = "unanswered",
    limit: int = 20,
    summarize: bool = False,
    tenant: str = Depends(require_auth),
):
    """未命中/转人工反馈查询（§5.2/5.3）：供运营补知识与自动化解率分析。

    category: unanswered（未命中问题）| transfer（转人工原因）
    summarize=true 时返回高频 Top 榜，否则返回最近 limit 条。
    """
    if category not in {"unanswered", "transfer"}:
        return JSONResponse({"status": "error", "message": "category 仅支持 unanswered / transfer"}, status_code=400)
    if summarize:
        data = summarize(category, top_n=limit)
    else:
        data = recent(category, n=limit)
    return JSONResponse({"category": category, "count": len(data), "items": data})


class FeedbackRequest(BaseModel):
    rating: str  # up=👍 有帮助 / down=👎 没帮助
    session_id: str = "default"
    question: str = ""
    answer: str = ""


@app.post("/feedback")
def feedback(req: FeedbackRequest, request: Request, tenant: str = Depends(require_auth)):
    """满意度反馈 §6.1：用户对答案点赞/点踩，沉淀到 feedback:satisfaction 供运营分析。"""
    if req.rating not in ("up", "down"):
        return JSONResponse({"status": "error", "message": "rating 仅支持 up/down"}, status_code=400)
    set_request_context(tenant, request.state.request_id, req.session_id)
    record_satisfaction(req.rating, question=req.question, answer=req.answer)
    return JSONResponse({"status": "ok"})


@app.post("/knowledge/ingest")
def knowledge_ingest(tenant: str = Depends(require_auth)):
    try:
        from knowledge.ingest import ingest  # 懒加载，避免 RAG 依赖缺失时启动失败

        ingest()
        return JSONResponse({"status": "ok", "message": "知识库导入完成"})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request, tenant: str = Depends(require_auth)):
    """对话接口：默认 SSE 逐 token 推送；stream=false 时只返回最终答案（减 token 输出）。"""
    history = _get_history(tenant, req.session_id)
    # 补全请求上下文里的 session_id（require_auth 只设置了 tenant/request_id）
    set_request_context(tenant, request.state.request_id, req.session_id)

    # 快路径 §6.4：语义缓存命中 / 超预算降级（不调用 LLM），未命中走下方 LLM 路径
    fast = _try_fast_path(tenant, req.session_id, req.message, history)
    if fast is not None:
        answer, kind = fast
        flag = {"cached": True} if kind == "cached" else {"budget_exceeded": True}
        if not req.stream:
            return JSONResponse(
                {"answer": answer, "session_id": req.session_id, "tenant_id": tenant, **flag}
            )

        async def _fast_stream():
            yield sse.sse_event(1, {"delta": answer})
            yield sse.sse_event(2, {"done": True})

        return StreamingResponse(_fast_stream(), media_type="text/event-stream")

    if not req.stream:
        # 非流式：一次性返回最终答案，适合脚本/客户端简洁调用
        answer = await run_in_threadpool(_process_message, tenant, req.session_id, req.message, history)
        return JSONResponse({"answer": answer, "session_id": req.session_id, "tenant_id": tenant})

    # 断点续传 §6.3：客户端带 Last-Event-ID 重连时，从缓冲重放尾部，不重复生成
    stream_key = f"{tenant}:{req.session_id}"
    last_event_id = request.headers.get("last-event-id")
    if last_event_id is not None:
        try:
            from_id = int(last_event_id)
        except ValueError:
            from_id = 0
        replay = _stream_buffer.replay(stream_key, from_id)
        if replay is not None:
            async def _replay_stream():
                for eid, payload in replay:
                    yield sse.sse_event(eid, payload)

            return StreamingResponse(_replay_stream(), media_type="text/event-stream")

    _stream_buffer.new(stream_key)
    start = time.perf_counter()
    _append_turn(tenant, req.session_id, "user", req.message)

    # 生产线程：跑 LLM 流 + 后处理，独立于客户端是否在线（保证会话/审计落地、缓冲可重放）
    buf: queue.Queue = queue.Queue()

    def _producer() -> None:
        event_id = 0
        parts: list[str] = []
        status = "ok"
        try:
            for token in chat_stream(req.message, history):
                parts.append(token)
                event_id += 1
                _stream_buffer.append(stream_key, event_id, {"delta": token})
                buf.put(("token", event_id, token))
        except Exception:  # noqa: BLE001
            parts = ["系统繁忙，请稍后重试"]
            status = "error"
            observe_error(tenant)
            event_id += 1
            _stream_buffer.append(stream_key, event_id, {"delta": parts[0]})
            buf.put(("token", event_id, parts[0]))
        budget.record_call(tenant)
        # 后处理（脱敏/持久化/审计/未命中回流），跑在请求上下文内
        answer = "".join(parts)
        if config.ENABLE_PII_MASK:
            answer = mask_pii(answer)
        if status == "ok" and not history:
            cache.set_answer(tenant, req.message, answer)
        _append_turn(tenant, req.session_id, "assistant", answer)
        _maybe_record_unresolved(tenant, req.message, answer)
        observe_request(tenant, status)
        observe_latency(tenant, time.perf_counter() - start)
        record_audit("chat", question=req.message, answer=answer, status=status)
        event_id += 1
        _stream_buffer.append(stream_key, event_id, {"done": True})
        _stream_buffer.finish(stream_key)
        buf.put(("done", event_id, None))

    ctx = contextvars.copy_context()
    threading.Thread(target=lambda: ctx.run(_producer), daemon=True).start()

    async def event_stream():
        loop = asyncio.get_running_loop()
        while True:
            try:
                kind, eid, payload = await loop.run_in_executor(
                    None, buf.get, True, config.SSE_HEARTBEAT_SECONDS
                )
            except queue.Empty:
                yield sse.PING  # 心跳：长回答期间防空闲超时
                continue
            if kind == "done":
                yield sse.sse_event(eid, {"done": True})
                break
            yield sse.sse_event(eid, {"delta": payload})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/webhook/{channel}")
async def webhook_endpoint(
    channel: str, payload: dict, request: Request, tenant: str = Depends(require_auth)
):
    """企业 IM 渠道接入 §6.2：接收企业微信/钉钉/飞书回调，规整 → 处理 → 按平台格式回复。

    渠道鉴权复用 X-API-Key（生产由网关/渠道配置派生 tenant）；签名校验见
    channels.webhook.verify_signature（当前放行，生产补各平台 secret）。
    """
    if channel not in channels.webhook.SUPPORTED:
        return JSONResponse({"status": "error", "message": f"不支持的渠道：{channel}"}, status_code=404)
    msg = channels.webhook.parse(channel, payload)
    if msg is None:
        return JSONResponse(channels.webhook.reply(channel, "仅支持文本消息"))
    if not channels.webhook.verify_signature(channel, payload, dict(request.headers)):
        return JSONResponse(channels.webhook.reply(channel, "签名校验失败"), status_code=401)

    session_id = msg.session_key()
    set_request_context(tenant, request.state.request_id, session_id)
    history = _get_history(tenant, session_id)
    fast = _try_fast_path(tenant, session_id, msg.content, history)
    if fast is not None:
        answer, _ = fast
    else:
        answer = await run_in_threadpool(_process_message, tenant, session_id, msg.content, history)
    return JSONResponse(channels.webhook.reply(channel, answer))


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """WebSocket 渠道 §6.2：长连接多轮对话，客户端发 {"message","session_id","stream"}。

    鉴权：请求头 X-API-Key 或查询参数 api_key（浏览器无法设头时用后者）。
    当前返回完整答案（非流式）；流式逐 token 推送可复用 SSE 生产线程模式，留作后续。
    """
    # 浏览器 WS 无法设自定义头，token 走查询参数；API Key 也兼容头/查询参数
    token = websocket.query_params.get("token") or ""
    api_key = websocket.headers.get("X-API-Key") or websocket.query_params.get("api_key")
    if config.API_KEYS:
        tenant = resolve_token(token) if token else None
        if tenant is None:
            tenant = resolve_tenant(api_key)
        if tenant is None:
            await websocket.close(code=1008, reason="无效或缺失的 API Key 或 token")
            return
    else:
        tenant = "default"

    await websocket.accept()
    request_id = uuid.uuid4().hex
    try:
        while True:
            data = await websocket.receive_json()
            message = str(data.get("message", "")).strip()
            session_id = str(data.get("session_id") or "default")
            if not message:
                await websocket.send_json({"error": "message 不能为空"})
                continue
            set_request_context(tenant, request_id, session_id)
            history = _get_history(tenant, session_id)
            fast = _try_fast_path(tenant, session_id, message, history)
            if fast is not None:
                answer, kind = fast
                flag = {"cached": True} if kind == "cached" else {"budget_exceeded": True}
                await websocket.send_json({"answer": answer, "session_id": session_id, **flag})
                continue
            answer = await run_in_threadpool(_process_message, tenant, session_id, message, history)
            await websocket.send_json({"answer": answer, "session_id": session_id})
    except WebSocketDisconnect:
        pass


def run_gradio():
    import gradio as gr

    def respond(message, history):
        # Gradio 6.x 的 history 是 MessageDict 列表（dict：role/content/metadata/options），
        # 不是旧版的 [(user, assistant), ...] 元组列表，需按 dict 读取 role/content。
        def _as_text(content) -> str:
            # content 可能是 str，也可能是 list（多模态）；客服场景统一转文本
            return "".join(str(c) for c in content) if isinstance(content, list) else str(content)

        history_dicts = []
        for turn in history or []:
            if isinstance(turn, dict):
                role, content = turn.get("role"), turn.get("content")
            elif hasattr(turn, "role") and hasattr(turn, "content"):
                # 兜底：ChatMessage 等带 role/content 属性的对象
                role, content = turn.role, turn.content
            elif isinstance(turn, (tuple, list)) and len(turn) >= 2:
                # 兼容旧版 (user, assistant) 二元组
                history_dicts.append({"role": "user", "content": _as_text(turn[0])})
                if turn[1]:
                    history_dicts.append({"role": "assistant", "content": _as_text(turn[1])})
                continue
            else:
                continue
            if role is None or content is None:
                continue
            history_dicts.append({"role": role, "content": _as_text(content)})
        try:
            return chat(message, history_dicts)
        except Exception:  # noqa: BLE001
            return "系统繁忙，请稍后重试"

    # 启动前后台预热模型，避免首条消息冷启动
    _warm_up_in_background()
    gr.ChatInterface(
        fn=respond,
        title="智能客服 Agent",
        save_history=True,  # 历史对话存浏览器 localStorage，刷新/切会话不丢
    ).launch()


def main():
    parser = argparse.ArgumentParser(description="智能客服 Agent")
    parser.add_argument("--mode", choices=["api", "gradio"], default="api")
    args = parser.parse_args()

    if args.mode == "gradio":
        run_gradio()
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
