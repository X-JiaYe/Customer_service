"""FastAPI 入口：SSE 流式接口 + Gradio 调试界面。

用法：
    python main.py --mode api      # 启动 FastAPI（默认）
    python main.py --mode gradio   # 启动 Gradio ChatInterface
"""
import argparse
import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

import config  # noqa: F401
from agent import chat, chat_stream, warm_up
from audit import record_audit, set_request_context
from auth import require_auth
from feedback import classify_unresolved, record_unanswered, recent, summarize
from metrics import observe_error, observe_latency, observe_request, metrics_response
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

# 会话管理：Redis 持久化（多 worker 共享、重启不丢），key = session:{tenant}:{session_id}
_session_store = RedisSessionStore()


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


def _maybe_record_unresolved(question: str, answer: str) -> None:
    """未命中监控 §5.2：答案判定为「未解决」时记录，供运营补知识分析。"""
    reason = classify_unresolved(answer)
    if reason:
        record_unanswered(question, reason=reason)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    stream: bool = True  # false = 只返回最终答案，不走 SSE 逐 token 推送（客户端减 token 输出）


@app.get("/health")
def health():
    return {"status": "ok"}


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

    if not req.stream:
        # 非流式：一次性返回最终答案，适合脚本/客户端简洁调用
        start = time.perf_counter()
        _append_turn(tenant, req.session_id, "user", req.message)
        status = "ok"
        try:
            answer = await run_in_threadpool(chat, req.message, history)
        except Exception:  # noqa: BLE001
            answer = "系统繁忙，请稍后重试"
            status = "error"
            observe_error(tenant)
        answer = _postprocess_answer(answer)
        _append_turn(tenant, req.session_id, "assistant", answer)
        _maybe_record_unresolved(req.message, answer)
        observe_request(tenant, status)
        observe_latency(tenant, time.perf_counter() - start)
        record_audit("chat", question=req.message, answer=answer, status=status)
        return JSONResponse({"answer": answer, "session_id": req.session_id, "tenant_id": tenant})

    async def event_stream():
        start = time.perf_counter()
        _append_turn(tenant, req.session_id, "user", req.message)
        answer_parts: list[str] = []
        status = "ok"
        try:
            # 在线程池中迭代同步生成器，逐 token 异步推送；内部已加锁串行化
            async for token in iterate_in_threadpool(chat_stream(req.message, history)):
                answer_parts.append(token)
                yield f"data: {json.dumps({'delta': token}, ensure_ascii=False)}\n\n"
        except Exception:  # noqa: BLE001
            answer_parts = ["系统繁忙，请稍后重试"]
            status = "error"
            observe_error(tenant)
            yield f"data: {json.dumps({'delta': answer_parts[0]}, ensure_ascii=False)}\n\n"
        answer = "".join(answer_parts)
        # 已流出的 token 无法撤回（SSE 固有限制）；此处仅对持久化历史脱敏，避免 Redis 留存明文 PII
        if config.ENABLE_PII_MASK:
            answer = mask_pii(answer)
        _append_turn(tenant, req.session_id, "assistant", answer)
        _maybe_record_unresolved(req.message, answer)
        observe_request(tenant, status)
        observe_latency(tenant, time.perf_counter() - start)
        record_audit("chat", question=req.message, answer=answer, status=status)
        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
