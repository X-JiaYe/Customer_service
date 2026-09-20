"""FastAPI 入口：电商内容 RAG 知识库问答接口 + Gradio 调试界面。

用法：
    python main.py --mode api      # 启动 FastAPI（默认）
    python main.py --mode gradio   # 启动 Gradio ChatInterface
"""
import argparse
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import config  # noqa: F401
from agent import chat, warm_up
from auth import require_auth
from security import check_output_safety, mask_pii
from store import MemorySessionStore


def _warm_up_in_background() -> None:
    """后台预热模型（embedding + ChromaDB），避免首条消息冷启动。"""

    def _run():
        try:
            warm_up()
            print("[warmup] 模型预热完成")
        except Exception as e:  # noqa: BLE001
            print(f"[warmup] 预热失败（不影响服务，首条消息将冷启动）：{e}")

    threading.Thread(target=_run, daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时自动导入知识库（幂等），保证一键启动后即可用。"""
    if config.AUTO_INGEST:
        try:
            from knowledge.ingest import ingest  # 懒加载，避免 RAG 依赖缺失时启动失败

            ingest()
        except Exception as e:  # noqa: BLE001
            print(f"[startup] 自动导入知识库失败（可稍后调用 /knowledge/ingest 重试）：{e}")
    # 后台预热模型，避免首条消息冷启动（embedding + ChromaDB）
    _warm_up_in_background()
    yield


app = FastAPI(title="电商内容 RAG 知识库", lifespan=lifespan)

# 前后端分离：前端独立部署后跨域访问，此处按来源白名单放行
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 会话管理：进程内内存存储（单实例），key = session_id
_session_store = MemorySessionStore()


def _get_history(session_id: str) -> list[dict]:
    return _session_store.get_history(session_id)


def _append_turn(session_id: str, role: str, content: str) -> None:
    _session_store.append_turn(session_id, role, content)


def _postprocess_answer(text: str) -> str:
    """答案后处理：PII 脱敏 + 输出安全兜底（命中敏感承诺时附加官方口径提示）。"""
    if not text:
        return text
    if config.ENABLE_PII_MASK:
        text = mask_pii(text)
    if check_output_safety(text):
        text = text.rstrip() + "\n\n（注：以上信息请以官方渠道最新说明为准。）"
    return text


def _process_message(session_id: str, message: str, history: list[dict]) -> str:
    """处理单条消息：LLM 生成 + 后处理 + 会话持久化。返回最终答案。

    同步函数（内部直接调 chat），异步端点用 run_in_threadpool 包裹，避免阻塞事件循环。
    """
    _append_turn(session_id, "user", message)
    try:
        answer = chat(message, history)
    except Exception:  # noqa: BLE001
        answer = "系统繁忙，请稍后重试"
    answer = _postprocess_answer(answer)
    _append_turn(session_id, "assistant", answer)
    return answer


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request, _: None = Depends(require_auth)):
    """对话接口：非流式，一次性返回最终答案。"""
    history = _get_history(req.session_id)
    answer = await run_in_threadpool(_process_message, req.session_id, req.message, history)
    return JSONResponse({"answer": answer, "session_id": req.session_id})


@app.post("/knowledge/ingest")
def knowledge_ingest(_: None = Depends(require_auth)):
    try:
        from knowledge.ingest import ingest  # 懒加载，避免 RAG 依赖缺失时启动失败

        ingest()
        return JSONResponse({"status": "ok", "message": "知识库导入完成"})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


def run_gradio():
    import gradio as gr

    def respond(message, history):
        # Gradio 6.x 的 history 是 MessageDict 列表（dict：role/content/metadata/options），
        # 不是旧版的 [(user, assistant), ...] 元组列表，需按 dict 读取 role/content。
        def _as_text(content) -> str:
            # content 可能是 str，也可能是 list（多模态）；统一转文本
            return "".join(str(c) for c in content) if isinstance(content, list) else str(content)

        history_dicts = []
        for turn in history or []:
            if isinstance(turn, dict):
                role, content = turn.get("role"), turn.get("content")
            elif hasattr(turn, "role") and hasattr(turn, "content"):
                role, content = turn.role, turn.content
            elif isinstance(turn, (tuple, list)) and len(turn) >= 2:
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
        title="电商内容 RAG 知识库",
        save_history=True,  # 历史对话存浏览器 localStorage，刷新/切会话不丢
    ).launch()


def main():
    parser = argparse.ArgumentParser(description="电商内容 RAG 知识库")
    parser.add_argument("--mode", choices=["api", "gradio"], default="api")
    args = parser.parse_args()

    if args.mode == "gradio":
        run_gradio()
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
