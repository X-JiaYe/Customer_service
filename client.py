"""智能客服 API 的简洁命令行客户端（零依赖，仅标准库）。

目标：减少 token 输出 —— 默认只打印最终答案，不逐 token 刷屏、不打印原始 SSE/JSON。

用法：
    python client.py "你们的产品支持哪些支付方式？"          # 单轮问答（只打印最终答案）
    python client.py "查订单 TK20250301" --session demo        # 多轮会话
    python client.py "..." --stream                             # 需要看逐 token 流式输出时
    python client.py --health                                   # 健康检查
    python client.py --ingest                                   # 触发知识库重新导入
"""
import argparse
import json
import os
import sys
import urllib.request

# Windows GBK 控制台 → UTF-8，避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_URL = "http://127.0.0.1:8000"


def _post(base_url: str, path: str, payload: dict | None = None, api_key: str | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else b""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def _post_stream(base_url: str, path: str, payload: dict, api_key: str | None = None):
    """流式模式：逐 token 打印（仅演示用，默认关闭）。"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            try:
                evt = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if evt.get("delta"):
                sys.stdout.write(evt["delta"])
                sys.stdout.flush()
        sys.stdout.write("\n")


def main():
    p = argparse.ArgumentParser(description="智能客服 API 简洁客户端")
    p.add_argument("message", nargs="?", help="要发送的问题")
    p.add_argument("--session", default="default", help="会话 ID（多轮记忆）")
    p.add_argument("--url", default=DEFAULT_URL, help="服务地址")
    p.add_argument("--stream", action="store_true", help="逐 token 流式输出（默认关闭，只给最终答案）")
    p.add_argument("--api-key", default=os.environ.get("CS_API_KEY", "sk-local-dev"), help="API Key（默认取环境变量 CS_API_KEY）")
    p.add_argument("--health", action="store_true", help="健康检查")
    p.add_argument("--ingest", action="store_true", help="触发知识库重新导入")
    args = p.parse_args()

    base_url = args.url.rstrip("/")

    if args.health:
        print(json.dumps(_post(base_url, "/health"), ensure_ascii=False))
        return
    if args.ingest:
        print(json.dumps(_post(base_url, "/knowledge/ingest", api_key=args.api_key), ensure_ascii=False))
        return
    if not args.message:
        p.print_help()
        return

    payload = {"message": args.message, "session_id": args.session, "stream": args.stream}
    if args.stream:
        _post_stream(base_url, "/chat", payload, api_key=args.api_key)
    else:
        result = _post(base_url, "/chat", payload, api_key=args.api_key)
        # 只打印最终答案，过滤掉冗余字段，进一步减少输出
        print(result.get("answer", ""))


if __name__ == "__main__":
    main()
