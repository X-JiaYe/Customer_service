"""电商知识库 API 的简洁命令行客户端（零依赖，仅标准库）。

目标：减少 token 输出 —— 默认只打印最终答案。

用法：
    python client.py "各部门岗位职责怎么划分？"        # 单轮问答（只打印最终答案）
    python client.py "..." --session demo                # 多轮会话
    python client.py --health                            # 健康检查
    python client.py --ingest                            # 触发知识库重新导入
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


def main():
    p = argparse.ArgumentParser(description="电商知识库 API 简洁客户端")
    p.add_argument("message", nargs="?", help="要发送的问题")
    p.add_argument("--session", default="default", help="会话 ID（多轮记忆）")
    p.add_argument("--url", default=DEFAULT_URL, help="服务地址")
    p.add_argument("--api-key", default=os.environ.get("CS_API_KEY", ""), help="API Key（配置了 API_KEY 时才需要）")
    p.add_argument("--health", action="store_true", help="健康检查")
    p.add_argument("--ingest", action="store_true", help="触发知识库重新导入")
    args = p.parse_args()

    base_url = args.url.rstrip("/")

    if args.health:
        print(json.dumps(_post(base_url, "/health"), ensure_ascii=False))
        return
    if args.ingest:
        print(json.dumps(_post(base_url, "/knowledge/ingest", api_key=args.api_key or None), ensure_ascii=False))
        return
    if not args.message:
        p.print_help()
        return

    payload = {"message": args.message, "session_id": args.session}
    result = _post(base_url, "/chat", payload, api_key=args.api_key or None)
    # 只打印最终答案，过滤掉冗余字段，进一步减少输出
    print(result.get("answer", ""))


if __name__ == "__main__":
    main()
