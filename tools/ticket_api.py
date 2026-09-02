"""工单创建工具：生成工单 ID，写入 tickets.jsonl。"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smolagents import tool

import config


@tool
def create_ticket(title: str, description: str, priority: str) -> str:
    """创建客服工单。

    Args:
        title: 工单标题，一句话简述问题
        description: 问题的详细描述
        priority: 优先级，可选 high / medium / low
    """
    if priority not in {"high", "medium", "low"}:
        priority = "medium"

    ticket_id = f"TK-{int(time.time())}"
    record = {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "priority": priority,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(config.TICKETS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[工单] {record}")
    return f"工单创建成功，工单号：{ticket_id}（优先级：{priority}）"
