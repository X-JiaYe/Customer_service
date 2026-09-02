"""工单创建工具：封装 TicketAPI 契约，dev 环境写本地 tickets.jsonl。"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smolagents import tool

import config
from audit import record_audit
from business_api import TicketAPI, mock_only


class MockTicketAPI:
    """TicketAPI 契约的 mock 实现：生成工单 ID 并追加写本地 tickets.jsonl。"""

    def create_ticket(self, title: str, description: str, priority: str) -> str:
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
        return ticket_id


def _get_ticket_api() -> TicketAPI:
    # 生产环境需接真实实现；当前只有 mock，prod 下显式报错
    mock_only("create_ticket")
    return MockTicketAPI()


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

    ticket_id = _get_ticket_api().create_ticket(title, description, priority)
    record_audit("create_ticket", ticket_id=ticket_id, priority=priority)
    return f"工单创建成功，工单号：{ticket_id}（优先级：{priority}）"
