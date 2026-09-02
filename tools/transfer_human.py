"""转人工工具。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smolagents import tool

from audit import current_context, record_audit
from metrics import observe_transfer


@tool
def transfer_to_human(reason: str) -> str:
    """将对话转接人工客服。

    Args:
        reason: 转人工的原因，例如投诉、情绪激动、超出能力范围等
    """
    tenant = current_context().get("tenant_id") or "default"
    observe_transfer(tenant)
    record_audit("transfer_to_human", reason=reason)
    print(f"[转人工] {time.strftime('%Y-%m-%d %H:%M:%S')} 原因：{reason}")
    return "正在为您转接人工客服，请稍候..."
