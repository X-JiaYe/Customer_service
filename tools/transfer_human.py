"""转人工工具。

注意（能力边界）：当前仅记录转人工日志 + 案例回流（feedback），并无真实人工坐席端 /
队列 / 上下文摘要交接，属「半闭环」（升级方案待办 ④）。生产接坐席系统后再补真实派单。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smolagents import tool

from audit import current_context, record_audit
from feedback import record_transfer
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
    record_transfer(reason)  # §5.3 转人工案例回流，供自动化解率分析
    print(f"[转人工] {time.strftime('%Y-%m-%d %H:%M:%S')} 原因：{reason}")
    return "正在为您转接人工客服，请稍候..."
