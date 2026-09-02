"""转人工工具。"""
import time

from smolagents import tool


@tool
def transfer_to_human(reason: str) -> str:
    """将对话转接人工客服。

    Args:
        reason: 转人工的原因，例如投诉、情绪激动、超出能力范围等
    """
    print(f"[转人工] {time.strftime('%Y-%m-%d %H:%M:%S')} 原因：{reason}")
    return "正在为您转接人工客服，请稍候..."
