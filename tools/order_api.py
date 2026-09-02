"""订单查询工具（模拟 API）。"""
import json

from smolagents import tool

# 模拟订单数据（5-10 条）
_ORDERS = [
    {
        "order_id": "TK20250301",
        "customer": "张三",
        "status": "已发货",
        "amount": 299.0,
        "logistics": "顺丰 SF1234567890，预计 2026-09-05 送达",
    },
    {
        "order_id": "TK20250302",
        "customer": "李四",
        "status": "待付款",
        "amount": 1299.0,
        "logistics": "尚未发货",
    },
    {
        "order_id": "TK20250303",
        "customer": "王五",
        "status": "已签收",
        "amount": 89.9,
        "logistics": "中通 ZT0987654321，已签收",
    },
    {
        "order_id": "TK20250410",
        "customer": "赵六",
        "status": "退款中",
        "amount": 459.0,
        "logistics": "退款处理中，预计 3-5 个工作日到账",
    },
    {
        "order_id": "TK20250411",
        "customer": "孙七",
        "status": "已完成",
        "amount": 199.0,
        "logistics": "京东物流 JD5566778899，已签收",
    },
    {
        "order_id": "TK20250412",
        "customer": "周八",
        "status": "运输中",
        "amount": 799.0,
        "logistics": "圆通 YT1122334455，已到达转运中心",
    },
]


@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单信息。

    Args:
        order_id: 订单号，例如 TK20250301
    """
    for order in _ORDERS:
        if order["order_id"] == order_id:
            return json.dumps(order, ensure_ascii=False, indent=2)
    return f"未找到订单 {order_id}，请确认订单号是否正确。"
