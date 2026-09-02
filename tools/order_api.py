"""订单查询工具：封装 OrderAPI 契约，dev 环境用内置 mock 数据。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smolagents import tool

from audit import record_audit
from business_api import OrderAPI, mock_only

# 模拟订单数据（仅 dev 环境使用；prod 需接真实系统实现 OrderAPI 契约）
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


class MockOrderAPI:
    """OrderAPI 契约的 mock 实现（内置样例数据）。"""

    def query_order(self, order_id: str) -> dict:
        for order in _ORDERS:
            if order["order_id"] == order_id:
                return order
        raise KeyError(order_id)


def _get_order_api() -> OrderAPI:
    # 生产环境需接真实实现；当前只有 mock，prod 下显式报错
    mock_only("query_order")
    return MockOrderAPI()


@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单信息。

    Args:
        order_id: 订单号，例如 TK20250301
    """
    try:
        order = _get_order_api().query_order(order_id)
    except KeyError:
        record_audit("query_order", order_id=order_id, found=False)
        return f"未找到订单 {order_id}，请确认订单号是否正确。"
    record_audit("query_order", order_id=order_id, found=True)
    return json.dumps(order, ensure_ascii=False, indent=2)
