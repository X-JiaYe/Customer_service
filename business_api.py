"""真实业务接口契约（Protocol）+ 环境隔离。

- OrderAPI / TicketAPI：用 typing.Protocol 声明接口签名（鉴权 / 幂等 / 超时 / 重试约定
  见各方法 docstring）。真实第三方系统接入后，只需提供实现该契约的类并在工具里替换 mock。
- mock_only()：生产环境（ENV=prod）下调用 mock 实现时显式报错，杜绝「把测试数据当生产」。
"""
import config
from typing import Protocol, runtime_checkable


class EnvError(RuntimeError):
    """生产环境缺少真实业务接口实现时抛出。"""


@runtime_checkable
class OrderAPI(Protocol):
    def query_order(self, order_id: str) -> dict:
        """按订单号查询订单。返回订单详情 dict；订单不存在抛 KeyError。

        约定：调用方需持有鉴权凭据（由真实实现内部注入 token/签名）；接口幂等只读；
        超时 5s，失败重试 1 次后抛错。
        """
        ...


@runtime_checkable
class TicketAPI(Protocol):
    def create_ticket(self, title: str, description: str, priority: str) -> str:
        """创建工单，返回 ticket_id。

        约定：幂等（相同内容重复提交返回同一 ticket_id）；超时 5s，失败重试 1 次。
        """
        ...


def mock_only(name: str) -> None:
    """生产环境（ENV=prod）下调用 mock 实现时显式报错，避免误用。"""
    if config.ENV == "prod":
        raise EnvError(
            f"当前 ENV=prod，但「{name}」仅有 mock 实现。"
            "请接入真实业务系统（实现 OrderAPI / TicketAPI 契约）后再上线。"
        )
