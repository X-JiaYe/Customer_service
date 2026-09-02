"""测试业务接口契约 + mock 隔离（不触发模型/Redis 连接）。"""
import pytest

import config
from business_api import EnvError, mock_only
from tools.order_api import MockOrderAPI


def test_mock_order_api_found():
    order = MockOrderAPI().query_order("TK20250301")
    assert order["customer"] == "张三"


def test_mock_order_api_missing():
    with pytest.raises(KeyError):
        MockOrderAPI().query_order("NOPE")


def test_mock_only_dev_noop(monkeypatch):
    monkeypatch.setattr(config, "ENV", "dev")
    mock_only("query_order")  # 不抛错


def test_mock_only_prod_raises(monkeypatch):
    monkeypatch.setattr(config, "ENV", "prod")
    with pytest.raises(EnvError):
        mock_only("query_order")
