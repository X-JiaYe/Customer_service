"""测试 feedback.py 的未解决判定（纯函数，不触发 Redis/模型）。"""
from feedback import classify_unresolved


def test_classify_unresolved_uncertain():
    assert classify_unresolved("这个问题我不确定，建议您转人工确认") == "不确定"


def test_classify_unresolved_transfer():
    assert classify_unresolved("正在为您转接人工客服，请稍候") == "转接人工"


def test_classify_unresolved_degrade():
    assert classify_unresolved("系统当前繁忙，请稍后重试") == "稍后重试"


def test_classify_unresolved_none():
    assert classify_unresolved("您好，订单已发货，预计 3 天后送达") is None
    assert classify_unresolved("") is None
