"""测试 security.py 的 PII 脱敏与输出审核（纯函数，不触发任何重依赖）。"""
from security import check_output_safety, mask_pii


def test_mask_phone():
    assert mask_pii("请联系 13812345678") == "请联系 138****5678"


def test_mask_email():
    assert mask_pii("邮箱 admin@example.com 欢迎咨询") == "邮箱 a****@example.com 欢迎咨询"


def test_mask_idcard():
    assert mask_pii("身份证 110101199001011234 已登记") == "身份证 110101********1234 已登记"


def test_mask_bankcard():
    assert mask_pii("卡号 6222021234567890123") == "卡号 6222***********0123"


def test_mask_multiple_and_noop():
    text = "手机 13912345678，邮箱 foo@bar.com"
    assert "139****5678" in mask_pii(text)
    assert "foo@bar.com" not in mask_pii(text)
    # 无 PII 时原样返回
    assert mask_pii("你好，请问如何退款") == "你好，请问如何退款"


def test_check_safety_detects_strong_promise():
    hits = check_output_safety("我们保证一定退款")
    types = {h["type"] for h in hits}
    assert "强承诺" in types


def test_check_safety_detects_price_and_time():
    hits = check_output_safety("支付 199 元，3 个工作日内到账")
    types = {h["type"] for h in hits}
    assert "价格" in types
    assert "时限" in types


def test_check_safety_detects_disclaimer():
    hits = check_output_safety("本公司概不负责任何损失")
    assert any(h["type"] == "免责词" for h in hits)


def test_check_safety_empty():
    assert check_output_safety("") == []
    assert check_output_safety("这是正常的产品介绍") == []
