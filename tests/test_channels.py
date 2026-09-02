"""测试多渠道接入 §6.2：统一消息模型 + 企业 IM 解析/回复（纯函数，不触发网络）。"""
from channels import webhook
from channels.schema import Message


def test_message_session_key_explicit_wins():
    m = Message(channel="wecom", channel_user_id="u1", content="hi", session_id="s-123")
    assert m.session_key() == "s-123"


def test_message_session_key_derived():
    m = Message(channel="wecom", channel_user_id="u1", content="hi")
    assert m.session_key() == "wecom:u1"


def test_parse_wecom_text():
    m = webhook.parse_wecom({"MsgType": "text", "FromUserName": "zhangsan", "Content": "怎么退款"})
    assert m.channel == "wecom"
    assert m.channel_user_id == "zhangsan"
    assert m.content == "怎么退款"


def test_parse_wecom_non_text_returns_none():
    assert webhook.parse_wecom({"MsgType": "image", "FromUserName": "zhangsan"}) is None


def test_parse_dingtalk_text():
    m = webhook.parse_dingtalk({"msgtype": "text", "senderId": "u_9", "text": {"content": "查订单"}})
    assert m.channel == "dingtalk"
    assert m.channel_user_id == "u_9"
    assert m.content == "查订单"


def test_parse_dingtalk_non_text_returns_none():
    assert webhook.parse_dingtalk({"msgtype": "markdown"}) is None


def test_parse_feishu_text():
    payload = {
        "header": {"event_type": "im.message.receive_v1"},
        "event": {
            "message": {"message_type": "text", "content": "你好"},
            "sender": {"sender_id": {"open_id": "ou_abc"}},
        },
    }
    m = webhook.parse_feishu(payload)
    assert m.channel == "feishu"
    assert m.channel_user_id == "ou_abc"
    assert m.content == "你好"


def test_parse_feishu_url_verify_returns_none():
    # 飞书 URL 验证回调没有 im.message.receive_v1 事件类型，应返回 None
    assert webhook.parse_feishu({"challenge": "xxx"}) is None


def test_reply_formats():
    assert webhook.reply_wecom("好的") == {"MsgType": "text", "Text": {"Content": "好的"}}
    assert webhook.reply_dingtalk("好的") == {"msgtype": "text", "text": {"content": "好的"}}
    assert webhook.reply_feishu("好的") == {"msg_type": "text", "content": {"text": "好的"}}


def test_dispatch_parse_and_reply():
    assert webhook.parse("dingtalk", {"msgtype": "text", "senderId": "u", "text": {"content": "x"}}).content == "x"
    assert webhook.parse("unknown", {}) is None
    assert webhook.reply("unknown", "hi") == {"channel": "unknown", "text": "hi"}
