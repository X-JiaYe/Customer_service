"""企业 IM 渠道适配 §6.2：企业微信 / 钉钉 / 飞书 文本消息 ↔ 统一 Message。

各平台回调（webhook）原始 payload 结构不同，这里只负责：
- `parse_*`：从平台 payload 提取 (channel_user_id, content)，规整为 Message；
- `reply_*`：把最终答案格式化为平台约定的回复体。

纯函数、可单测；签名校验（WeCom signature / DingTalk·Feishu 的 HMAC secret）需各平台
密钥，留接口见 `verify_signature`，生产环境按渠道配置补齐。

平台字段参考：
- 企业微信自建应用回调：{MsgType, FromUserName, Content, ...}，回复为 JSON（或 XML，二选一）。
- 钉钉机器人：{senderId, text:{content}, ...}，回复 {msgtype:"text", text:{content}}。
- 飞书事件：{header:{event_type,...}, event:{message:{chat_id, content, ...}, sender:{sender_id:{open_id}}}}，
  回复 {msg_type:"text", content:{text}}；URL 验证需回显 challenge。
"""
from channels.schema import Message

# 各渠道默认租户（未按渠道配置映射时兜底；生产应由渠道配置派生）
SUPPORTED = ("wecom", "dingtalk", "feishu")


# ---------- 解析：平台 payload → Message ----------

def parse_wecom(payload: dict) -> Message | None:
    """企业微信自建应用回调 → Message（仅文本消息，其他类型返回 None）。"""
    if payload.get("MsgType") != "text":
        return None
    return Message(
        channel="wecom",
        channel_user_id=str(payload.get("FromUserName", "")),
        content=str(payload.get("Content", "")),
    )


def parse_dingtalk(payload: dict) -> Message | None:
    """钉钉机器人回调 → Message（仅文本，其他类型返回 None）。"""
    if payload.get("msgtype") != "text":
        return None
    text = payload.get("text") or {}
    return Message(
        channel="dingtalk",
        channel_user_id=str(payload.get("senderId", "")),
        content=str(text.get("content", "")),
    )


def parse_feishu(payload: dict) -> Message | None:
    """飞书事件回调 → Message（仅文本消息；URL 验证/其他事件返回 None）。"""
    header = payload.get("header") or {}
    if header.get("event_type") != "im.message.receive_v1":
        return None
    event = payload.get("event") or {}
    message = event.get("message") or {}
    if message.get("message_type") != "text":
        return None
    sender = (event.get("sender") or {}).get("sender_id") or {}
    return Message(
        channel="feishu",
        channel_user_id=str(sender.get("open_id", "")),
        content=str(message.get("content", "")),
    )


_PARSERS = {
    "wecom": parse_wecom,
    "dingtalk": parse_dingtalk,
    "feishu": parse_feishu,
}


# ---------- 回复：答案 → 平台回复体 ----------

def reply_wecom(text: str) -> dict:
    return {"MsgType": "text", "Text": {"Content": text}}


def reply_dingtalk(text: str) -> dict:
    return {"msgtype": "text", "text": {"content": text}}


def reply_feishu(text: str) -> dict:
    return {"msg_type": "text", "content": {"text": text}}


_REPLIERS = {
    "wecom": reply_wecom,
    "dingtalk": reply_dingtalk,
    "feishu": reply_feishu,
}


# ---------- 统一入口 ----------

def parse(channel: str, payload: dict) -> Message | None:
    """按渠道解析回调 payload → Message；未知渠道或非文本返回 None。"""
    fn = _PARSERS.get(channel)
    return fn(payload) if fn else None


def reply(channel: str, text: str) -> dict:
    """按渠道格式化回复体；未知渠道回退为通用 JSON（含 channel/text）。"""
    fn = _REPLIERS.get(channel)
    return fn(text) if fn else {"channel": channel, "text": text}


def verify_signature(channel: str, payload: dict, headers: dict) -> bool:
    """渠道签名校验占位 §6.2：生产需按平台 secret 校验，此处默认放行并留日志接口。

    企业微信：signature = sha1(sort(token, timestamp, nonce, echostr))；
    钉钉/飞书：timestamp + "\n" + secret 的 HMAC-SHA256 → base64 签名比对。
    """
    # 默认放行（开发模式）；生产需在渠道配置里提供 secret 后启用校验
    return True
