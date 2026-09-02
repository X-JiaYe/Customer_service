"""多渠道接入 §6.2：统一消息模型 + 各渠道适配器（REST / Webhook / WebSocket）。

- `schema`：跨渠道统一的消息模型 Message。
- `webhook`：企业 IM（企业微信 / 钉钉 / 飞书）回调解析与回复格式化。
- 渠道适配器把各平台原始消息规整为 Message，回答后再格式化回平台约定格式，
  使 agent 核心（chat/chat_stream）不感知渠道差异。
"""
from channels.schema import Message
from channels import webhook

__all__ = ["Message", "webhook"]
