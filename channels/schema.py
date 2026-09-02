"""跨渠道统一消息模型 §6.2。

各渠道（REST / 企业微信 / 钉钉 / 飞书 / WebSocket）的原始消息先规整为 Message，
agent 核心只消费 Message，回答后再由渠道适配器格式化为平台约定格式。
"""
from dataclasses import dataclass, field


@dataclass
class Message:
    """一条规整后的用户消息。

    - channel：渠道标识（rest / wecom / dingtalk / feishu / websocket）。
    - channel_user_id：渠道内用户标识（如企业微信 UserID / 钉钉 senderId），
      用于跨请求稳定关联同一用户的多轮会话。
    - tenant：租户（由渠道配置或鉴权派生）。
    - session_id：会话 id；为空时按 channel+user 派生稳定会话，保证同一用户多轮不丢上下文。
    """

    channel: str
    channel_user_id: str
    content: str
    tenant: str = "default"
    session_id: str = ""
    stream: bool = True
    extra: dict = field(default_factory=dict)

    def session_key(self) -> str:
        """渠道 + 用户 派生的稳定会话 id（未显式给 session_id 时使用）。"""
        return self.session_id or f"{self.channel}:{self.channel_user_id}"
