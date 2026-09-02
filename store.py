"""Redis 连接与存储：会话持久化、审计、限流共用同一个连接。

会话用 Redis LIST 存储（原子 RPUSH + LTRIM 限长 + EXPIRE 设 TTL），
key = session:{tenant_id}:{session_id}。多 worker / 重启后会话不丢。
"""
import json

import redis as _redis

import config

_client = None


def get_redis():
    """懒加载单例 Redis 客户端（decode_responses=True）。"""
    global _client
    if _client is None:
        _client = _redis.from_url(config.REDIS_URL, decode_responses=True)
    return _client


class RedisSessionStore:
    """基于 Redis 的会话历史存储。"""

    def __init__(self):
        self._r = get_redis()

    def _key(self, tenant_id: str, session_id: str) -> str:
        return f"session:{tenant_id}:{session_id}"

    def get_history(self, tenant_id: str, session_id: str) -> list[dict]:
        """返回会话历史（[{role, content}, ...]），空会话返回 []。"""
        raw_list = self._r.lrange(self._key(tenant_id, session_id), 0, -1)
        history = []
        for raw in raw_list:
            try:
                item = json.loads(raw)
                if isinstance(item, dict):
                    history.append(item)
            except (json.JSONDecodeError, TypeError):
                continue
        return history

    def append_turn(self, tenant_id: str, session_id: str, role: str, content: str) -> None:
        """原子追加一条对话，限制保留最近 N 轮，刷新 TTL。"""
        key = self._key(tenant_id, session_id)
        pipe = self._r.pipeline()
        pipe.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
        # 只保留最近 MAX_HISTORY_TURNS*2 条（与 _build_prompt 折叠窗口一致），避免无限增长
        pipe.ltrim(key, -config.MAX_HISTORY_TURNS * 2, -1)
        pipe.expire(key, config.SESSION_TTL_SECONDS)
        pipe.execute()
