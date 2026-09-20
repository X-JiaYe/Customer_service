"""进程内内存会话存储（单实例 MVP，无外部依赖）。

会话用 dict 存储（session_id -> [{role, content}, ...]），惰性过期清理，
不含租户维度（多租户已移除）。重启后会话不保留。
"""
import threading
import time

import config


class MemorySessionStore:
    """基于进程内 dict 的会话历史存储，线程安全，带 TTL 惰性清理。"""

    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}
        self._last_access: dict[str, float] = {}
        self._lock = threading.Lock()

    def _gc(self) -> None:
        now = time.time()
        expired = [k for k, ts in self._last_access.items() if now - ts > config.SESSION_TTL_SECONDS]
        for k in expired:
            self._sessions.pop(k, None)
            self._last_access.pop(k, None)

    def get_history(self, session_id: str) -> list[dict]:
        """返回会话历史（[{role, content}, ...]），空会话返回 []。"""
        with self._lock:
            self._gc()
            self._last_access[session_id] = time.time()
            return list(self._sessions.get(session_id, []))

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        """追加一条对话，限制保留最近 N 轮，刷新访问时间。"""
        with self._lock:
            self._gc()
            hist = self._sessions.setdefault(session_id, [])
            hist.append({"role": role, "content": content})
            # 只保留最近 MAX_HISTORY_TURNS*2 条（与 agent 折叠窗口一致），避免无限增长
            limit = config.MAX_HISTORY_TURNS * 2
            if len(hist) > limit:
                del hist[: len(hist) - limit]
            self._last_access[session_id] = time.time()
