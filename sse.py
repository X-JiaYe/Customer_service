"""SSE 流式增强 §6.3：心跳 + 事件 ID + 断点续传缓冲。

- PING：SSE 注释行心跳，长回答期间防止代理/网关空闲超时，也让客户端能感知半开连接。
- sse_event：构造带 `id:` 的事件，客户端可据此追踪进度（Last-Event-ID 断点续传）。
- StreamBuffer：按 stream_key 缓存已产出的事件，断线重连后从某个 event_id 重放尾部。

StreamBuffer 线程安全、纯内存 + TTL，可单测（不触发模型 / 网络）。
"""
import json
import threading
import time

import config

PING = ": ping\n\n"


def sse_event(event_id: int, data: dict) -> str:
    """构造一条带 id 的 SSE 事件（data 为 dict，自动 JSON 序列化）。"""
    return f"id: {event_id}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class StreamBuffer:
    """断点续传缓冲：按 stream_key 缓存已产出的 SSE 事件，支持断线后从某个 event_id 重放。

    - 事件以 (event_id, payload_dict) 追加；TTL 过期自动清理，防内存泄漏。
    - 线程安全：生产线程（跑 LLM 流）写入，请求线程（重连）重放。
    """

    def __init__(self, ttl_seconds: float | None = None):
        self._ttl = ttl_seconds if ttl_seconds is not None else config.SSE_STREAM_TTL_SECONDS
        self._streams: dict[str, dict] = {}
        self._lock = threading.Lock()

    def new(self, stream_key: str) -> None:
        with self._lock:
            self._gc()
            self._streams[stream_key] = {"events": [], "done": False, "ts": time.time()}

    def append(self, stream_key: str, event_id: int, payload: dict) -> None:
        with self._lock:
            s = self._streams.get(stream_key)
            if s is not None:
                s["events"].append((event_id, payload))
                s["ts"] = time.time()

    def finish(self, stream_key: str) -> None:
        with self._lock:
            s = self._streams.get(stream_key)
            if s is not None:
                s["done"] = True
                s["ts"] = time.time()

    def replay(self, stream_key: str, from_event_id: int) -> list[tuple[int, dict]] | None:
        """返回 event_id > from_event_id 的尾部事件；stream_key 不存在则返回 None。"""
        with self._lock:
            s = self._streams.get(stream_key)
            if s is None:
                return None
            return [(eid, p) for eid, p in s["events"] if eid > from_event_id]

    def _gc(self) -> None:
        now = time.time()
        expired = [k for k, s in self._streams.items() if now - s["ts"] > self._ttl]
        for k in expired:
            del self._streams[k]
