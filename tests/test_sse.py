"""测试 sse.py 的 SSE 事件格式化与断点续传缓冲（纯逻辑，不触发模型/网络）。"""
import sse


def test_sse_event_format():
    out = sse.sse_event(3, {"delta": "你好"})
    assert out == 'id: 3\ndata: {"delta": "你好"}\n\n'


def test_stream_buffer_append_and_replay():
    buf = sse.StreamBuffer(ttl_seconds=120)
    buf.new("k")
    buf.append("k", 1, {"delta": "a"})
    buf.append("k", 2, {"delta": "b"})
    buf.append("k", 3, {"delta": "c"})

    assert buf.replay("k", 0) == [(1, {"delta": "a"}), (2, {"delta": "b"}), (3, {"delta": "c"})]
    assert buf.replay("k", 2) == [(3, {"delta": "c"})]
    assert buf.replay("k", 3) == []


def test_stream_buffer_replay_unknown_key():
    buf = sse.StreamBuffer(ttl_seconds=120)
    assert buf.replay("nope", 0) is None


def test_stream_buffer_finish_keeps_events():
    buf = sse.StreamBuffer(ttl_seconds=120)
    buf.new("k")
    buf.append("k", 1, {"delta": "x"})
    buf.append("k", 2, {"done": True})
    buf.finish("k")
    # finish 不清空事件，重放仍能拿到全部尾部
    assert buf.replay("k", 0) == [(1, {"delta": "x"}), (2, {"done": True})]


def test_stream_buffer_gc_removes_expired():
    buf = sse.StreamBuffer(ttl_seconds=-1)  # 立即过期
    buf.new("k1")
    buf.append("k1", 1, {"delta": "x"})
    buf.new("k2")  # new() 触发 _gc，清理过期的 k1
    assert buf.replay("k1", 0) is None
    assert buf.replay("k2", 0) == []
