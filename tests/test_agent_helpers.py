"""测试 agent.py 的纯函数：消息组装。不触发模型加载。"""
from agent import _build_messages


def test_build_messages_no_history():
    msgs = _build_messages("各部门职责", None, "上下文")
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "【检索上下文】" in msgs[-1]["content"]
    assert "各部门职责" in msgs[-1]["content"]


def test_build_messages_with_history():
    history = [
        {"role": "user", "content": "查订单"},
        {"role": "assistant", "content": "已发货"},
    ]
    msgs = _build_messages("什么时候到", history, "上下文")
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert roles[2] == "assistant"
    assert msgs[1]["content"] == "查订单"
    assert msgs[2]["content"] == "已发货"
    assert msgs[-1]["role"] == "user"


def test_build_messages_truncates_history():
    # 超过 MAX_HISTORY_TURNS*2 条时只保留最近
    history = [{"role": "user", "content": f"q{i}"} for i in range(100)]
    msgs = _build_messages("now", history, "上下文")
    contents = [m["content"] for m in msgs[1:-1]]
    assert "q0" not in contents  # 最老的一条被截掉
    assert "q99" in contents
