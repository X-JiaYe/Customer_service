"""测试 agent.py 的纯函数：JSON 反转义、闭合结构剥离、历史折叠。均不触发模型加载。"""
import agent
from agent import _build_prompt, _extract_answer_json, _json_unescape, _strip_json_closers


def test_json_unescape_basic():
    assert _json_unescape("a\\nb") == "a\nb"
    assert _json_unescape("a\\tb") == "a\tb"
    assert _json_unescape('say \\"hi\\"') == 'say "hi"'
    assert _json_unescape("a\\\\b") == "a\\b"  # \\ -> 单反斜杠


def test_json_unescape_holds_trailing_backslash():
    # 末尾孤立反斜杠应被保留（前缀稳定），等待下一增量
    assert _json_unescape("abc\\") == "abc"
    assert _json_unescape("abc\\\\") == "abc\\"  # 两个反斜杠 = 一个字面反斜杠


def test_json_unescape_unicode():
    assert _json_unescape("\\u4e2d") == "中"
    # 未凑满 4 位十六进制时保留（前缀稳定）
    assert _json_unescape("\\u4e2") == ""


def test_strip_json_closers():
    assert _strip_json_closers('订单..."}') == "订单..."
    assert _strip_json_closers('foo}bar"}') == "foo}bar"  # 答案末尾的 } 保留
    assert _strip_json_closers('say "hi""}') == 'say "hi"'


def test_extract_answer_json():
    assert _extract_answer_json('{"answer": "你好"}') == "你好"
    assert _extract_answer_json('{"answer": "a\\nb"}') == "a\nb"
    assert _extract_answer_json('{"answer": "部分') == "部分"  # 未闭合也能抽取
    assert _extract_answer_json('{"answer": "foo}bar"}') == "foo}bar"


def test_build_prompt_no_history():
    assert _build_prompt("你好", None) == "你好"
    assert _build_prompt("你好", []) == "你好"


def test_build_prompt_with_history():
    history = [
        {"role": "user", "content": "查订单"},
        {"role": "assistant", "content": "已发货"},
    ]
    p = _build_prompt("什么时候到", history)
    assert "【历史对话】" in p
    assert "客户：查订单" in p
    assert "客服：已发货" in p
    assert "【客户最新问题】" in p
    assert p.endswith("什么时候到")


def test_build_prompt_truncates_history():
    # 超过 MAX_HISTORY_TURNS*2 条时只保留最近
    history = [{"role": "user", "content": f"q{i}"} for i in range(100)]
    p = _build_prompt("now", history)
    assert "q0" not in p  # 最老的一条被截掉
    assert "q99" in p
