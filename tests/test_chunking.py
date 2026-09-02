"""测试 knowledge/chunking.py 的父子块构建（纯函数）。"""
from knowledge.chunking import build_parent_chunks


def test_small_children_merge_into_parent():
    children = ["aaa", "bbb", "ccc"]
    parents, mapping = build_parent_chunks(children, parent_size=10)
    assert parents == ["aaabbbccc"]
    assert mapping == [0, 0, 0]


def test_children_split_at_parent_size():
    # parent_size=5：每个 child 长度 3，两条即 6 > 5，故每条自成 parent
    children = ["aaa", "bbb", "ccc"]
    parents, mapping = build_parent_chunks(children, parent_size=5)
    assert parents == ["aaa", "bbb", "ccc"]
    assert mapping == [0, 1, 2]


def test_mixed_grouping():
    # 长度 3+3=6>5 → 前两条合一组会超，故第1条单独；第2、3条 3+3=6>5 也超 → 各单独
    children = ["aaa", "bbb", "cc"]
    parents, mapping = build_parent_chunks(children, parent_size=6)
    # aaa(3)+bbb(3)=6 <=6 合并；+cc(2)=8>6 → cc 另起
    assert parents == ["aaabbb", "cc"]
    assert mapping == [0, 0, 1]


def test_oversized_child_becomes_own_parent():
    children = ["x" * 10, "yy"]
    parents, mapping = build_parent_chunks(children, parent_size=5)
    assert parents == ["x" * 10, "yy"]
    assert mapping == [0, 1]


def test_empty_input():
    assert build_parent_chunks([], 100) == ([], [])
