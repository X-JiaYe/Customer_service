"""父子块构建 §5.7：child chunk（检索用，精准）归并为 parent chunk（注入 LLM 用，上下文完整）。

纯函数，不依赖 ChromaDB / 模型，便于单测。
"""


def build_parent_chunks(child_chunks: list[str], parent_size: int) -> tuple[list[str], list[int]]:
    """把相邻 child chunk 顺序归并为 parent chunk。

    - 每个 parent 累积到不超过 parent_size 字符（单条 child 超长则自成 parent）。
    - 返回 (parents, child_to_parent)：child_to_parent[i] 为第 i 个 child 所属 parent 的下标。
    """
    parents: list[str] = []
    child_to_parent: list[int] = []
    for child in child_chunks:
        if not parents or len(parents[-1]) + len(child) > parent_size:
            parents.append(child)
        else:
            parents[-1] += child
        child_to_parent.append(len(parents) - 1)
    return parents, child_to_parent
