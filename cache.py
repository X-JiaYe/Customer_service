"""语义缓存 §6.4：相同问法直接复用缓存答案，省 LLM 调用与 token。

- `normalize`：去空白/标点/大小写，使「支持哪些付款方式？」「支持哪些付款方式」「支持哪些付款方式!」
  命中同一缓存（同一问题不同问法/标点/输入法全角差异不重复调用 LLM）。
- `get_answer` / `set_answer`：Redis 读写，fail-open（Redis 不可用时不阻塞主流程）。
- 纯函数 `normalize` / `_key` 可单测。

局限（README 待办）：当前为「归一化精确命中」；「相似问法」（换词但语义相同）需 embedding
相似度匹配，归入后续升级（复用 RAG 的 BGE 向量做近邻比对）。
"""
import hashlib
import json
import re

import config
from store import get_redis

_PREFIX = "cache:"
# 去掉空白与中英文标点/全角符号，再统一小写，作为缓存键的归一化输入
_PUNCT = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’()（）\[\]【】…·～~]+")


def normalize(question: str) -> str:
    """归一化问题（去空白/标点/大小写），作为缓存键；空串也返回空串。"""
    return _PUNCT.sub("", question).lower()


def _key(tenant: str, question: str) -> str:
    digest = hashlib.md5(normalize(question).encode("utf-8")).hexdigest()
    return f"{_PREFIX}{tenant}:{digest}"


def get_answer(tenant: str, question: str):
    """命中缓存返回答案字符串，未命中返回 None（Redis 异常也返回 None，fail-open）。"""
    if not config.CACHE_ENABLED:
        return None
    try:
        raw = get_redis().get(_key(tenant, question))
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


def set_answer(tenant: str, question: str, answer: str) -> None:
    """写入缓存（带 TTL）；Redis 异常静默忽略（fail-open）。"""
    if not config.CACHE_ENABLED:
        return
    try:
        get_redis().set(
            _key(tenant, question),
            json.dumps(answer, ensure_ascii=False),
            ex=config.CACHE_TTL_SECONDS,
        )
    except Exception:  # noqa: BLE001
        pass
