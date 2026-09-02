"""安全兜底：PII 脱敏 + 输出审核规则（纯函数、零重依赖，可独立单测）。

设计原则：
- 脱敏 mask_pii：对手机号/邮箱/身份证/银行卡做正则打码，用于答案后处理与历史持久化。
- 审核 check_output_safety：识别「承诺词 / 价格 / 时限 / 免责词」，返回结构化命中项。
  只做「命中识别」，是否拦截由调用方决定（agent 的 final_answer_checks 拦截无来源依据的
  价格/时限/强承诺；main.py 对残留命中项附加官方口径提示）。
- 明确不做：第三方内容审核（黄反/违法违规）——留 vendor 接口，本次仅规则版。
"""
import re

# ---------- PII 正则 ----------
# 手机号：1[3-9] 开头 11 位
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# 身份证：18 位（末位可能为 X/x）。须在银行卡之前匹配，避免 18 位银行卡被误判为身份证
_IDCARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
# 银行卡：16~19 位纯数字
_BANKCARD_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")
# 邮箱
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# ---------- 输出审核规则 ----------
# 强承诺词：客服场景下无依据即风险较高，参与「无来源即拦截」
_STRONG_PROMISE_RE = re.compile(r"保证|绝对|100%|百分百|最快|独家|唯一|承诺")
# 软承诺词：仅打标提示，不参与硬拦截（“一定/肯定/包”在正常语境常见，误伤率高）
_SOFT_PROMISE_RE = re.compile(r"一定|肯定")
# 具体价格（金额 + 货币单位）
_PRICE_RE = re.compile(r"(?:人民币|￥|¥|\$|USD)?\s*\d+(?:\.\d+)?\s*(?:元|块钱|块|美元|美金|欧元|日元)")
# 具体时限承诺（数字 + 时间单位 + 动作）
_TIME_RE = re.compile(r"\d+\s*(?:天|小时|分钟|个工作日|日内)(?:内)?(?:完成|解决|处理|到账|发货|回复|退款|到货)")
# 免责/推责敏感词
_DISCLAIMER_RE = re.compile(r"概不负责|免责|不承担|与我司无关|不负责任")


def _mask_by_pos(s: str, head: int, tail: int) -> str:
    """保留首 head 位、尾 tail 位，中间打星号。"""
    if len(s) <= head + tail:
        return "*" * len(s)
    return s[:head] + "*" * (len(s) - head - tail) + s[-tail:]


def _mask_email(addr: str) -> str:
    local, _, domain = addr.partition("@")
    if len(local) <= 1:
        masked_local = "*"
    else:
        masked_local = local[0] + "*" * (len(local) - 1)
    return f"{masked_local}@{domain}"


def mask_pii(text: str) -> str:
    """对常见 PII 做正则打码。无命中则原样返回。"""
    if not text:
        return text
    # 顺序敏感：身份证(18位) 在银行卡(16~19位) 之前
    text = _PHONE_RE.sub(lambda m: _mask_by_pos(m.group(0), 3, 4), text)
    text = _IDCARD_RE.sub(lambda m: _mask_by_pos(m.group(0), 6, 4), text)
    text = _BANKCARD_RE.sub(lambda m: _mask_by_pos(m.group(0), 4, 4), text)
    text = _EMAIL_RE.sub(lambda m: _mask_email(m.group(0)), text)
    return text


def _snippet(text: str, pos: int, width: int = 12) -> str:
    """截取命中词附近上下文，便于日志/审计定位。"""
    return text[max(0, pos - width): pos + width]


def check_output_safety(text: str) -> list[dict]:
    """识别输出中的敏感承诺，返回结构化命中项列表。

    返回：[
        {"type": "强承诺" | "软承诺" | "价格" | "时限" | "免责词", "word": ..., "context": ...},
        ...
    ]
    """
    hits: list[dict] = []
    if not text:
        return hits

    for m in _STRONG_PROMISE_RE.finditer(text):
        hits.append({"type": "强承诺", "word": m.group(0), "context": _snippet(text, m.start())})
    for m in _SOFT_PROMISE_RE.finditer(text):
        hits.append({"type": "软承诺", "word": m.group(0), "context": _snippet(text, m.start())})
    for m in _PRICE_RE.finditer(text):
        hits.append({"type": "价格", "word": m.group(0).strip(), "context": _snippet(text, m.start())})
    for m in _TIME_RE.finditer(text):
        hits.append({"type": "时限", "word": m.group(0).strip(), "context": _snippet(text, m.start())})
    for m in _DISCLAIMER_RE.finditer(text):
        hits.append({"type": "免责词", "word": m.group(0), "context": _snippet(text, m.start())})
    return hits
