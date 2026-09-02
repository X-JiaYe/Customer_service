"""集中配置管理：从 .env 读取环境变量，提供项目级配置常量。"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Windows 控制台默认 GBK，强制 UTF-8 输出，避免 smolagents/rich 打印特殊字符（如 •）时崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE_DIR = Path(__file__).resolve().parent

# 加载项目根目录的 .env（无论从哪个 cwd 启动都能读到）
load_dotenv(BASE_DIR / ".env")

# ---------- DeepSeek LLM ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
# smolagents 的 LiteLLMModel 通过 litellm 的 OpenAI 兼容模式接入 DeepSeek
LITELLM_MODEL_ID = f"openai/{DEEPSEEK_MODEL}"

# ---------- Embedding / Reranker 模型 ----------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# ---------- 知识库 / 检索参数 ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
# 父子块 §5.7：child chunk 用于检索（精准），parent chunk 用于注入 LLM（上下文更完整）
CHUNK_PARENT_SIZE = int(os.getenv("CHUNK_PARENT_SIZE", "1500"))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "20"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
# 是否启用 Reranker 重排（1=启用；0=关闭，可跳过约 2.3GB 的 reranker 模型下载）
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "1") == "1"

# ---------- Agent ----------
MAX_STEPS = 5
# 已迁移到 ToolCallingAgent（模型只做原生函数调用，不执行代码），此配置已废弃保留
EXECUTOR_TYPE = os.getenv("EXECUTOR_TYPE", "local")
# 多轮对话注入上下文的最多轮数（每轮=用户+客服各一条）
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))

# ---------- 服务 / 会话 ----------
# 启动时是否自动导入知识库（空库时生效，幂等）
AUTO_INGEST = os.getenv("AUTO_INGEST", "1") == "1"
# 会话空闲过期秒数（超时清理，防内存泄漏）
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
# 单次注入 LLM 的检索结果字符上限（防上下文超长）
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "3000"))

# ---------- Redis / 会话 / 审计 ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# 是否记录审计日志（业务动作留痕，写 Redis）
AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "1") == "1"

# ---------- 鉴权 / 多租户 / 限流 ----------
# API_KEYS: JSON 映射 {"api_key": "tenant_id"}；为空 → 开发模式（不鉴权，tenant=default）
API_KEYS = json.loads(os.getenv("API_KEYS", "{}") or "{}")
# 每个 租户+IP+接口 每分钟请求上限
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_WINDOW_SECONDS = 60

# ---------- 安全兜底 ----------
# RAG 置信度门槛：低于此值判定为「低置信度」，返回不确定 + 建议转人工，不硬编
RAG_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.0"))
# 是否对输出做 PII 脱敏（手机号/邮箱/身份证/银行卡）
ENABLE_PII_MASK = os.getenv("ENABLE_PII_MASK", "1") == "1"

# ---------- 业务对接（mock / 生产区分） ----------
# ENV: dev=业务工具用 mock 数据；prod=必须接真实业务接口（mock 路径显式报错）
ENV = os.getenv("ENV", "dev")

# ---------- 可靠性 / 熔断降级（§5.9） ----------
# LLM 单次调用超时（秒）；透传给 LiteLLM 的 timeout
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
# LLM 单次调用失败后的最大重试次数（LiteLLM num_retries，仅对瞬时错误重试）
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
# 熔断器：连续失败达阈值 → 打开；冷却期后进入半开
CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3"))
CIRCUIT_RESET_SECONDS = float(os.getenv("CIRCUIT_RESET_SECONDS", "60"))

# ---------- 流式断线重连（§6.3） ----------
# SSE 心跳间隔（秒）：长回答期间无 token 产出时发送注释行，防代理空闲超时
SSE_HEARTBEAT_SECONDS = float(os.getenv("SSE_HEARTBEAT_SECONDS", "15"))
# 断点续传缓冲 TTL（秒）：断线后在此窗口内重连可重放，超时则需重新生成
SSE_STREAM_TTL_SECONDS = float(os.getenv("SSE_STREAM_TTL_SECONDS", "120"))

# ---------- Token 成本控制（§6.4） ----------
# 语义缓存：相同问法（归一化后）复用缓存答案，省 LLM 调用
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
# 成本记账 + 预算告警：按租户核算每小时代价（以 LLM 调用次数计，token 精确计量待接 LLM 元数据）
BUDGET_ENABLED = os.getenv("BUDGET_ENABLED", "1") == "1"
BUDGET_MAX_CALLS_PER_HOUR = int(os.getenv("BUDGET_MAX_CALLS_PER_HOUR", "1000"))

# ---------- 路径（绝对化，避免 cwd 差异） ----------
CHROMA_PERSIST_DIR = str((BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "./knowledge/chroma_db")).resolve())
DOCS_DIR = str((BASE_DIR / "knowledge" / "docs").resolve())
CHROMA_COLLECTION = "customer_service_kb"
# 用户侧前端（§6.1）：轻量静态页，无构建，直接由 FastAPI 托管
STATIC_DIR = str((BASE_DIR / "static").resolve())

# ---------- 其他 ----------
TICKETS_FILE = str((BASE_DIR / "tickets.jsonl").resolve())
