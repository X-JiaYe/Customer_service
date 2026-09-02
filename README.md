# 智能客服 Agent

基于 **LLM + RAG** 的 B2B 智能客服系统。用户提问后，Agent 自主判断意图并调用对应工具完成回答，支持四大核心能力：

| 能力 | 工具 | 说明 |
|------|------|------|
| 知识库问答 | `knowledge_retriever` | BM25 + 向量混合检索 + Reranker 重排 |
| 订单查询 | `query_order` | 按订单号查询（模拟数据） |
| 工单创建 | `create_ticket` | 生成工单号，写入本地 `tickets.jsonl` |
| 转人工 | `transfer_to_human` | 记录转人工日志并返回提示 |

Agent 使用 **ToolCallingAgent**（模型只做原生函数调用、不执行代码，免代码执行沙箱风险）。

---

## 一、架构：前端 + 后端

- **后端**：FastAPI + SSE 流式响应（`main.py`），三个接口 `/chat`、`/health`、`/knowledge/ingest`。
- **前端（图形）**：Gradio ChatInterface，浏览器里对话，开发调试用。
- **前端（命令行）**：`client.py`，零依赖，默认只打印最终答案。
- **LLM**：DeepSeek（OpenAI 兼容接口，经 LiteLLM 接入）。
- **RAG**：ChromaDB（向量）+ BM25（词法）→ 合并去重 → FlagEmbedding `bge-reranker-v2-m3` 重排。

## 二、技术栈

- Agent 框架：`smolagents`（ToolCallingAgent）
- LLM：DeepSeek（`deepseek-chat`）
- 向量库：ChromaDB（嵌入式，本地持久化）
- Embedding：`BAAI/bge-small-zh-v1.5`
- Reranker：`BAAI/bge-reranker-v2-m3`
- 混合检索：`rank-bm25` + ChromaDB 向量 + FlagEmbedding 重排
- 后端：FastAPI + Uvicorn（SSE 流式）
- 前端：Gradio ChatInterface
- 部署：Docker Compose

## 三、项目结构

```
customer-service/
├── main.py                 # FastAPI 入口（SSE 流式 + Gradio）
├── agent.py                # ToolCallingAgent 初始化 + 工具注册 + 流式抽取
├── client.py               # 命令行客户端（零依赖，减 token 输出）
├── config.py               # 集中配置（读 .env）
├── tools/
│   ├── rag_retriever.py    # 混合检索工具（BM25 + 向量 + Reranker）
│   ├── order_api.py        # 订单查询工具（模拟 API）
│   ├── ticket_api.py       # 工单创建工具
│   └── transfer_human.py   # 转人工工具
├── knowledge/
│   ├── ingest.py           # 文档导入：文件 → 分块 → Embedding → ChromaDB
│   ├── docs/               # 知识库文档（.txt / .md / .pdf）
│   └── chroma_db/          # ChromaDB 持久化目录（自动生成，勿手动改）
├── tests/                  # pytest 测试
├── requirements.txt        # 运行时依赖（版本已钉死）
├── requirements-dev.txt    # 开发/测试依赖
├── Dockerfile
├── docker-compose.yml
└── .env                    # 密钥与配置（不入库）
```

## 四、快速开始

### 方式 A：本地运行（开发调试推荐）

**前置条件**：Python 3.12 64-bit、DeepSeek API Key。

```bash
cd customer-service

# 1) 创建虚拟环境（务必用 Python 3.12）
python -m venv .venv
# Windows 下之后一律用 .venv/Scripts/python.exe，避免误用系统 Python

# 2) 安装依赖（国内可加 -i 换源加速）
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 3) 配置环境变量：复制 .env 并填入 DeepSeek Key
#    （项目已含 .env 模板，把 DEEPSEEK_API_KEY 改成你的真实 key）

# 4) 导入知识库（首次或文档有更新时执行）
.venv/Scripts/python.exe -m knowledge.ingest

# 5) 启动后端 API（默认端口 8000）
.venv/Scripts/python.exe main.py --mode api

# 或者启动图形界面（Gradio，浏览器打开）
.venv/Scripts/python.exe main.py --mode gradio
```

> **注意**：必须用 `.venv/Scripts/python.exe`（Python 3.12 64-bit），系统 Python 3.9（32-bit）会因加载不了 64 位包而报错。

### 方式 B：Docker 一键启动（部署推荐）

```bash
cd customer-service

# 首次会构建镜像（国内已内置镜像加速：Docker Hub 镜像 + 阿里云 PyPI + CPU 版 torch）
docker compose up -d --build

# 查看健康状态
curl http://127.0.0.1:8000/health
# 预期返回 {"status":"ok"}
```

> **国内网络前置**：如果构建时 `registry-1.docker.io` 连接失败，需给 Docker Desktop 配置 registry 镜像（编辑 `~/.docker/daemon.json` 增加 `registry-mirrors` 后重启 Docker Desktop）。

## 五、使用方式

### 0. 浏览器网页（用户侧前端 §6.1）

启动 API 后浏览器打开 `http://127.0.0.1:8000/` 即是聊天页（`static/index.html`，轻量无构建，可直接替换为 Vue+Vite）。支持：

- **流式渲染**：答案逐 token 实时显示（SSE）；
- **引用点击**：答案末尾附「来源/参考」的行会被抽出为可点击复制块；
- **转人工**：一键发送转人工请求（走 `transfer_to_human` 工具）；
- **满意度**：每条答案可 👍/👎，POST 到 `/feedback` 沉淀为 `feedback:satisfaction`。

网页右上角可填 `X-API-Key`（留空 = 开发模式）；会话 id 存于浏览器 `sessionStorage`，刷新保持同一会话。

### 1. 图形界面（Gradio）

启动后浏览器打开 Gradio 给的地址（默认 `http://127.0.0.1:7860`），直接对话即可。

### 2. 命令行客户端（推荐脚本调用）

```bash
cd customer-service

# 单轮问答（默认只打印最终答案）
.venv/Scripts/python.exe client.py "你们的产品支持哪些支付方式？"

# 多轮会话（--session 保持上下文记忆）
.venv/Scripts/python.exe client.py "查订单 TK20250301 的状态" --session demo
.venv/Scripts/python.exe client.py "那它的物流到哪了？" --session demo

# 健康检查 / 重新导入知识库
.venv/Scripts/python.exe client.py --health
.venv/Scripts/python.exe client.py --ingest

# 想看逐 token 流式输出时加 --stream
.venv/Scripts/python.exe client.py "..." --stream
```

### 3. 直接调 API

```bash
# 非流式：只返回最终答案
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你们的产品支持哪些支付方式？","stream":false}'

# 流式：SSE 逐 token 推送
curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我查订单 TK20250301","stream":true}'
```

SSE 流带 `id:` 事件序号 + 心跳注释行（长回答期间防空闲超时）；断线后带 `Last-Event-ID: <最后收到的 id>` 重连即可**断点续传**（在 `SSE_STREAM_TTL_SECONDS` 窗口内从缓冲重放，不重复生成）。

**Token 成本控制（§6.4）**：
- **语义缓存**（`cache.py`）：单轮（无历史）相同问法命中缓存直接复用答案，省一次 LLM 调用。归一化（去空白/标点/大小写）后「支持哪些付款方式？」与「支持哪些付款方式！」命中同一缓存；命中时响应带 `cached: true`。缓存答案带 TTL（`CACHE_TTL_SECONDS`）。
- **成本记账 + 预算告警**（`budget.py`）：按租户核算每小时 LLM 调用次数，超 `BUDGET_MAX_CALLS_PER_HOUR` 时降级（返回话术，不再调用 LLM），响应带 `budget_exceeded: true`。
- 命中/超预算均有 Prometheus 指标（`cs_cache_hits_total`）。当前以「调用次数」计，精确 token 计量待接 LLM 返回元数据（见 `metrics.observe_tokens` 待办）。

**多渠道接入（§6.2）**：`channels/` 定义跨渠道统一消息模型（`Message`）与企业 IM 适配器（企业微信 / 钉钉 / 飞书的回调解析与回复格式化），agent 核心不感知渠道差异。REST `/chat`、Webhook `/webhook/{channel}`、WebSocket `/ws` 三条通道共用同一套会话/缓存/预算/审计逻辑（`_try_fast_path` + `_process_message`）。企业 IM 的**签名校验**留接口 `channels.webhook.verify_signature`（当前开发模式放行，生产按平台 secret 补齐）。

### 4. 多租户（租户机器人）

项目支持**多租户知识隔离**：每个租户持有一把独立 API Key，登录后只能检索「本租户专属知识 + 全局共享知识」，跨租户知识不可见。

#### 租户 ↔ API Key 对照

| API Key | 租户 | 可见知识 |
|---|---|---|
| `sk-local-dev` | `default`（开发/默认） | 共享知识（FAQ / 产品手册 / 用户指南） |
| `sk-huayuan` | `tenant_a`（华远科技） | 共享知识 + 《华远科技_园区安防方案》 |
| `sk-lanjing` | `tenant_b`（蓝鲸制造） | 共享知识 + 《蓝鲸制造_产线维保手册》 |

#### 用不同租户身份调用

```bash
# 华远科技（tenant_a）：问门禁/安防，命中专属方案
.venv/Scripts/python.exe client.py "门禁访客预约怎么操作？" --api-key sk-huayuan

# 蓝鲸制造（tenant_b）：问维保/备件，命中专属手册
.venv/Scripts/python.exe client.py "冲压线多久巡检一次？" --api-key sk-lanjing

# 用 A 租户的 key 问 B 租户的内容 → 只回共享知识，不泄露蓝鲸
.venv/Scripts/python.exe client.py "产线设备维保巡检" --api-key sk-huayuan
```

```bash
# 直接调 API 同理，带 X-API-Key 头
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-huayuan" \
  -d '{"message":"门禁访客预约怎么操作？","stream":false}'
```

前端登录页填对应 API Key 即可（网页右上角 `X-API-Key`，留空 = 开发模式 `default`）。

#### 新增一个租户

1. 在 `knowledge/docs/` 放该租户的专属文档，并在 `manifest.json` 里给它加 `"tenant": "tenant_x"`。
2. 在 `.env` 的 `API_KEYS` 里加一条映射 `"sk-xxx":"tenant_x"`。
3. 重新导入：`.venv/Scripts/python.exe -m knowledge.ingest`。

> 不带 `tenant` 的文档 = **共享知识**，对所有租户可见；`tenant` 指定的文档仅该租户可见。

## 六、API 接口

| 接口 | 方法 | 请求体 | 响应 |
|------|------|--------|------|
| `/` | GET | — | 用户侧聊天页（`static/index.html`） |
| `/health` | GET | — | `{"status":"ok"}` |
| `/knowledge/ingest` | POST | — | `{"status":"ok","message":"..."}` |
| `/chat` | POST | `{"message": "...", "session_id": "default", "stream": true}` | 见下 |
| `/feedback` | POST | `{"rating":"up/down","session_id":"...","question":"","answer":""}` | `{"status":"ok"}` |
| `/webhook/{channel}` | POST | 企业 IM 回调原始 payload | 平台约定回复体 |
| `/ws` | WebSocket | `{"message":"...","session_id":"default"}` | `{"answer":"..."}` |

`/chat` 响应：
- `stream=true`（默认）：`text/event-stream`，逐 token 推送 `data: {"delta":"..."}`，结束推 `data: {"done":true}`。
- `stream=false`：JSON `{"answer":"...","session_id":"..."}`，一次返回最终答案。

`/webhook/{channel}` 支持 `wecom` / `dingtalk` / `feishu`：把各平台文本消息回调规整为统一消息、处理后按平台格式回复（如钉钉 `{"msgtype":"text","text":{"content":...}}`）；非文本消息回「仅支持文本消息」。

`/ws` 鉴权走请求头 `X-API-Key` 或查询参数 `api_key`；当前返回完整答案（非流式），多轮会话用 `session_id` 保持上下文。

## 七、配置说明（`.env` 关键项）

| 变量 | 说明 | 默认 |
|------|------|------|
| `DEEPSEEK_API_KEY` | **必填**，DeepSeek API Key | — |
| `DEEPSEEK_BASE_URL` | DeepSeek 接口地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 模型名 | `deepseek-chat` |
| `EMBEDDING_MODEL` | 向量模型 | `BAAI/bge-small-zh-v1.5` |
| `RERANKER_MODEL` | 重排模型 | `BAAI/bge-reranker-v2-m3` |
| `ENABLE_RERANKER` | 是否启用重排（1/0） | `1` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 分块大小 / 重叠 | `500` / `50` |
| `RETRIEVE_TOP_K` / `RERANK_TOP_N` | 粗排数 / 精排取 N | `20` / `5` |
| `HF_ENDPOINT` | HuggingFace 镜像（国内加速） | `https://hf-mirror.com` |
| `HF_HUB_DISABLE_XET` | 禁用 Xet 存储（大文件走普通 HTTP，避开镜像 401） | `1` |

> **Reranker 说明**：`bge-reranker-v2-m3` 约 2.3GB，首次检索才下载（国内较慢）。知识库较小时建议设 `ENABLE_RERANKER=0` 跳过，BM25+向量两路已够用；文档上量后再开启。

## 八、更新知识库

把新的 `.txt` / `.md` / `.pdf` 放进 `knowledge/docs/`，然后：

```bash
.venv/Scripts/python.exe -m knowledge.ingest      # 本地
# 或
docker compose exec app python -m knowledge.ingest # Docker 内
```

导入是**增量**的（按「文件名 + 修改时间 + 生命周期元数据」判断，未变化自动跳过）。

### 知识生命周期治理（§5.1）

`knowledge/docs/manifest.json` 声明每个文档的生命周期元数据（未声明的走默认值：已发布 / 永不过期 / 无 owner / v1.0.0）：

```json
{
  "FAQ_常见问题与售后服务.md": {
    "owner": "客服运营组",
    "status": "approved",
    "valid_until": "2027-12-31",
    "version": "1.2.0"
  },
  "华远科技_园区安防方案.md": {
    "owner": "华远科技",
    "status": "approved",
    "version": "1.0.0",
    "tenant": "tenant_a"
  }
}
```

- `status`：`draft`（草稿）→ `pending`（待审）→ `approved`（已发布）→ `deprecated`（已下架）。
- `valid_until`：ISO 日期，到期后自动**不再被检索**（到期下架）。
- `tenant`：可选，多租户专属文档的归属租户；缺省 = **共享知识**（所有租户可见）。详见「五、使用方式 → 4. 多租户」。
- 只有 `approved` 且未过期的知识才会进入检索；`draft`/`pending`/`deprecated` 一律过滤。
- 每次入库追加一条 `knowledge/ingest_history.jsonl`，可用 `lifecycle.list_history(path, source)` 回溯任意文档的历史版本。

### 长上下文优化（父子块 §5.7）

导入时把文档切成 **child chunk**（检索用，精准）并归并为 **parent chunk**（注入 LLM 用，上下文完整，`CHUNK_PARENT_SIZE` 控制）。检索只走 child，命中的 child 会展开为 parent 完整上下文，再经**近重复去重 + 按预算裁剪**（`knowledge/context.py`）组装，替代原来的「简单拼接 + 硬截断」，避免超长文档/跨段推理时上下文被拦腰截断。

### 备份与容灾（§5.8）

- 知识库（ChromaDB）持久化在 `knowledge/chroma_db/`（Docker 内已 bind mount 到宿主机）；Redis 会话/审计开启 **AOF 持久化**，异常宕机最多丢 1 秒写。
- 快照 / 恢复 / 重建（`python -m knowledge.backup --help`）：
  ```bash
  .venv/Scripts/python.exe -m knowledge.backup --backup                 # 快照向量库
  .venv/Scripts/python.exe -m knowledge.backup --restore backup/<快照>   # 回滚
  .venv/Scripts/python.exe -m knowledge.backup --rebuild                # 从 docs 全量重导（灾难恢复兜底）
  .venv/Scripts/python.exe -m knowledge.backup --verify                 # 恢复后自检
  ```

## 九、测试

```bash
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest -q
```

### RAG 检索评测（Recall@K / MRR，§5.4）

知识库更新后，用种子评测集离线验证检索质量（`eval/seed_dataset.json`，query → 期望文档）：

```bash
.venv/Scripts/python.exe -m eval.rag_eval                     # 默认评测集 + K=5,10
.venv/Scripts/python.exe -m eval.rag_eval --k 5 10 20         # 自定义 K
ENABLE_RERANKER=0 .venv/Scripts/python.exe -m eval.rag_eval   # 跳过重排，仅 BM25+向量
```

输出 Recall@K / MRR 与未命中样本（query → 期望 → 实际 top5），供补充知识或调参参考。

## 十、常见问题

1. **系统 Python 报 `cannot import name '_imaging' from PIL`**：用了 32 位 Python 3.9，请改用 `.venv/Scripts/python.exe`（Python 3.12 64-bit）。
2. **Docker 构建 `registry-1.docker.io` 连接失败**：国内网络需给 Docker Desktop 配 registry 镜像。
3. **首次启动慢 / RAG 首答慢**：冷启动要加载 embedding 模型（约十几秒）；reranker 2.3GB 首次下载较慢，可先 `ENABLE_RERANKER=0`。
4. **Windows 控制台中文乱码**：本项目已在 `config.py` / `client.py` 强制 UTF-8 输出，若仍有乱码请确认终端编码为 UTF-8。
5. **进程退出时报 `ResourceTracker.__del__` 错误**：Windows/Python3.12 + `multiprocess` 库的已知良性噪声，不影响结果。
6. **下载 reranker 时报 `401 Unauthorized`（`cas-server.xethub.hf.co`）**：大模型文件走 HuggingFace 的 Xet 存储，`hf-mirror.com` 不代理其 CAS 接口导致回落真实域名鉴权失败。在 `.env` 里设 `HF_HUB_DISABLE_XET=1`（项目已默认配置）即可改走普通 HTTP 下载。

## 十一、验收场景

启动后发送以下问题，Agent 应调用对应工具：

| 问题 | 预期调用工具 |
|------|--------------|
| 你们的产品支持哪些支付方式？ | `knowledge_retriever` |
| 帮我查一下订单 TK20250301 的状态 | `query_order` |
| 我要投诉，帮我转人工 | `transfer_to_human` |
| 帮我提一个工单，问题是页面加载很慢 | `create_ticket` |
