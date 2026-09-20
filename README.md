# 电商内容 RAG 知识库

基于 **LLM + RAG** 的跨境电商内部知识问答系统。员工提问后，Agent 从内部知识库（岗位职责、客服 FAQ、运营手册、亚马逊/独立站运营细则）中检索相关内容作答，帮助员工**尽职尽责、分工明确**。

| 能力 | 说明 |
|------|------|
| 知识库问答 | `knowledge_retriever`：BM25（词法）+ 向量（语义）混合检索，命中片段附「【来源】」标注 |
| 低置信度拒答 | 检索结果与问题相关性不足时明确告知「暂不确定」，不硬编 |
| 不编造纪律 | 无来源依据不报具体价格/时限/数字，规避绝对化承诺 |

知识问答采用 **检索 + 生成（retrieve-then-generate）**：先做 BM25+向量混合检索得到上下文，再把「上下文 + 问题」一次性交给 LLM 生成答案。不依赖模型的 function calling 能力，单次请求只打一次 LLM。

> 定位：**最小 MVP**。单进程、内存会话、单工具 RAG 问答，零外部依赖（无需 Docker/Redis）。已移除多租户、SSE 流式、Redis、语义缓存、预算、审计、指标、多渠道、多业务工具、熔断器等非核心层。

---

## 一、技术栈

- 生成：`smolagents` 的 `LiteLLMModel`（检索 + 生成，单次 LLM 调用）
- LLM：智谱 GLM（`glm-4-flash`，OpenAI 兼容，经 LiteLLM 接入）
- 向量库：ChromaDB（嵌入式，本地持久化）
- Embedding：`BAAI/bge-small-zh-v1.5`
- 混合检索：`rank-bm25` + ChromaDB 向量（Reranker 默认关闭，`ENABLE_RERANKER=0`）
- 后端：FastAPI + Uvicorn（非流式）
- 前端：Gradio ChatInterface（调试）+ `client.py`（命令行）

## 二、项目结构

```
customer-service/
├── main.py                 # FastAPI 入口（/chat + /health + /knowledge/ingest）
├── agent.py                # 检索 + 生成：RAG 检索器 + LLM 生成
├── client.py               # 命令行客户端（零依赖，只打印最终答案）
├── config.py               # 集中配置（读 .env）
├── auth.py                 # 单 API Key 鉴权（可选，留空 = 免鉴权）
├── store.py                # 进程内内存会话存储
├── security.py             # PII 脱敏 + 输出安全兜底
├── tools/
│   └── rag_retriever.py    # 混合检索工具（BM25 + 向量）
├── knowledge/
│   ├── ingest.py           # 文档导入：文件 → 分块 → Embedding → ChromaDB
│   ├── chunking.py         # 父子块构建
│   ├── context.py          # 上下文组装（去重 + 预算裁剪）
│   ├── embedding.py        # BGE Embedding 函数
│   ├── lifecycle.py        # 知识生命周期（status / valid_until / version）
│   ├── backup.py           # 向量库快照 / 恢复 / 重建
│   └── docs/               # 知识库文档（.md / .txt / .pdf）
├── eval/                   # RAG 检索离线评测（Recall@K / MRR）
├── tests/                  # pytest 测试
├── requirements.txt        # 运行时依赖（版本已钉死）
├── Dockerfile
├── docker-compose.yml
└── .env                    # 密钥与配置（不入库）
```

## 三、快速开始

### 方式 A：本地运行（开发调试推荐）

**前置条件**：Python 3.12 64-bit、智谱 API Key。

```bash
cd customer-service

# 1) 创建虚拟环境（务必用 Python 3.12）
python -m venv .venv
# Windows 下之后一律用 .venv/Scripts/python.exe，避免误用系统 Python

# 2) 安装依赖（国内可加 -i 换源加速）
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 3) 配置环境变量：把 .env 里 LLM_API_KEY 改成你的真实智谱 Key
#    （项目已含 .env 模板，见 .env.example）

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

## 四、使用方式

### 1. 图形界面（Gradio）

启动后浏览器打开 Gradio 给的地址（默认 `http://127.0.0.1:7860`），直接对话即可。

### 2. 命令行客户端（推荐脚本调用）

```bash
cd customer-service

# 单轮问答（默认只打印最终答案）
.venv/Scripts/python.exe client.py "各部门岗位职责怎么划分？"

# 多轮会话（--session 保持上下文记忆）
.venv/Scripts/python.exe client.py "亚马逊 FBA 备货要提前多久下单？" --session demo
.venv/Scripts/python.exe client.py "那断货率红线是多少？" --session demo

# 健康检查 / 重新导入知识库
.venv/Scripts/python.exe client.py --health
.venv/Scripts/python.exe client.py --ingest
```

### 3. 直接调 API

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"退换货政策是什么？"}'

# 配置了 API_KEY 时，带上 X-API-Key 头
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"message":"各部门岗位职责怎么划分？"}'
```

## 五、知识库文档

`knowledge/docs/` 下为知识库内容（面向员工的跨境电商运营规范）：

| 文档 | 负责人 | 说明 |
|------|--------|------|
| 员工岗位职责与分工手册.md | 人事行政部 | 组织架构、岗位分工、勤勉履职与 KPI |
| 客服部_售前售后与争议处理FAQ.md | 客服部 | 售前/物流/退换/争议处理与响应 SLA |
| 运营部_店铺日常运营操作手册.md | 运营部 | 每日例行、选品上架、广告、活动复盘 |
| 亚马逊运营组_美国站运营细则.md | 亚马逊运营组 | FBA 备货、ACoS 红线、类目合规 |
| 独立站运营组_DTC品牌站运营细则.md | 独立站运营组 | 投放 ROI、SEO、EDM、支付物流 |

每个文档在 `manifest.json` 中声明生命周期元数据（`owner`/`status`/`version`）。

## 六、API 接口

| 接口 | 方法 | 请求体 | 响应 |
|------|------|--------|------|
| `/health` | GET | — | `{"status":"ok"}` |
| `/chat` | POST | `{"message":"...","session_id":"default"}` | `{"answer":"...","session_id":"..."}` |
| `/knowledge/ingest` | POST | — | `{"status":"ok","message":"..."}` |

`/chat` 为非流式：一次性返回最终答案。

## 七、配置说明（`.env` 关键项）

| 变量 | 说明 | 默认 |
|------|------|------|
| `LLM_API_KEY` | **必填**，智谱 API Key | — |
| `LLM_BASE_URL` | 智谱接口地址 | `https://open.bigmodel.cn/api/paas/v4/` |
| `LLM_MODEL` | 模型名 | `glm-4-flash` |
| `EMBEDDING_MODEL` | 向量模型 | `BAAI/bge-small-zh-v1.5` |
| `ENABLE_RERANKER` | 是否启用重排（1/0，默认关闭） | `0` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 分块大小 / 重叠 | `500` / `50` |
| `RETRIEVE_TOP_K` / `RERANK_TOP_N` | 粗排数 / 精排取 N | `20` / `5` |
| `API_KEY` | 单值鉴权 Key（空 = 免鉴权） | — |
| `RAG_CONFIDENCE_THRESHOLD` | 低置信度门槛 | `0.5` |
| `ENABLE_PII_MASK` | 输出 PII 脱敏（1/0） | `1` |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | LLM 超时（秒）/ 重试次数 | `60` / `2` |
| `HF_ENDPOINT` / `HF_HUB_DISABLE_XET` | HuggingFace 镜像 / 禁 Xet | `hf-mirror.com` / `1` |

## 八、更新知识库

把新的 `.txt` / `.md` / `.pdf` 放进 `knowledge/docs/`，然后：

```bash
.venv/Scripts/python.exe -m knowledge.ingest      # 本地
# 或
docker compose exec app python -m knowledge.ingest # Docker 内
```

导入是**增量**的（按「文件名 + 修改时间 + 生命周期元数据」判断，未变化自动跳过）。

### 知识生命周期治理

`knowledge/docs/manifest.json` 声明每个文档的生命周期元数据（未声明的走默认值：已发布 / 永不过期 / 无 owner / v1.0.0）：

```json
{
  "员工岗位职责与分工手册.md": {
    "owner": "人事行政部",
    "status": "approved",
    "valid_until": "2027-12-31",
    "version": "2.0.0"
  }
}
```

- `status`：`draft`（草稿）→ `pending`（待审）→ `approved`（已发布）→ `deprecated`（已下架）。
- `valid_until`：ISO 日期，到期后自动**不再被检索**（到期下架）。
- 只有 `approved` 且未过期的知识才会进入检索；`draft`/`pending`/`deprecated` 一律过滤。
- 每次入库追加一条 `knowledge/ingest_history.jsonl`，可用 `lifecycle.list_history(path, source)` 回溯任意文档的历史版本。

### 长上下文优化（父子块）

导入时把文档切成 **child chunk**（检索用，精准）并归并为 **parent chunk**（注入 LLM 用，上下文完整，`CHUNK_PARENT_SIZE` 控制）。检索只走 child，命中的 child 会展开为 parent 完整上下文，再经**近重复去重 + 按预算裁剪**（`knowledge/context.py`）组装，避免超长文档/跨段推理时上下文被拦腰截断。

### 备份与容灾

知识库（ChromaDB）持久化在 `knowledge/chroma_db/`（Docker 内已 bind mount 到宿主机）：

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

### RAG 检索评测（Recall@K / MRR）

知识库更新后，用种子评测集离线验证检索质量（`eval/seed_dataset.json`，query → 期望文档）：

```bash
.venv/Scripts/python.exe -m eval.rag_eval                     # 默认评测集 + K=5,10
.venv/Scripts/python.exe -m eval.rag_eval --k 5 10 20         # 自定义 K
```

输出 Recall@K / MRR 与未命中样本（query → 期望 → 实际 top5），供补充知识或调参参考。

## 十、常见问题

1. **系统 Python 报 `cannot import name '_imaging' from PIL`**：用了 32 位 Python 3.9，请改用 `.venv/Scripts/python.exe`（Python 3.12 64-bit）。
2. **首次启动慢 / RAG 首答慢**：冷启动要加载 embedding 模型（约十几秒）。
3. **Windows 控制台中文乱码**：本项目已在 `config.py` / `client.py` 强制 UTF-8 输出，若仍有乱码请确认终端编码为 UTF-8。
4. **进程退出时报 `ResourceTracker.__del__` 错误**：Windows/Python3.12 + `multiprocess` 库的已知良性噪声，不影响结果。
5. **会话在重启后丢失**：会话存于进程内内存，重启即清空（单实例 MVP 的取舍，后续多实例可接回外部存储）。

## 十一、验收场景

启动后发送以下问题，Agent 应调用 `knowledge_retriever` 检索作答：

| 问题 | 预期来源文档 |
|------|--------------|
| 各部门岗位职责怎么划分？ | 员工岗位职责与分工手册.md |
| 退换货政策是什么？ | 客服部_售前售后与争议处理FAQ.md |
| 亚马逊 ACoS 红线是多少？ | 亚马逊运营组_美国站运营细则.md |
| 独立站广告 ROAS 目标是多少？ | 独立站运营组_DTC品牌站运营细则.md |
