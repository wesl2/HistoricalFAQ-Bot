# Historical Chat Bot

基于 RAG 架构的历史文献智能问答系统。上传 EPUB 历史文献，AI 基于文献内容生成回答，并附带可溯源的引用来源。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 核心功能

- **📚 RAG 文档检索** — 基于 BM25 + 向量混合检索，RRF 融合排序，从入库文献中精准召回相关内容
- **🔗 引用溯源与校验** — 每条 AI 回答底部附带可折叠的引用来源面板，点击 `[n]` 高亮对应文献片段，引用编号经一致性校验
- **⚡ 流式输出** — SSE 流式传输，首 token 响应快，打字机效果逐字呈现
- **📤 EPUB 上传入库** — 用户可直接上传 EPUB 文献，自动解析、语义切分、向量化入库，即刻可检索
- **💬 对话历史持久化** — 基于 UUID 的会话管理，对话记录存储于 PostgreSQL，支持历史会话回看与删除
- **🧠 Query Rewriting** — 自动指代消解（如"他"→"唐太宗"），多轮对话上下文连贯
- **🎨 单文件前端** — Tailwind CSS + 原生 JavaScript，零构建依赖，侧边栏/底部输入区均支持拖拽调整尺寸

## 技术架构

```
┌─────────────┐      HTTP/SSE       ┌─────────────────────────────┐
│  frontend   │ ◄─────────────────► │  FastAPI (Uvicorn)          │
│  (单文件)    │                     │  - ChatEngine               │
└─────────────┘                     │  - ResponseGenerator        │
                                    │  - SearchRouter             │
                                    └─────────────┬───────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    ▼                             ▼                             ▼
            ┌──────────────┐           ┌──────────────────┐           ┌─────────────┐
            │  PostgreSQL  │           │   DeepSeek API   │           │   BGE-M3    │
            │  + pgvector  │           │   (LLM)          │           │  (Embedding)│
            └──────────────┘           └──────────────────┘           └─────────────┘
```

## 技术栈

| 层级 | 技术 |
|------|------|
| **后端** | FastAPI + Python 3.10+ |
| **前端** | HTML5 + Tailwind CSS (CDN) + JavaScript（单文件 `frontend.html`） |
| **数据库** | PostgreSQL 14+ + pgvector 扩展 |
| **嵌入模型** | BGE-M3（1024 维，支持中英文混合） |
| **LLM** | DeepSeek API（支持本地 Qwen / API 双模式切换） |
| **检索** | BM25 全文检索 + 向量相似度检索，RRF 融合排序 |
| **文档处理** | PDR 语义切分、EPUB 解析 |
| **部署** | Docker / Uvicorn / systemd / screen |

## 项目结构

```
HistoricalFAQ-Bot/
├── src/
│   ├── api/              # FastAPI 服务入口 (main.py)
│   ├── chat/             # 对话引擎 (ChatEngine, ResponseGenerator)
│   ├── data_pipeline/    # 数据处理管道 (PDR 语义切分、EPUB 解析)
│   ├── embedding/        # BGE-M3 嵌入模型封装
│   ├── llm/              # LLM 调用层 (本地/API 双模式)
│   ├── rag/              # RAG 链与回调
│   ├── retrieval/        # 检索系统 (BM25、向量、混合、RRF、SearchRouter)
│   ├── tools/            # 工具函数
│   ├── monitoring/       # 可观测性（Langfuse 追踪预留）
│   └── vectorstore/      # PostgreSQL 连接池与数据访问
├── config/               # 配置文件 (数据库、模型、检索参数)
├── data/
│   ├── raw/              # 原始 EPUB 文献
│   ├── processed/        # 处理后数据
│   ├── qa_pairs/         # QA 对（已弃用，全部走 RAG）
│   └── finetune/         # 微调数据
├── scripts/              # 数据入库与评测脚本 (ingest_documents.py, eval_ragas.py)
├── prompts/              # Prompt 模板 (RAG、多查询、对话)
├── logs/                 # 运行日志
├── frontend.html         # 前端界面（单文件，零构建）
├── requirements.txt      # Python 依赖
├── Dockerfile            # Docker 构建
├── docker-compose.yml    # Docker Compose 配置
└── service_start.md      # 服务启动指南
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/wesl2/HistoricalFAQ-Bot.git
cd HistoricalFAQ-Bot

# 创建虚拟环境（推荐）
conda create -n RAG python=3.10 -y
conda activate RAG

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# DeepSeek API Key（必填）
export API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 系统编码（防止中文乱码）
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# LLM 模式：api 或 local（默认 api）
export LLM_MODE=api
```

### 3. 初始化数据库

确保 PostgreSQL 已安装 pgvector 扩展，并创建数据库：

```bash
python scripts/init_db.py
```

### 4. 启动服务

```bash
# 开发模式（热重载）
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 生产后台运行（screen）
screen -dmS faq-bot bash -c 'export API_KEY="..."; uvicorn src.api.main:app --host 0.0.0.0 --port 8000'
```

详见 [`service_start.md`](./service_start.md)。

### 5. 访问系统

- **前端界面**：http://localhost:8000/frontend.html
- **API 文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/api/health

## API 接口

### 标准问答

```http
POST /api/query
Content-Type: application/json

{
  "question": "唐太宗的用人之道有什么特点？",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "history": []
}
```

### 流式问答（SSE）

```http
POST /api/query/stream
Content-Type: application/json

{
  "question": "贞观之治的主要措施有哪些？",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "stream": true
}
```

### 上传 EPUB 文献

```http
POST /api/upload
Content-Type: multipart/form-data

file: <your-epub-file>
```

### 已入库文档列表

```http
GET /api/documents
```

### 健康检查

```http
GET /api/health
```

## Docker 部署

```bash
# 构建并启动（包含 PostgreSQL）
docker-compose up -d

# 查看日志
docker-compose logs -f api
```

## 前端特性

| 特性 | 说明 |
|------|------|
| 会话管理 | 左侧栏显示历史会话，支持新建、切换、删除 |
| 流式输出 | SSE 实时逐字渲染，带打字机光标 |
| 引用溯源 | 每条 AI 消息底部 `📚 引用来源 (n)` 折叠面板，含书名、章节、摘要 |
| EPUB 上传 | 底部上传按钮，进度提示，入库后自动刷新 BM25 索引 |
| 拖拽调整 | 左侧会话栏横向拖拽（200~400px），底部输入区竖直拖拽（150~500px） |
| 提示卡片 | 右侧显示使用流程与收录书目，可折叠收起 |

## 数据示例

当前已入库文献：

- 《唐太宗传》(赵克尧、许道勋)
- 《剑桥中国隋唐史：589-906 年》(费正清 等)

> 支持用户上传更多 EPUB 历史文献扩展知识库。

## 核心设计亮点

1. **引用编号重映射** — `_format_citation_footer()` 确保每条回答的引用编号从 `[1]` 开始，避免跨消息混乱
2. **异常分层映射** — `chat_engine.py` catch 所有异常返回 `error_code`；`main.py` 映射为 HTTP 状态码；全局处理器兜底
3. **依赖注入** — `FAQRetriever`/`DocRetriever` 通过构造函数接收 `embedding_fn`，避免循环依赖
4. **连接池** — `psycopg2.pool.ThreadedConnectionPool(min=1, max=10)`，同步操作通过 `asyncio.to_thread()` 隔离
5. **限流保护** — `asyncio.Semaphore` 限制 LLM 并发，防止打爆 API

## 路线图 / 预留功能

以下功能已预留接口和脚本位置，欢迎贡献实现：

| 功能 | 路径 | 说明 |
|------|------|------|
| **vLLM 本地推理** | `src/llm/vllm_engine.py` | 包装 vLLM 实现本地 Qwen 高并发推理，替代现有本地模式，支持 Continuous Batching 与多卡并行 |
| **Langfuse 可观测性** | `src/monitoring/langfuse_tracker.py` | 集成 Langfuse 追踪检索→生成的全链路，覆盖延迟、Token 消耗、文档召回质量、用户评分反馈 |
| **RAGAS 自动评测** | `scripts/eval_ragas.py` | 基于 RAGAS 框架评测 faithfulness、answer_relevancy、context_recall 等指标，生成 HTML 评测报告 |

## 常见问题

### Q1: 模型加载失败？
检查 `API_KEY` 环境变量是否设置正确，或模型路径是否存在于本地模式。

### Q2: 数据库连接失败？
确保 PostgreSQL 服务运行且已启用 pgvector 扩展，检查 `config/pg_config.py` 中的连接参数。

### Q3: 中文乱码？
```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
```

### Q4: 如何生产部署？
购买带公网 IP 的云服务器，Nginx 反向代理 + HTTPS，systemd 或 Docker 持久化运行。详见 `service_start.md`。

## 许可证

MIT License
