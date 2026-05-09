# HistoricalFAQ-Bot 项目完整结构

## 项目概述

基于 RAG 架构的历史人物智能问答系统，融合 FAQ 精准问答和文档 RAG 检索，支持本地/API 双模 LLM。

## 目录结构

```
HistoricalFAQ-Bot/
├── README.md                          # 项目说明
├── PROJECT_STRUCTURE.md               # 本文件：项目结构说明
├── requirements.txt                   # Python 依赖
├── docker-compose.yml                 # Docker 部署配置
│
├── config/                            # 配置模块
│   ├── __init__.py
│   ├── pg_config.py                   # PostgreSQL 配置
│   ├── model_config.py                # 模型配置（Embedding/LLM/Reranker）
│   └── retrieval_config.py            # 检索策略配置
│
├── src/                               # 源代码
│   ├── __init__.py
│   │
│   ├── api/                           # API 接口层
│   │   └── main.py                    # FastAPI 主应用
│   │
│   ├── chat/                          # 对话引擎
│   │   ├── __init__.py
│   │   ├── chat_engine.py             # 核心对话流程
│   │   ├── response_generator.py      # 回答生成器
│   │   └── demo.py                    # CLI 演示脚本
│   │
│   ├── data_pipeline/                 # 数据处理流水线
│   │   └── qa_transformer.py          # RAG → FAQ 格式转换
│   │
│   ├── embedding/                     # 向量化模块
│   │   ├── __init__.py
│   │   └── embedding_local.py         # BGE-M3 本地推理
│   │
│   ├── llm/                           # LLM 双模架构
│   │   ├── __init__.py
│   │   ├── base_llm.py                # LLM 抽象基类
│   │   ├── local_llm.py               # 本地模型（Qwen）
│   │   ├── api_llm.py                 # API 模型（DeepSeek）
│   │   └── llm_factory.py             # LLM 工厂（单例模式）
│   │
│   ├── retrieval/                     # 检索策略
│   │   ├── __init__.py
│   │   ├── faq_retriever.py           # FAQ 检索器
│   │   ├── doc_retriever.py           # 文档 RAG 检索器
│   │   └── search_router.py           # 检索路由器（混合策略）
│   │
│   └── vectorstore/                   # 向量存储
│       ├── __init__.py
│       ├── pg_pool.py                 # 连接池单例
│       ├── pg_schema.py               # 数据库表结构
│       ├── pg_indexer.py              # 数据索引器
│       └── pg_search.py               # 混合检索 SQL
│
├── scripts/                           # 工具脚本
│   ├── init_db.py                     # 数据库初始化
│   └── ingest_data.py                 # 数据导入
│
├── data/                              # 数据目录
│   ├── raw/                           # 原始文档（PDF/TXT）
│   ├── processed/                     # 清洗后数据
│   ├── qa_pairs/                      # FAQ 问答对
│   └── finetune/                      # 微调数据
│
├── tutorial/                          # 教程文档
│   └── HistoricalFAQBot-完整项目实战课程.md   # 完整教程
│
└── tests/                             # 测试目录（待创建）
    ├── test_chat_engine.py
    └── performance_test.py
```

## 核心设计模式

### 1. 单例模式 - 连接池 (pg_pool.py)
```python
_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(...)
    return _pool
```

### 2. 工厂模式 - LLM 切换 (llm_factory.py)
```python
class LLMFactory:
    @staticmethod
    def create_llm(mode: str = None):
        if mode == "local":
            return LocalLLM()
        elif mode == "api":
            return APILLM()
```

### 3. 策略模式 - 检索路由 (search_router.py)
```python
class SearchRouter:
    def search(self, query: str) -> SearchContext:
        # 根据置信度选择不同策略
        # FAQ_ONLY / HYBRID / RAG
```

## 数据库表结构

### 1. faq_knowledge - FAQ 主表
```sql
- id: SERIAL PRIMARY KEY
- question: TEXT                    # 标准问题
- similar_question: TEXT            # 相似问法
- similar_question_vector: vector   # BGE-M3 向量
- answer: TEXT                      # 答案
- search_vector: tsvector           # 全文检索
- category: VARCHAR(50)             # 类别
- source_doc: VARCHAR(200)          # 来源文档
- source_page: INTEGER              # 页码
- confidence: FLOAT                 # 可信度
- created_by: VARCHAR(50)           # 创建方式
- created_at / updated_at: TIMESTAMP
```

### 2. doc_chunks - 文档片段表
```sql
- id: SERIAL PRIMARY KEY
- chunk_text: TEXT                  # 文本片段
- chunk_vector: vector              # 向量
- doc_name: VARCHAR(200)            # 文档名
- doc_page: INTEGER                 # 页码
- chunk_index: INTEGER              # 片段序号
- search_vector: tsvector           # 全文检索
```

### 3. chat_history - 对话历史表
```sql
- id: SERIAL PRIMARY KEY
- session_id: VARCHAR(100)          # 会话 ID
- user_query: TEXT                  # 用户问题
- retrieved_faq_ids: INTEGER[]      # 检索到的 FAQ IDs
- retrieved_doc_ids: INTEGER[]      # 检索到的文档 IDs
- llm_response: TEXT                # LLM 回答
- llm_mode: VARCHAR(20)             # local/api
- created_at: TIMESTAMP
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
export PG_URL="postgresql://user:pass@localhost:5432/faq_db"
export LLM_MODE="local"

# 3. 初始化数据库
python scripts/init_db.py

# 4. 转换并导入数据
python src/data_pipeline/qa_transformer.py wang_qa.jsonl wang_faq.jsonl
python scripts/ingest_data.py data/qa_pairs/wang_faq.jsonl

# 5. 启动服务
python src/chat/demo.py              # CLI 模式
python -m src.api.main               # API 模式

# 6. Docker 部署
docker-compose up -d
```

## 关键配置

### 环境变量
```bash
# 数据库
PG_URL=postgresql://user:pass@host:5432/db

# LLM 模式
LLM_MODE=local  # 或 api

# API 密钥（使用 API 模式时）
API_KEY=your-api-key
API_PROVIDER=deepseek  # openai/claude/deepseek/zhipu
```

### 模型路径配置 (config/model_config.py)
```python
EMBEDDING_CONFIG = {
    "model_path": "/root/autodl-tmp/models/bge-m3-finetuned-wang",
    "device": "cuda"
}

LLM_CONFIG = {
    "local": {
        "model_path": "/root/autodl-tmp/models/qwen/Qwen1.5-7B-Chat"
    },
    "api": {
        "provider": "deepseek",
        "api_key": "${API_KEY}"
    }
}
```

## 检索策略

```
用户查询
    ↓
FAQ 检索（相似度计算）
    ↓
├── 相似度 > 0.90 → FAQ_ONLY（直接返回答案）
├── 相似度 0.85-0.90 → HYBRID（FAQ + 文档融合）
└── 相似度 < 0.85 → RAG（纯文档检索 + LLM 生成）
```

## 性能指标

- 向量维度：1024 (BGE-M3)
- 响应时间：< 3 秒（平均）
- 并发连接：10（连接池最大）
- 支持的 FAQ 数量：无上限（取决于数据库）

## 扩展建议

1. **添加新历史人物**
   - 准备新的 QA 数据
   - 运行 qa_transformer.py 转换
   - 运行 ingest_data.py 导入

2. **微调 Embedding**
   - 使用 FlagEmbedding 训练
   - 更新 EMBEDDING_CONFIG["model_path"]

3. **切换 LLM**
   - 设置 LLM_MODE=api
   - 配置 API_KEY

## 调试技巧

```python
# 查看 SQL 执行
import logging
logging.basicConfig(level=logging.DEBUG)

# 测试向量检索
python -c "
from src.embedding import get_embedding
from src.vectorstore import FAQSearcher
vec = get_embedding('测试问题')
results = FAQSearcher().search(vec, top_k=3)
for r in results:
    print(f'{r.score:.3f}: {r.question}')
"
```

## 常见问题

**Q: 连接池创建失败？**
A: 检查 PG_URL 环境变量，确保 PostgreSQL 服务已启动。

**Q: 模型加载缓慢？**
A: 首次加载会下载模型，后续使用缓存。或使用本地路径。

**Q: 检索结果为空？**
A: 检查数据是否已导入，向量是否正确计算。

**Q: 如何切换 LLM？**
A: 设置环境变量 `export LLM_MODE=api`，然后重启服务。

## 贡献指南

1. Fork 项目
2. 创建分支 (`git checkout -b feature/xxx`)
3. 提交更改 (`git commit -am 'Add xxx'`)
4. 推送分支 (`git push origin feature/xxx`)
5. 创建 Pull Request

## License

MIT License
