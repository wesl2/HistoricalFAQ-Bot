# PGVector 集成说明

> **状态**: ✅ 完整集成  
> **向量存储**: PostgreSQL + pgvector  
> **不使用**: Chroma/FAISS

---

## ✅ PG 相关内容完整保留

| 文件 | 状态 | 用途 |
|------|------|------|
| `src/vectorstore/pg_pool.py` | ✅ 完整 | 数据库连接池 |
| `src/vectorstore/pg_schema.py` | ✅ 完整 | 表结构定义 |
| `src/vectorstore/pg_indexer.py` | ✅ 完整 | 数据导入 |
| `src/vectorstore/pg_search.py` | ✅ 完整 | 检索 SQL |
| `config/pg_config.py` | ✅ 完整 | 数据库配置 |

---

## 📊 PostgreSQL 表结构

### 1. FAQ 知识表 (`faq_knowledge`)

```sql
CREATE TABLE faq_knowledge (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,              -- 标准问题
    similar_question TEXT NOT NULL,      -- 相似问法
    similar_question_vector vector(1024),-- BGE-M3 向量
    answer TEXT NOT NULL,                -- 答案
    search_vector tsvector,              -- 全文检索
    category VARCHAR(50),                -- 类别
    source_doc VARCHAR(200),             -- 来源文档
    source_page INTEGER,                 -- 页码
    confidence FLOAT DEFAULT 0.9,        -- 置信度
    created_by VARCHAR(50) DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSW 向量索引
CREATE INDEX idx_hnsw_faq ON faq_knowledge
USING hnsw (similar_question_vector vector_cosine_ops);

-- GIN 全文检索索引
CREATE INDEX idx_gin_faq ON faq_knowledge
USING gin (search_vector);
```

**用途**：存储 FAQ 问答对，支持向量相似度检索和全文检索。

---

### 2. 文档片段表 (`doc_chunks`)

```sql
CREATE TABLE doc_chunks (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,           -- 文档片段内容
    chunk_vector vector(1024),          -- BGE-M3 向量
    doc_name VARCHAR(200) NOT NULL,     -- 来源文档名
    doc_page INTEGER,                   -- 页码
    chunk_index INTEGER,                -- 片段序号
    search_vector tsvector,             -- 全文检索
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSW 向量索引
CREATE INDEX idx_hnsw_doc ON doc_chunks
USING hnsw (chunk_vector vector_cosine_ops);
```

**用途**：存储 RAG 文档片段，支持向量检索。

---

### 3. 对话历史表 (`chat_history`)

```sql
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,   -- 会话 ID
    role VARCHAR(20) NOT NULL,          -- 'human' 或 'ai'
    content TEXT NOT NULL,              -- 消息内容
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 会话索引
CREATE INDEX idx_chat_history_session ON chat_history (session_id);
```

**用途**：存储对话历史，支持 LangChain 标准格式。

---

## 🚀 使用方式

### 基本用法

```python
from src.rag.standard_rag import create_standard_rag

# 创建 RAG 系统（使用 PGVector）
rag = create_standard_rag(
    llm_mode="local",
    session_id="user_123",
    retriever_search_type="hybrid"  # "vector", "faq", "hybrid"
)

# 查询（自动从 PG 检索）
answer = rag.query("李世民是谁？")

# 流式查询
for chunk in rag.stream_query("玄武门之变是什么？"):
    print(chunk, end="", flush=True)
```

### 添加数据

```python
from src.rag.standard_rag import create_standard_rag
from langchain_core.documents import Document

rag = create_standard_rag()

# 添加文档（自动导入 PG）
docs = [
    Document(
        page_content="李世民（598 年－649 年），唐朝第二位皇帝...",
        metadata={"question": "李世民是谁？", "source": "李世民传"}
    )
]
rag.add_documents(docs)
```

---

## 🔧 检索类型

| 类型 | 说明 | 用途 |
|------|------|------|
| `"faq"` | 仅 FAQ 检索 | 标准问题快速回答 |
| `"vector"` | 仅文档检索 | RAG 深度检索 |
| `"hybrid"` | 混合检索 | FAQ + 文档融合（推荐） |

---

## 📋 数据导入方式

### 1. FAQ 导入

```bash
# 使用脚本导入 JSONL
python scripts/ingest_data.py data/qa_pairs/wang_faq.jsonl
```

### 2. 文档导入

```bash
# 处理文档并导入 PG
python scripts/process_documents.py data/raw/ --advanced
```

### 3. 代码导入

```python
from src.vectorstore.pg_indexer import FAQIndexer

indexer = FAQIndexer()
count = indexer.index_from_file("data/qa_pairs/wang_faq.jsonl")
print(f"导入 {count} 条记录")
```

---

## 🎯 架构说明

```
用户查询
    ↓
[StandardLLM] (BaseChatModel)
    ↓
[PGVectorRetriever] ← 使用项目自研 pg_search 模块
    ↓
[PostgreSQL + pgvector]
    ├── faq_knowledge 表（FAQ 知识）
    ├── doc_chunks 表（文档片段）
    └── chat_history 表（对话历史）
```

**关键**：完全使用 PostgreSQL + pgvector，不使用 Chroma/FAISS！

---

## ✅ 验证

```python
# 验证 PGVector 集成
from src.rag.standard_retriever import get_pgvector_retriever
from config.pg_config import PG_TABLE_NAME, PG_DOC_TABLE, PG_CHAT_TABLE

print(f'FAQ 表: {PG_TABLE_NAME}')
print(f'文档表: {PG_DOC_TABLE}')
print(f'对话表: {PG_CHAT_TABLE}')

# 输出:
# FAQ 表: faq_knowledge
# 文档表: doc_chunks
# 对话表: chat_history
```

---

## 🎉 总结

| 问题 | 答案 |
|------|------|
| **删除 PG 内容了吗？** | ❌ 没有！全部保留 |
| **使用 Chroma/FAISS 吗？** | ❌ 不使用！只用 PGVector |
| **数据存储在哪里？** | PostgreSQL 3 张表 |
| **向量维度？** | 1024 维（BGE-M3） |
| **检索方式？** | 向量 + 全文 + 混合 |

**你的 PGVector 完整保留，现在标准模块也使用 PGVector！** 
