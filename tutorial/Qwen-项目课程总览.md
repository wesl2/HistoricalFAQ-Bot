# HistoricalFAQ-Bot 智能问答系统 - 完整项目实战课程

> 本课程将带你从零开始构建一个基于 RAG 架构的历史人物智能 FAQ 问答系统。
> 
> **技术栈**: FastAPI + LangChain + PostgreSQL/pgvector + BGE-M3 + BCE-Reranker + Qwen/DeepSeek
> 
> **适合人群**: 想学习 RAG 架构、向量数据库、大模型应用开发的同学

---

## 📚 课程总览

| 课时 | 主题 | 时长 | 核心产出 | 难度 |
| --- | --- | --- | --- | --- |
| 第 1 课 | 项目架构与环境搭建 | 2-3h | 项目结构 + 依赖配置 | ⭐ |
| 第 2 课 | PostgreSQL + pgvector 部署 | 3-4h | 数据库配置 + 连接测试 | ⭐⭐ |
| 第 3 课 | Embedding 向量化模块 | 4-5h | embedding_local.py | ⭐⭐ |
| 第 4 课 | 数据库表设计与初始化 | 4-5h | pg_schema.py + pg_pool.py | ⭐⭐ |
| 第 5 课 | FAQ 检索模块实现 | 5-6h | faq_retriever.py | ⭐⭐⭐ |
| 第 6 课 | 文档 RAG 检索模块 | 5-6h | doc_retriever.py | ⭐⭐⭐ |
| 第 7 课 | 混合检索策略路由 | 4-5h | search_router.py (核心) | ⭐⭐⭐⭐ |
| 第 8 课 | **标准 LLM 架构** | 5-6h | **standard_llm.py** (重构) | ⭐⭐⭐ |
| 第 9 课 | 对话引擎与回答生成 | 5-6h | chat_engine.py | ⭐⭐⭐⭐ |
| 第 10 课 | FastAPI 后端服务开发 | 4-5h | main.py + API 接口 | ⭐⭐⭐ |
| 第 11 课 | Web 前端界面实现 | 3-4h | frontend.html | ⭐⭐ |
| 第 12 课 | **LangChain 标准集成** | 5-6h | **standard_*.py** (重构) | ⭐⭐⭐⭐ |
| 第 13 课 | **RRF 融合排序** | 5-6h | **PGVectorRetriever** (重构) | ⭐⭐⭐⭐ |
| 第 14 课 | 流式输出与 SSE | 3-4h | standard_streaming.py | ⭐⭐⭐ |
| 第 15 课 | Callback 可观测性系统 | 3-4h | callbacks.py | ⭐⭐⭐ |
| 第 16 课 | Prompt 外部化管理 | 2-3h | prompts/模板系统 | ⭐⭐ |
| 第 17 课 | 数据处理流水线 | 4-5h | qa_transformer.py | ⭐⭐⭐ |
| 第 18 课 | 数据导入与测试 | 3-4h | 完整测试流程 | ⭐⭐ |
| 第 19 课 | Docker 容器化部署 | 3-4h | Dockerfile + docker-compose | ⭐⭐⭐ |
| 第 20 课 | 性能优化与面试准备 | 3-4h | 优化方案 + 面试 Q&A | ⭐⭐⭐ |

**总课时**: 20 课 | **预计总时长**: 80-100 小时

---

## 🎯 学习目标

完成本课程后，你将能够：

1. ✅ 理解 RAG 架构的核心原理和实现流程
2. ✅ 掌握向量数据库 (pgvector) 的使用和优化
3. ✅ 实现混合检索策略 (FAQ + 文档 RAG + **RRF 融合排序**)
4. ✅ 构建 LLM 双模架构 (本地 + API)，**使用 LangChain 标准接口**
5. ✅ 集成 LangChain 生态 (LCEL、Multi-Query、Rerank、Streaming)
6. ✅ **从自定义桥接迁移到标准架构**（公司级最佳实践）
7. ✅ 开发完整的 FastAPI 后端服务
8. ✅ 实现可观测性系统 (日志、监控、Token 统计)
9. ✅ 掌握 Docker 容器化部署流程
10. ✅ 获得一个完整的项目经历，适合求职面试

---

# 第 1 课：项目架构与环境搭建

## 🎯 目标

理解 RAG 架构，搭建项目基础结构

## ⏱️ 时间：2-3 小时

## 📖 理论学习 (30 分钟)

### RAG 架构核心流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  用户提问   │ ──→ │ 文本向量化  │ ──→ │  混合检索   │ ──→ │  返回答案   │
│  "王洪文是  │     │  Embedding  │     │ FAQ + 文档  │     │  LLM 生成   │
│   谁？"     │     │  (1024 维)   │     │  策略路由   │     │  带引用     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           ↓                    ↓
                    ┌─────────────┐     ┌─────────────┐
                    │ BGE-M3 模型 │     │ PostgreSQL  │
                    │ 本地推理    │     │ + pgvector  │
                    └─────────────┘     └─────────────┘
```

### 为什么用 RAG？

| 方案 | 优点 | 缺点 |
|------|------|------|
| 纯 LLM | 通用性强 | 幻觉严重、知识滞后 |
| 纯检索 | 答案准确 | 灵活性差 |
| **RAG** | **准确 + 灵活** | **实现复杂** |

### 技术栈选型

| 组件 | 技术 | 理由 |
|------|------|------|
| 后端框架 | FastAPI | 异步高性能、自动文档 |
| 向量数据库 | PostgreSQL + pgvector | 部署简单、支持事务、数据统一 |
| Embedding | BGE-M3 | 中文 SOTA、支持微调 |
| LLM | Qwen/DeepSeek/**通义千问**/Kimi | 双模架构、5+ 厂商兼容 |
| 检索策略 | **RRF 融合排序** | 跨源去重、工业级实践 |
| 前端 | HTML5 + Tailwind | 轻量、无需构建 |
| 部署 | Docker | 一键部署、环境隔离 |

## 💻 代码复现

### 1. 创建项目目录结构

```bash
# 创建根目录
mkdir -p HistoricalFAQ-Bot/{config,src,data,scripts,prompts,vectorstore,logs,tutorial}

# 创建 src 子模块
mkdir -p HistoricalFAQ-Bot/src/{api,chat,data_pipeline,embedding,llm,rag,retrieval,vectorstore,tools}

# 创建数据子目录
mkdir -p HistoricalFAQ-Bot/data/{raw,processed,qa_pairs,finetune}

# 创建__init__.py
find HistoricalFAQ-Bot/{config,src,src/*} -type d -exec touch {}/__init__.py \;
```

### 2. 最终目录结构（已更新）

> **重要说明**：以下目录结构反映当前代码状态，部分旧文件（如 `base_llm.py`、`langchain_integration.py`）已被废弃并删除。

```
HistoricalFAQ-Bot/
├── config/                    # 配置模块
│   ├── __init__.py
│   ├── pg_config.py          # 数据库配置（PGVector、连接池）
│   ├── model_config.py       # 模型配置（LLM、Embedding、BM25）
│   └── retrieval_config.py   # 检索配置
│
├── src/                       # 源代码
│   ├── api/                  # FastAPI 服务
│   │   └── main.py           # 后端 API（含流式 SSE）
│   ├── chat/                 # 对话引擎
│   │   ├── chat_engine.py    # 对话引擎（兼容新旧模式）
│   │   └── response_generator.py
│   ├── data_pipeline/        # 数据处理
│   │   ├── document_processor.py  # 文档加载和分块
│   │   └── qa_transformer.py      # QA 格式转换
│   ├── embedding/            # 向量化
│   │   ├── embedding_local.py           # BGE-M3 封装
│   │   └── embedding_local_practice.py  # 练习版本
│   ├── llm/                  # LLM 标准接口（已重构）
│   │   ├── standard_llm.py   # ✅ 标准 LangChain 接口
│   │   ├── local_llm.py.deleted      # ❌ 已废弃
│   │   ├── api_llm.py.deleted        # ❌ 已废弃
│   │   └── llm_factory.py.deleted    # ❌ 已废弃
│   ├── rag/                  # 标准 RAG 模块（已重构）
│   │   ├── standard_rag.py           # ✅ 标准 RAG 统一入口
│   │   ├── standard_chain.py         # ✅ LCEL Chain 构建
│   │   ├── standard_retriever.py     # ✅ PGVector 检索器（含 RRF）
│   │   ├── standard_memory.py        # ✅ 对话历史管理
│   │   ├── standard_streaming.py     # ✅ 流式输出实现
│   │   ├── callbacks.py              # 可观测性回调
│   │   └── langchain_integration.py.deleted  # ❌ 已废弃
│   ├── retrieval/            # 检索策略
│   │   ├── search_router.py          # 三级检索路由
│   │   ├── faq_retriever.py          # FAQ 检索
│   │   ├── doc_retriever.py          # 文档检索
│   │   └── bm25_retriever.py         # BM25 全文检索
│   ├── tools/                # Agent 工具
│   │   └── tools.py
│   └── vectorstore/          # 向量存储（PostgreSQL + pgvector）
│       ├── pg_pool.py        # 连接池（单例模式）
│       ├── pg_pool_practice.py
│       ├── pg_schema.py      # 表结构定义
│       ├── pg_schema_practice.py
│       ├── pg_indexer.py     # 数据导入
│       └── pg_search.py      # 检索实现（FAQSearcher、HybridSearcher）
│
├── prompts/                   # 提示词模板
│   ├── rag_template.txt
│   ├── conversational_template.txt
│   └── multi_query_template.txt
│
├── data/                      # 数据目录
│   ├── raw/                  # 原始文档
│   ├── processed/            # 处理后数据
│   ├── qa_pairs/             # FAQ 问答对
│   └── finetune/             # 微调数据
│
├── scripts/                   # 工具脚本
│   ├── init_db.py
│   ├── ingest_data.py
│   └── start_server.py
│
├── tutorial/                  # 教程文档
│   ├── Qwen-项目课程总览.md
│   ├── Qwen-第 6-10 课.md
│   ├── Qwen-第 11-15 课.md
│   └── Qwen-第 16-20 课.md
│
├── frontend.html              # Web 界面
├── docker-compose.yml         # Docker 部署
├── Dockerfile
├── requirements.txt           # 依赖
├── README.md                  # 项目文档
├── PROJECT_STRUCTURE.md       # 项目结构说明
├── PGVector 集成说明.md        # PGVector 集成文档
├── 标准 LangChain 迁移指南.md   # 迁移指南
└── BM25功能说明.md            # BM25 功能文档
```

### 📁 关键变更说明

| 旧文件/模块 | 状态 | 替代方案 |
|------------|------|----------|
| `src/llm/base_llm.py` | ❌ 已删除 | 使用 `langchain_core.BaseChatModel` |
| `src/llm/local_llm.py` | ❌ 已删除 | 使用 `standard_llm.py` + `HuggingFacePipeline` |
| `src/llm/api_llm.py` | ❌ 已删除 | 使用 `standard_llm.py` + `ChatOpenAI` |
| `src/llm/llm_factory.py` | ❌ 已删除 | 使用 `StandardLLM.create()` |
| `src/rag/langchain_integration.py` | ❌ 已废弃 | 拆分为 `standard_*.py` 模块 |
| `src/rag/advanced_retriever.py` | ⚠️ 已备份 | 使用 `standard_retriever.py` (PGVector) |
| Chroma/FAISS 向量库 | ❌ 已废弃 | 使用 PostgreSQL + pgvector |

### 3. 编写 requirements.txt

```txt
# Web 框架
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-multipart==0.0.6

# LangChain 生态
langchain==0.2.0
langchain-core==0.2.0
langchain-community==0.2.0
langchain-huggingface==0.0.3
langchain-text-splitters==0.2.0

# 向量数据库
chromadb==0.4.24
faiss-cpu==1.7.4

# 深度学习
torch==2.1.0
transformers==4.35.0
sentence-transformers==2.2.2
BCEmbedding==0.1.0

# PostgreSQL
psycopg2-binary==2.9.9
pgvector==0.2.4

# 工具库
numpy==1.24.3
loguru==0.7.2
requests==2.31.0
PyPDF2==3.0.1

# 测试
pytest==7.4.3
```

### 4. 安装依赖

```bash
cd HistoricalFAQ-Bot

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import fastapi; import torch; import psycopg2; print('✅ 依赖安装成功')"
```

## 📝 学习要点

1. **RAG 的核心价值**: 避免大模型幻觉，基于真实知识库回答
2. **为什么选 pgvector**: 
   - 部署简单（一个 PostgreSQL）
   - 支持 ACID 事务
   - 运维成本低（无需额外组件）
3. **双模 LLM 架构**: 本地模式省钱，API 模式省心

## ✅ 检查项

- [ ] 目录结构完整
- [ ] 依赖安装成功
- [ ] 能解释 RAG 流程
- [ ] 理解 pgvector 的优势

---

# 第 2 课：PostgreSQL + pgvector 部署

## 🎯 目标

部署数据库，理解向量存储基础

## ⏱️ 时间：3-4 小时

## 📖 理论学习 (45 分钟)

### 向量数据库核心概念

| 概念 | 说明 |
|------|------|
| **向量** | 文本的高维数学表示 (如 1024 维浮点数) |
| **相似度** | 余弦相似度衡量语义相近程度 (-1 到 1) |
| **向量索引** | HNSW/IVF 等算法加速检索 |

### pgvector 核心 SQL

```sql
-- 1. 启用扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 创建带向量字段的表
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    name TEXT,
    embedding vector(1024)  -- 1024 维向量
);

-- 3. 创建向量索引 (HNSW)
CREATE INDEX idx_hnsw 
ON items 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. 向量检索 (<=> 距离越小越相似)
SELECT * FROM items 
ORDER BY embedding <=> '[0.1,0.2,...]'::vector 
LIMIT 10;

-- 5. 带过滤的检索
SELECT * FROM items 
WHERE category = 'history'
ORDER BY embedding <=> '[0.1,0.2,...]'::vector 
LIMIT 10;
```

### 距离度量对比

| 操作符 | 距离类型 | 公式 | 适用场景 |
|--------|----------|------|----------|
| `<->` | L2 距离 | $\sqrt{\sum(x-y)^2}$ | 通用 |
| `<=>` | 余弦距离 | $1 - \frac{x·y}{||x||·||y||}$ | **推荐** |
| `<+>` | 内积距离 | $-x·y$ | 归一化向量 |

## 💻 代码复现

### 1. Docker 部署 PostgreSQL + pgvector

```bash
# 拉取镜像
docker pull ankane/pgvector:latest

# 启动容器
docker run -d \
  --name postgres \
  -e POSTGRES_USER=faq_user \
  -e POSTGRES_PASSWORD=faq_password \
  -e POSTGRES_DB=faq_db \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  ankane/pgvector:latest

# 验证
docker exec -it postgres psql -U faq_user -d faq_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 2. 本地安装 (可选)

```bash
# Ubuntu
sudo apt install postgresql-15-pgvector

# 或从源码编译
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 3. 数据库连接测试

```python
# test_db.py
import psycopg2
from psycopg2.extras import execute_values

# 连接配置
config = {
    "host": "localhost",
    "port": 5432,
    "user": "faq_user",
    "password": "faq_password",
    "database": "faq_db"
}

# 测试连接
conn = psycopg2.connect(**config)
cursor = conn.cursor()

# 启用扩展
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# 创建测试表
cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_items (
        id SERIAL PRIMARY KEY,
        name TEXT,
        embedding vector(3)
    );
""")

# 插入测试数据
cursor.execute("""
    INSERT INTO test_items (name, embedding) VALUES
    ('苹果', '[0.1, 0.2, 0.3]'),
    ('香蕉', '[0.4, 0.5, 0.6]'),
    ('橙子', '[0.7, 0.8, 0.9]');
""")

# 向量检索
cursor.execute("""
    SELECT name, embedding <=> '[0.2, 0.3, 0.4]'::vector AS distance
    FROM test_items
    ORDER BY distance
    LIMIT 2;
""")

print("检索结果:")
for row in cursor.fetchall():
    print(f"  {row[0]}: 距离={row[1]:.4f}")

# 清理
cursor.execute("DROP TABLE test_items;")
conn.commit()
cursor.close()
conn.close()

print("\n✅ 数据库测试成功!")
```

## 📝 学习要点

1. **为什么用余弦相似度**: 对向量长度不敏感，适合文本语义
2. **HNSW 索引参数**:
   - `m`: 节点连接数，越大召回率越高 (推荐 16)
   - `ef_construction`: 构建范围，越大质量越高 (推荐 64)
3. **向量归一化**: 归一化后，余弦相似度 = 内积

## ✅ 检查项

- [ ] PostgreSQL + pgvector 运行正常
- [ ] 能执行向量创建和检索
- [ ] 理解 HNSW 索引原理

---

# 第 3 课：Embedding 向量化模块

## 🎯 目标

实现文本向量化功能，理解 BGE-M3 模型

## ⏱️ 时间：4-5 小时

## 📖 理论学习 (1 小时)

### Embedding 模型原理

```
文本 → Tokenizer → Transformer → [CLS] 向量 → L2 归一化 → 1024 维向量
```

### BGE-M3 模型特点

| 特性 | 说明 |
|------|------|
| **维度** | 1024 维 |
| **最大长度** | 8192 tokens |
| **语言** | 多语言 (中文优化) |
| **类型** | Dense Retrieval |
| **微调支持** | ✅ 支持 |

### 为什么选 BGE-M3？

1. **中文 SOTA**: 中文检索任务表现最佳
2. **长文本支持**: 8192 tokens，适合文档
3. **可微调**: 针对特定领域优化
4. **开源免费**: 无商业限制

## 💻 代码复现

### 1. 创建配置模块 (config/model_config.py)

```python
# -*- coding: utf-8 -*-
"""模型配置模块"""

import os

# Embedding 模型配置
EMBEDDING_CONFIG = {
    # 模型路径 (微调后)
    "model_path": os.getenv(
        "EMBEDDING_MODEL_PATH",
        "/root/autodl-tmp/models/bge-m3-finetuned-wang"
    ),
    
    # 计算设备
    "device": os.getenv("EMBEDDING_DEVICE", "cuda"),
    
    # 最大序列长度
    "max_length": int(os.getenv("EMBEDDING_MAX_LENGTH", "512")),
    
    # 批处理大小
    "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
    
    # 是否使用 FP16
    "use_fp16": os.getenv("EMBEDDING_USE_FP16", "true").lower() == "true",
    
    # 向量维度
    "vector_dim": 1024,
    
    # 是否归一化
    "normalize": True
}
```

### 2. 实现向量化模块 (src/embedding/embedding_local.py)

```python
# -*- coding: utf-8 -*-
"""本地 Embedding 模型封装"""

import logging
import torch
import torch.nn.functional as F
from typing import Union, List
from transformers import AutoTokenizer, AutoModel
from config.model_config import EMBEDDING_CONFIG

logger = logging.getLogger(__name__)

# 全局模型实例 (单例模式)
_tokenizer = None
_model = None
_device = None


def _load_model():
    """延迟加载模型"""
    global _tokenizer, _model, _device
    
    if _model is None:
        logger.info(f"加载 Embedding 模型：{EMBEDDING_CONFIG['model_path']}")
        
        _device = torch.device(EMBEDDING_CONFIG['device'])
        
        _tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_CONFIG['model_path']
        )
        
        _model = AutoModel.from_pretrained(
            EMBEDDING_CONFIG['model_path']
        ).to(_device)
        
        _model.eval()
        
        if EMBEDDING_CONFIG['use_fp16'] and _device.type == 'cuda':
            _model = _model.half()
        
        logger.info("Embedding 模型加载完成")


def compute_embedding(text: Union[str, List[str]]) -> Union[List[float], None]:
    """
    计算文本的嵌入向量
    
    Args:
        text: 输入文本或文本列表
    
    Returns:
        归一化后的向量 (1024 维)
    """
    _load_model()
    
    try:
        # 编码输入
        encoded_input = _tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=EMBEDDING_CONFIG['max_length'],
            return_tensors='pt'
        ).to(_device)
        
        # 计算向量
        with torch.no_grad():
            model_output = _model(**encoded_input)
            # 取 [CLS] token 作为句子表示
            sentence_embedding = model_output[0][:, 0]
            # L2 归一化
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
        
        # 转换为 Python list
        if isinstance(text, str):
            return sentence_embedding.cpu().numpy()[0].tolist()
        else:
            return sentence_embedding.cpu().numpy().tolist()
    
    except Exception as e:
        logger.error(f"计算 embedding 失败：{e}")
        return None


def get_embedding(text: Union[str, List[str]]) -> List[float]:
    """
    获取文本向量 (兼容接口)
    """
    result = compute_embedding(text)
    if result is None:
        # 返回零向量作为 fallback
        dim = EMBEDDING_CONFIG['vector_dim']
        return [0.0] * dim if isinstance(text, str) else [[0.0] * dim]
    return result


# 测试
if __name__ == "__main__":
    vec = get_embedding("王洪文是谁？")
    print(f"向量维度：{len(vec)}")
    print(f"前 10 个值：{vec[:10]}")
```

### 3. 测试向量化

```python
# test_embedding.py
from src.embedding.embedding_local import get_embedding

# 单条文本
text = "王洪文的生平"
vec = get_embedding(text)
print(f"文本：{text}")
print(f"向量维度：{len(vec)}")

# 批量文本
texts = ["王洪文", "王洪文的生平", "王洪文是谁"]
vecs = get_embedding(texts)
print(f"\n批量向量数量：{len(vecs)}")

# 计算相似度
import numpy as np
v1 = np.array(vecs[0])
v2 = np.array(vecs[1])
similarity = np.dot(v1, v2)  # 已归一化，点积=余弦相似度
print(f"相似度：{similarity:.4f}")
```

## 📝 学习要点

1. **为什么用 [CLS]**: BERT 类模型的约定，[CLS] 位置编码聚合全句信息
2. **L2 归一化的作用**: 点积 = 余弦相似度，计算更高效
3. **单例模式**: 避免重复加载模型，节省内存

## ✅ 检查项

- [ ] 模型加载成功
- [ ] 能计算单条/批量向量
- [ ] 理解归一化的意义

---

# 第 4 课：数据库表设计与初始化

## 🎯 目标

设计数据库表结构，实现连接池和初始化

## ⏱️ 时间：4-5 小时

## 📖 理论学习 (45 分钟)

### 表结构设计

#### 1. FAQ 主表 (faq_knowledge)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | SERIAL | 主键 |
| question | TEXT | 标准问题 |
| similar_question | TEXT | 相似问法 |
| similar_question_vector | vector(1024) | 问题向量 |
| answer | TEXT | 答案 |
| search_vector | tsvector | 全文检索 |
| category | VARCHAR(50) | 类别 |
| source_doc | VARCHAR(200) | 来源文档 |
| confidence | FLOAT | 可信度 |

#### 2. 文档片段表 (doc_chunks)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | SERIAL | 主键 |
| chunk_text | TEXT | 文本片段 |
| chunk_vector | vector(1024) | 片段向量 |
| doc_name | VARCHAR(200) | 文档名 |
| doc_page | INTEGER | 页码 |
| chunk_index | INTEGER | 片段序号 |

#### 3. 对话历史表 (chat_history) - LangChain 标准格式

| 字段 | 类型 | 说明 |
|------|------|------|
| id | SERIAL | 主键 |
| session_id | VARCHAR(100) | 会话 ID |
| role | VARCHAR(20) | 'human' 或 'ai' |
| content | TEXT | 消息内容 |

**说明**：采用 LangChain `PostgresChatMessageHistory` 标准格式，自动持久化对话历史。相比自定义格式（存储 FAQ IDs、LLM 模式等），标准格式更简洁，与 LangChain 生态无缝集成。

### 连接池原理

```
用户请求 ──→ 连接池 ──→ 空闲连接 1
                    ├─→ 空闲连接 2
                    └─→ 空闲连接 3
    
优点:
1. 避免频繁创建/销毁连接
2. 控制并发连接数
3. 提高响应速度
```

## 💻 代码复现

### 1. 数据库配置 (config/pg_config.py)

```python
# -*- coding: utf-8 -*-
"""PostgreSQL 数据库配置"""

import os

# 连接 URL
PG_URL = os.getenv(
    "PG_URL",
    "postgresql://faq_user:faq_password@localhost:5432/faq_db"
)

# 表名配置
PG_TABLE_NAME = "faq_knowledge"
PG_DOC_TABLE = "doc_chunks"
PG_CHAT_TABLE = "chat_history"

# 向量维度
VECTOR_DIM = 1024

# 连接池配置
POOL_MIN_CONN = 1
POOL_MAX_CONN = 10
```

### 2. 连接池实现 (src/vectorstore/pg_pool.py)

```python
# -*- coding: utf-8 -*-
"""数据库连接池 (单例模式)"""

import logging
import psycopg2
from psycopg2 import pool
from config.pg_config import PG_URL, POOL_MIN_CONN, POOL_MAX_CONN

logger = logging.getLogger(__name__)

# 全局连接池
_pool = None


def get_pool():
    """获取连接池 (单例)"""
    global _pool
    
    if _pool is None:
        logger.info("创建数据库连接池")
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=POOL_MIN_CONN,
            maxconn=POOL_MAX_CONN,
            dsn=PG_URL
        )
        logger.info(f"连接池创建成功：min={POOL_MIN_CONN}, max={POOL_MAX_CONN}")
    
    return _pool


def get_connection():
    """获取单个连接 (上下文管理器)"""
    pool = get_pool()
    conn = pool.getconn()
    
    class ConnectionContext:
        def __init__(self, conn, pool):
            self.conn = conn
            self.pool = pool
        
        def __enter__(self):
            return self.conn
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                self.conn.rollback()
            self.pool.putconn(self.conn)
    
    return ConnectionContext(conn, pool)


# 测试
if __name__ == "__main__":
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        print(f"数据库版本：{cursor.fetchone()[0]}")
        cursor.close()
```

### 3. 表结构定义 (src/vectorstore/pg_schema.py)

```python
# -*- coding: utf-8 -*-
"""数据库表结构定义"""

import logging
from .pg_pool import get_connection
from config.pg_config import PG_TABLE_NAME, PG_DOC_TABLE, PG_CHAT_TABLE, VECTOR_DIM

logger = logging.getLogger(__name__)


def create_tables(drop_existing: bool = False):
    """创建所有数据库表和索引"""
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        try:
            # 启用 pgvector 扩展
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # 删除旧表
            if drop_existing:
                cursor.execute(f"DROP TABLE IF EXISTS {PG_TABLE_NAME} CASCADE")
                cursor.execute(f"DROP TABLE IF EXISTS {PG_DOC_TABLE} CASCADE")
                cursor.execute(f"DROP TABLE IF EXISTS {PG_CHAT_TABLE} CASCADE")
            
            # 创建 FAQ 表
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {PG_TABLE_NAME} (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    similar_question TEXT NOT NULL,
                    similar_question_vector vector({VECTOR_DIM}),
                    answer TEXT NOT NULL,
                    search_vector tsvector
                        GENERATED ALWAYS AS (to_tsvector('simple', similar_question)) STORED,
                    category VARCHAR(50),
                    source_doc VARCHAR(200),
                    source_page INTEGER,
                    confidence FLOAT DEFAULT 0.9,
                    created_by VARCHAR(50) DEFAULT 'auto',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 创建 FAQ 表索引
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_hnsw_faq
                ON {PG_TABLE_NAME}
                USING hnsw (similar_question_vector vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_gin_faq
                ON {PG_TABLE_NAME}
                USING gin (search_vector);
            """)
            
            # 创建文档片段表
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {PG_DOC_TABLE} (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    chunk_vector vector({VECTOR_DIM}),
                    doc_name VARCHAR(200) NOT NULL,
                    doc_page INTEGER,
                    chunk_index INTEGER,
                    search_vector tsvector
                        GENERATED ALWAYS AS (to_tsvector('simple', chunk_text)) STORED,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 创建文档表索引
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_hnsw_doc
                ON {PG_DOC_TABLE}
                USING hnsw (chunk_vector vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            
            # 创建对话历史表
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {PG_CHAT_TABLE} (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) NOT NULL,
                    user_query TEXT NOT NULL,
                    retrieved_faq_ids INTEGER[],
                    retrieved_doc_ids INTEGER[],
                    llm_response TEXT,
                    llm_mode VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
            logger.info("所有表和索引创建成功")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"创建表失败：{e}")
            raise
        finally:
            cursor.close()


def init_database():
    """初始化数据库"""
    create_tables(drop_existing=False)
```

### 4. 初始化脚本 (scripts/init_db.py)

```python
#!/usr/bin/env python3
"""数据库初始化脚本"""

import logging
from src.vectorstore.pg_schema import init_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("开始初始化数据库...")
    init_database()
    logger.info("数据库初始化完成!")
```

## 📝 学习要点

1. **生成列**: `search_vector` 使用 GENERATED ALWAYS 自动维护
2. **索引选择**: HNSW 用于向量，GIN 用于全文检索
3. **连接池管理**: 用完后必须归还，避免连接泄漏

## ✅ 检查项

- [ ] 连接池工作正常
- [ ] 所有表创建成功
- [ ] 索引已建立

---

# 第 5 课：FAQ 检索模块实现

## 🎯 目标

实现 FAQ 检索功能，理解相似度计算

## ⏱️ 时间：5-6 小时

## 📖 理论学习 (1 小时)

### FAQ 检索流程

```
用户问题 → Embedding → 向量 → SQL 检索 → 相似度排序 → Top-K FAQ
```

### 相似度计算

```python
# 余弦相似度公式
similarity = cos(θ) = (A · B) / (||A|| × ||B||)

# 归一化后 (||A|| = ||B|| = 1)
similarity = A · B  # 点积
```

### 混合检索策略

| 检索类型 | 说明 | 适用场景 |
|----------|------|----------|
| 向量检索 | 语义相似度 | 模糊匹配 |
| 全文检索 | 关键词匹配 | 精确术语 |
| **混合检索** | **两者结合** | **生产环境** |

## 💻 代码复现

### 1. FAQ 检索器 (src/retrieval/faq_retriever.py)

```python
# -*- coding: utf-8 -*-
"""FAQ 检索器"""

import logging
from typing import List, NamedTuple
from src.embedding.embedding_local import get_embedding
from src.vectorstore.pg_pool import get_connection
from config.pg_config import PG_TABLE_NAME

logger = logging.getLogger(__name__)


class FAQResult(NamedTuple):
    """FAQ 检索结果"""
    id: int
    question: str
    similar_question: str
    answer: str
    similarity: float
    category: str
    source_doc: str


class FAQRetriever:
    """FAQ 检索器"""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[FAQResult]:
        """
        检索 FAQ
        
        Args:
            query: 用户问题
        
        Returns:
            FAQ 结果列表
        """
        # 1. 计算查询向量
        query_vector = get_embedding(query)
        
        # 2. SQL 检索
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # 向量相似度检索 (余弦距离)
            cursor.execute(f"""
                SELECT 
                    id, question, similar_question, answer,
                    1 - (similar_question_vector <=> %s::vector) AS similarity,
                    category, source_doc
                FROM {PG_TABLE_NAME}
                ORDER BY similarity DESC
                LIMIT %s;
            """, (f"[{','.join(map(str, query_vector))}]", self.top_k))
            
            results = []
            for row in cursor.fetchall():
                results.append(FAQResult(
                    id=row[0],
                    question=row[1],
                    similar_question=row[2],
                    answer=row[3],
                    similarity=row[4],
                    category=row[5],
                    source_doc=row[6]
                ))
            
            cursor.close()
            return results


# 测试
if __name__ == "__main__":
    retriever = FAQRetriever(top_k=3)
    results = retriever.retrieve("王洪文是谁？")
    
    for r in results:
        print(f"相似度：{r.similarity:.3f}")
        print(f"问题：{r.question}")
        print(f"答案：{r.answer[:50]}...")
        print()
```

## 📝 学习要点

1. **向量格式转换**: Python list → PostgreSQL vector 字符串
2. **余弦距离**: `<=>` 返回距离 (0=相同，2=相反)，需转换为相似度
3. **NamedTuple**: 结构化返回结果

## ✅ 检查项

- [ ] FAQ 检索功能正常
- [ ] 相似度计算正确
- [ ] 能处理批量查询

---

*(由于篇幅限制，第 6-20 课的大纲如下，详细内容将在后续课程中展开)*

---

# 第 6-20 课 课程大纲

## 第 6 课：文档 RAG 检索模块
- 文档分块策略
- 向量 + 全文混合检索
- 分页信息处理

## 第 7 课：混合检索策略路由
- 置信度阈值判断
- 三模式路由 (FAQ_ONLY/HYBRID/RAG)
- SearchContext 设计

## 第 8 课：LLM 双模架构实现
- BaseLLM 抽象基类
- LocalLLM (Qwen 本地推理)
- APILLM (DeepSeek/OpenAI)
- LLMFactory 工厂模式

## 第 9 课：对话引擎与回答生成
- ResponseGenerator 实现
- ChatEngine 核心流程
- 历史记录整合

## 第 10 课：FastAPI 后端服务开发
- API 路由设计
- 请求/响应模型
- 错误处理
- CORS 配置

## 第 11 课：Web 前端界面实现
- Tailwind CSS 布局
- 对话历史展示
- 流式输出适配

## 第 12 课：LangChain 集成 (上)
- LangChain 基础概念
- Document 加载器
- 向量存储 (Chroma/FAISS)
- RAG Chain 构建

## 第 13 课：LangChain 集成 (下)
- Multi-Query 检索
- BCE-Reranker 重排序
- 向量库持久化

## 第 14 课：流式输出与 SSE
- Server-Sent Events 原理
- 后端流式生成
- 前端 EventSource 接入

## 第 15 课：Callback 可观测性系统
- 日志记录
- 性能监控
- Token 使用统计

## 第 16 课：Prompt 外部化管理
- 提示词模板设计
- 动态加载机制
- 版本管理

## 第 17 课：数据处理流水线
- QA 格式转换
- 文档预处理
- 批量导入

## 第 18 课：数据导入与测试
- 完整测试流程
- 性能基准测试
- 问题排查

## 第 19 课：Docker 容器化部署
- Dockerfile 编写
- docker-compose 配置
- 生产环境优化

## 第 20 课：性能优化与面试准备
- 检索优化策略
- LLM 推理加速
- 面试 Q&A 准备
- 简历项目描述

---

## 📋 附录

### 环境要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4 核 | 8 核 + |
| 内存 | 8GB | 16GB+ |
| GPU | 无 | RTX 3060+ (12GB) |
| 存储 | 50GB | 100GB+ |

### 常见问题

**Q1: 模型下载慢？**
- 使用镜像源：`export HF_ENDPOINT=https://hf-mirror.com`

**Q2: GPU 显存不足？**
- 减小 batch_size
- 使用 FP16
- 减小 max_length

**Q3: 检索结果不准确？**
- 检查 Embedding 模型
- 调整阈值
- 添加更多 FAQ 数据

### 项目扩展方向

1. **多历史人物支持**: 扩展数据结构，支持多人物切换
2. **Agent 功能**: 集成搜索、计算器等工具
3. **多模态**: 支持图片、表格检索
4. **微调优化**: 针对特定领域微调 Embedding 和 LLM

---

> **课程持续更新中...** 完成本课程后，你将拥有一个完整的 RAG 项目经历，适合用于找实习或求职。
