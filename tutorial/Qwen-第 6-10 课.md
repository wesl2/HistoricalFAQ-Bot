# HistoricalFAQ-Bot 实战课程 - 第 6-10 课

> **承接第 1-5 课**: 已完成项目架构、数据库、Embedding、FAQ 检索基础
> 
> **本阶段目标**: 完成核心检索模块和 LLM 双模架构，实现完整的问答流程

---

# 第 6 课：文档 RAG 检索模块

## 🎯 目标

实现文档片段检索功能，理解文档分块和向量检索

## ⏱️ 时间：5-6 小时

## 📖 理论学习 (1 小时)

### 为什么需要文档检索？

| 检索类型 | 优点 | 缺点 | 适用场景 |
|----------|------|------|----------|
| **FAQ 检索** | 答案精准、响应快 | 覆盖有限、需人工维护 | 标准问题 |
| **文档检索** | 覆盖全面、自动处理 | 答案需生成、响应慢 | 探索性问题 |

### 文档分块策略

```
原始文档 (100 页 PDF)
    │
    ▼
文本提取 (纯文本)
    │
    ▼
分块处理 (每块 512 字符，重叠 50 字符)
    │
    ├── Chunk 1: [0-512]
    ├── Chunk 2: [462-974]  ← 重叠 50 字符，保持上下文连贯
    ├── Chunk 3: [924-1436]
    └── ...
    │
    ▼
每块计算向量 → 存入 doc_chunks 表
```

### 分块参数选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `chunk_size` | 512 | 适合 BGE-M3 的 512 长度 |
| `chunk_overlap` | 50 | 保持上下文连贯 |
| `top_k` | 10 | 宽召回，给重排序留空间 |

## 💻 代码复现

### 1. 理解项目中的文档检索实现

在项目中，文档检索由 `src/vectorstore/pg_search.py` 中的 `HybridSearcher` 实现：

```python
# src/vectorstore/pg_search.py (部分)

class HybridSearcher:
    """混合检索器：FAQ + 文档"""
    
    def __init__(self):
        self.faq_searcher = FAQSearcher()
        self.doc_table = PG_DOC_TABLE
    
    def _search_docs(
        self,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[SearchResult]:
        """检索文档片段"""
        vector_str = "[" + ",".join([str(v) for v in query_vector]) + "]"
        
        sql = f"""
            SELECT
                id, chunk_text, doc_name,
                1 - (chunk_vector <=> %s::vector) as similarity,
                doc_name as source_doc, doc_page, NULL as category
            FROM {self.doc_table}
            ORDER BY chunk_vector <=> %s::vector
            LIMIT %s;
        """
        
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (vector_str, vector_str, top_k))
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append(SearchResult(
                    id=row[0],
                    question="",  # 文档片段没有 question
                    answer=row[1],  # chunk_text 作为 answer
                    score=row[3],
                    source_doc=row[4],
                    source_page=row[5],
                    category="document"
                ))
            return results
```

### 2. 创建独立的文档检索器 (src/retrieval/doc_retriever.py)

```python
# -*- coding: utf-8 -*-
"""文档 RAG 检索器"""

import logging
from typing import List, NamedTuple, Optional
from dataclasses import dataclass
from src.embedding.embedding_local import get_embedding
from src.vectorstore.pg_pool import get_connection
from config.pg_config import PG_DOC_TABLE

logger = logging.getLogger(__name__)


@dataclass
class DocResult:
    """文档检索结果"""
    id: int
    content: str          # 文档片段内容
    doc_name: str         # 来源文档名
    doc_page: int         # 页码
    chunk_index: int      # 片段序号
    similarity: float     # 相似度分数
    category: str = "document"


class DocRetriever:
    """文档检索器"""
    
    def __init__(self, top_k: int = 10):
        """
        初始化文档检索器
        
        Args:
            top_k: 返回结果数量（建议设大一些，给重排序留空间）
        """
        self.top_k = top_k
        self.doc_table = PG_DOC_TABLE
    
    def retrieve(self, query: str) -> List[DocResult]:
        """
        检索文档片段
        
        Args:
            query: 用户问题
        
        Returns:
            文档检索结果列表
        """
        # 1. 计算查询向量
        query_vector = get_embedding(query)
        
        # 2. SQL 检索
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # 向量转字符串
            vector_str = "[" + ",".join([str(v) for v in query_vector]) + "]"
            
            # 向量相似度检索
            cursor.execute(f"""
                SELECT 
                    id, chunk_text, doc_name, doc_page, chunk_index,
                    1 - (chunk_vector <=> %s::vector) AS similarity
                FROM {self.doc_table}
                ORDER BY similarity DESC
                LIMIT %s;
            """, (vector_str, self.top_k))
            
            results = []
            for row in cursor.fetchall():
                results.append(DocResult(
                    id=row[0],
                    content=row[1],
                    doc_name=row[2],
                    doc_page=row[3],
                    chunk_index=row[4],
                    similarity=row[5]
                ))
            
            cursor.close()
            
            logger.info(f"文档检索完成：找到 {len(results)} 个片段")
            return results
    
    def retrieve_with_fulltext(
        self, 
        query: str, 
        vector_weight: float = 0.7
    ) -> List[DocResult]:
        """
        混合检索：向量 + 全文
        
        Args:
            query: 用户问题
            vector_weight: 向量检索权重
        
        Returns:
            混合检索结果
        """
        query_vector = get_embedding(query)
        vector_str = "[" + ",".join([str(v) for v in query_vector]) + "]"
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # 混合检索 SQL
            cursor.execute(f"""
                WITH vector_results AS (
                    SELECT 
                        id, chunk_text, doc_name, doc_page, chunk_index,
                        1 - (chunk_vector <=> %s::vector) AS vector_score,
                        0.0 AS text_score
                    FROM {self.doc_table}
                    ORDER BY vector_score DESC
                    LIMIT 20
                ),
                text_results AS (
                    SELECT 
                        id, chunk_text, doc_name, doc_page, chunk_index,
                        0.0 AS vector_score,
                        ts_rank(search_vector, plainto_tsquery('simple', %s)) AS text_score
                    FROM {self.doc_table}
                    WHERE search_vector @@ plainto_tsquery('simple', %s)
                    ORDER BY text_score DESC
                    LIMIT 20
                )
                SELECT 
                    COALESCE(v.id, t.id) as id,
                    COALESCE(v.chunk_text, t.chunk_text) as chunk_text,
                    COALESCE(v.doc_name, t.doc_name) as doc_name,
                    COALESCE(v.doc_page, t.doc_page) as doc_page,
                    COALESCE(v.chunk_index, t.chunk_index) as chunk_index,
                    (%s * COALESCE(v.vector_score, 0) + %s * COALESCE(t.text_score, 0)) 
                        AS combined_score
                FROM vector_results v
                FULL OUTER JOIN text_results t ON v.id = t.id
                ORDER BY combined_score DESC
                LIMIT %s;
            """, (
                vector_str, query, query, 
                vector_weight, 1 - vector_weight, 
                self.top_k
            ))
            
            results = []
            for row in cursor.fetchall():
                results.append(DocResult(
                    id=row[0],
                    content=row[1],
                    doc_name=row[2],
                    doc_page=row[3],
                    chunk_index=row[4],
                    similarity=row[5]
                ))
            
            cursor.close()
            return results


# 测试
if __name__ == "__main__":
    retriever = DocRetriever(top_k=5)
    results = retriever.retrieve("王洪文的生平")
    
    for r in results:
        print(f"相似度：{r.similarity:.3f}")
        print(f"来源：{r.doc_name} 第{r.doc_page}页")
        print(f"内容：{r.content[:100]}...")
        print()
```

### 3. 数据导入脚本 (scripts/ingest_data.py)

```python
#!/usr/bin/env python3
"""文档数据导入脚本"""

import logging
import os
from pathlib import Path
from typing import List
from src.vectorstore.pg_pool import get_connection
from src.embedding.embedding_local import get_embedding
from config.pg_config import PG_DOC_TABLE, BATCH_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    文本分块
    
    Args:
        text: 输入文本
        chunk_size: 每块大小
        overlap: 重叠大小
    
    Returns:
        文本块列表
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # 如果不是最后一块，尝试在句子边界处切分
        if end < len(text):
            # 查找最近的句号
            last_period = chunk.rfind('。')
            if last_period > chunk_size * 0.5:  # 至少在一半位置
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def process_document(file_path: str) -> List[dict]:
    """
    处理单个文档
    
    Args:
        file_path: 文件路径
    
    Returns:
        文档块列表
    """
    logger.info(f"处理文档：{file_path}")
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 分块
    chunks = chunk_text(text)
    logger.info(f"分块完成：{len(chunks)} 块")
    
    # 计算向量并构建记录
    records = []
    doc_name = Path(file_path).stem
    
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        
        vector = get_embedding(chunk)
        vector_str = "[" + ",".join([str(v) for v in vector]) + "]"
        
        records.append({
            "chunk_text": chunk,
            "chunk_vector": vector_str,
            "doc_name": doc_name,
            "doc_page": 0,  # 纯文本没有页码
            "chunk_index": i
        })
    
    return records


def ingest_documents(data_dir: str):
    """
    批量导入文档
    
    Args:
        data_dir: 文档目录
    """
    data_path = Path(data_dir)
    all_records = []
    
    # 收集所有文档
    for file_path in data_path.glob("*.txt"):
        records = process_document(str(file_path))
        all_records.extend(records)
    
    logger.info(f"总文档块数：{len(all_records)}")
    
    # 批量插入
    if not all_records:
        logger.warning("没有文档需要导入")
        return
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 清空旧数据
        cursor.execute(f"TRUNCATE TABLE {PG_DOC_TABLE} CASCADE")
        conn.commit()
        
        # 批量插入
        insert_sql = f"""
            INSERT INTO {PG_DOC_TABLE} 
            (chunk_text, chunk_vector, doc_name, doc_page, chunk_index)
            VALUES %s
        """
        
        from psycopg2.extras import execute_values
        
        batch = []
        for record in all_records:
            batch.append((
                record["chunk_text"],
                record["chunk_vector"],
                record["doc_name"],
                record["doc_page"],
                record["chunk_index"]
            ))
            
            if len(batch) >= BATCH_SIZE:
                execute_values(cursor, insert_sql, batch)
                conn.commit()
                logger.info(f"已导入 {len(batch)} 条记录...")
                batch = []
        
        if batch:
            execute_values(cursor, insert_sql, batch)
            conn.commit()
        
        cursor.close()
    
    logger.info(f"导入完成：共 {len(all_records)} 个文档块")


if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/processed"
    ingest_documents(data_dir)
```

## 📝 学习要点

1. **文档分块的重要性**: 分块太大包含噪声，太小丢失上下文
2. **重叠设计**: 保持语义连贯，避免信息丢失
3. **混合检索**: 向量检索 (语义) + 全文检索 (关键词) 互补

## ✅ 检查项

- [ ] 文档检索功能正常
- [ ] 能正确分块并导入数据
- [ ] 理解混合检索原理

---

# 第 7 课：混合检索策略路由

## 🎯 目标

实现智能检索路由，根据置信度自动选择检索策略

## ⏱️ 时间：4-5 小时

## 📖 理论学习 (45 分钟)

### 为什么需要检索路由？

```
用户问题："王洪文是谁？"  ──→ FAQ 检索 (相似度 0.95) ──→ 直接返回答案 ✅

用户问题："王洪文在文革中的具体活动有哪些？"
    └──→ FAQ 检索 (相似度 0.75) ──→ 置信度低 ──→ 转文档检索 🔀

用户问题："介绍一下王洪文和江青的关系"
    └──→ FAQ 检索 (相似度 0.60) + 文档检索 ──→ 融合结果 🔀
```

### 三模式路由策略

| 模式 | 触发条件 | 处理方式 |
|------|----------|----------|
| **FAQ_ONLY** | FAQ 相似度 > 0.90 | 直接返回 FAQ 答案 |
| **HYBRID** | FAQ 相似度 0.85-0.90 | FAQ + 文档融合 |
| **RAG** | FAQ 相似度 < 0.85 | 纯文档检索 + LLM 生成 |

### 路由决策流程

```
用户查询
    │
    ▼
FAQ 检索 (计算相似度)
    │
    ├── 相似度 > 0.90 ──→ FAQ_ONLY 模式
    │                       │
    │                       ▼
    │                   直接返回答案
    │
    ├── 相似度 0.85-0.90 ──→ HYBRID 模式
    │                       │
    │                       ▼
    │                   FAQ + 文档检索
    │                       │
    │                       ▼
    │                   LLM 生成答案
    │
    └── 相似度 < 0.85 ──→ RAG 模式
                            │
                            ▼
                        文档检索
                            │
                            ▼
                        LLM 生成答案
```

## 💻 代码复现

### 1. 定义检索上下文和枚举

```python
# src/retrieval/search_router.py (完整版)

# -*- coding: utf-8 -*-
"""检索策略路由器"""

import logging
from enum import Enum
from typing import List, NamedTuple, Optional
from dataclasses import dataclass, field
from .faq_retriever import FAQResult, FAQRetriever
from .doc_retriever import DocResult, DocRetriever

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """检索类型枚举"""
    FAQ_ONLY = "faq_only"      # 仅 FAQ
    HYBRID = "hybrid"          # 混合检索
    RAG = "rag"                # 纯文档检索


@dataclass
class SearchContext:
    """检索上下文"""
    faq_results: List[FAQResult] = field(default_factory=list)
    doc_results: List[DocResult] = field(default_factory=list)
    search_type: SearchType = SearchType.RAG
    confidence: float = 0.0


class SearchRouter:
    """检索路由器"""
    
    def __init__(
        self,
        high_threshold: float = 0.90,
        low_threshold: float = 0.85
    ):
        """
        初始化路由器
        
        Args:
            high_threshold: 高置信度阈值 (> 直接返回 FAQ)
            low_threshold: 低置信度阈值 (< 转 RAG)
        """
        self.faq_retriever = FAQRetriever(top_k=3)
        self.doc_retriever = DocRetriever(top_k=10)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
    
    def search(self, query: str) -> SearchContext:
        """
        检索路由主入口
        
        Args:
            query: 用户查询
        
        Returns:
            SearchContext: 包含检索结果和路由决策
        """
        # 1. FAQ 检索
        faq_results = self.faq_retriever.retrieve(query)
        
        # 2. 根据 FAQ 置信度决策
        if not faq_results:
            # 无 FAQ 匹配，直接走 RAG
            doc_results = self.doc_retriever.retrieve(query)
            return SearchContext(
                faq_results=[],
                doc_results=doc_results,
                search_type=SearchType.RAG,
                confidence=0.0
            )
        
        # 获取最高相似度
        max_similarity = faq_results[0].similarity
        
        # 3. 高置信度：直接返回 FAQ 答案
        if max_similarity >= self.high_threshold:
            logger.info(f"高置信度 ({max_similarity:.3f})，使用 FAQ_ONLY 模式")
            return SearchContext(
                faq_results=faq_results,
                doc_results=[],
                search_type=SearchType.FAQ_ONLY,
                confidence=max_similarity
            )
        
        # 4. 低置信度或无匹配：检索文档
        doc_results = self.doc_retriever.retrieve(query)
        
        if not faq_results:
            # 纯 RAG
            return SearchContext(
                faq_results=[],
                doc_results=doc_results,
                search_type=SearchType.RAG,
                confidence=0.0
            )
        
        # 5. 融合模式
        logger.info(f"中置信度 ({max_similarity:.3f})，使用 HYBRID 模式")
        return SearchContext(
            faq_results=faq_results,
            doc_results=doc_results,
            search_type=SearchType.HYBRID,
            confidence=max_similarity
        )


# 测试
if __name__ == "__main__":
    router = SearchRouter()
    
    # 测试问题 1：标准问题 (应触发 FAQ_ONLY)
    ctx1 = router.search("王洪文是谁？")
    print(f"问题 1: {ctx1.search_type.value}, 置信度：{ctx1.confidence:.3f}")
    
    # 测试问题 2：复杂问题 (应触发 HYBRID 或 RAG)
    ctx2 = router.search("王洪文在文革期间的具体活动")
    print(f"问题 2: {ctx2.search_type.value}, 置信度：{ctx2.confidence:.3f}")
```

### 2. 检索配置 (config/retrieval_config.py)

```python
# -*- coding: utf-8 -*-
"""检索策略配置"""

import os

RETRIEVAL_CONFIG = {
    # 检索模式
    "default_mode": os.getenv("RETRIEVAL_MODE", "hybrid"),
    
    # FAQ 检索配置
    "faq": {
        "top_k": int(os.getenv("FAQ_TOP_K", "5")),
        # 高置信度阈值 (直接回答)
        "high_confidence_threshold": 0.90,
        # 低置信度阈值 (转 RAG)
        "low_confidence_threshold": 0.85,
    },
    
    # 文档检索配置
    "doc": {
        "top_k": int(os.getenv("DOC_TOP_K", "10")),
        "similarity_threshold": 0.70,  # 宽召回
    },
    
    # 混合检索融合配置
    "fusion": {
        "method": "rrf",  # Reciprocal Rank Fusion
        "rrf_k": 60,
        "faq_weight": 0.6,
        "doc_weight": 0.4
    }
}
```

### 3. 测试检索路由

```python
# test_search_router.py

from src.retrieval.search_router import SearchRouter, SearchType

router = SearchRouter()

test_cases = [
    "王洪文是谁？",
    "王洪文的生平",
    "王洪文在文革中的角色",
    "王洪文和江青的关系",
    "介绍一下四人帮"
]

for query in test_cases:
    print(f"\n{'='*50}")
    print(f"问题：{query}")
    print(f"{'='*50}")
    
    ctx = router.search(query)
    
    print(f"路由模式：{ctx.search_type.value}")
    print(f"置信度：{ctx.confidence:.3f}")
    print(f"FAQ 结果：{len(ctx.faq_results)} 条")
    print(f"文档结果：{len(ctx.doc_results)} 条")
    
    if ctx.faq_results:
        print(f"\n最佳 FAQ 匹配:")
        faq = ctx.faq_results[0]
        print(f"  问题：{faq.question}")
        print(f"  答案：{faq.answer[:100]}...")
```

## 📝 学习要点

1. **阈值调优**: 根据实际数据调整阈值，平衡准确率和召回率
2. **降级策略**: FAQ 无匹配时自动降级到 RAG
3. **数据类设计**: 使用 `@dataclass` 简化代码

## ✅ 检查项

- [ ] 理解三模式路由逻辑
- [ ] 能根据日志判断路由决策
- [ ] 阈值可配置

---

# 第 8 课：标准 LLM 架构实现（LangChain BaseChatModel）

## 🎯 目标

理解 LangChain 标准 LLM 接口，放弃自定义桥接，直接使用 `BaseChatModel`。

## ⏱️ 时间：3-4 小时

## 📖 理论学习 (1 小时)

### 为什么使用 LangChain 标准接口？

**重构背景**：早期版本我们手写 `BaseLLM` 抽象类来封装本地模型和 API。这虽然能跑，但在接入 LCEL（LangChain 表达式语言）时，必须使用 `RunnableLambda` 做桥接，**失去了流式输出和标准调用链的优势**。

**当前架构**直接使用 LangChain 提供的适配器：
- **API 模型**：使用 `langchain_openai.ChatOpenAI`（兼容所有 OpenAI 格式接口，如 DeepSeek、Kimi、通义千问）。
- **本地模型**：使用 `langchain_huggingface.HuggingFacePipeline` + `ChatHuggingFace`。

### 架构对比

**❌ 旧版（已废弃）：自定义桥接（不支持原生 LCEL）**
```text
┌─────────────────────────────────────┐
│          BaseLLM (自定义抽象类)      │
│  - chat(messages) -> str            │
└─────────────────────────────────────┘
                 ▲
                 │ 需要 RunnableLambda 桥接
    ┌────────────┴────────────┐
    │                         │
    ▼                         ▼
┌─────────────┐         ┌─────────────┐
│  LocalLLM   │         │   APILLM    │
│ (Transformers)│       │(OpenAI SDK) │
└─────────────┘         └─────────────┘
```

**✅ 新版（当前代码）：LangChain 标准（原生支持 LCEL / Stream）**
```text
┌─────────────────────────────────────┐
│      BaseChatModel (LangChain 标准)  │
│  - invoke(messages) -> AIMessage    │
│  - stream(messages) -> Iterator     │
└─────────────────────────────────────┘
                 ▲
                 │ 原生支持 LCEL 管道
    ┌────────────┴────────────┐
    │                         │
    ▼                         ▼
┌─────────────────────┐ ┌──────────────────────┐
│ HuggingFacePipeline │ │ ChatOpenAI (API)     │
│ + ChatHuggingFace   │ │ 兼容 DeepSeek/Kimi/  │
└─────────────────────┘ │ 通义千问等           │
                        └──────────────────────┘
```

> **重要变更**：`src/llm/base_llm.py`、`local_llm.py`、`api_llm.py`、`llm_factory.py` 已被废弃并删除，统一使用 `standard_llm.py`。

## 💻 代码复现

### 1. 标准 LLM 工厂 (`src/llm/standard_llm.py`)

不再手写模型加载逻辑，直接包装 LangChain 官方组件：

```python
# -*- coding: utf-8 -*-
"""
标准 LangChain LLM 封装（公司级实践）
返回真正的 BaseChatModel，可在 LCEL 中直接使用
"""

from typing import Optional, Iterator
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import pipeline, TextIteratorStreamer
import torch
from threading import Thread

from config.model_config import LLM_CONFIG

logger = None  # 延迟导入，避免循环依赖


def get_logger():
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    return logger


class StandardLLM:
    """
    标准 LangChain LLM 工厂
    返回真正的 BaseChatModel，可在 LCEL 中直接使用
    """

    # 全局缓存（避免重复加载模型）
    _cache = {}

    @classmethod
    def create(cls, mode: str = None) -> BaseChatModel:
        """
        创建标准 LangChain ChatModel

        Args:
            mode: "local" 或 "api"，None 则使用默认配置

        Returns:
            BaseChatModel: 标准 LangChain 聊天模型
        """
        mode = mode or LLM_CONFIG["default_mode"]

        if mode not in cls._cache:
            if mode == "api":
                cls._cache[mode] = cls._create_api_llm()
            else:
                cls._cache[mode] = cls._create_local_llm()
            get_logger().info(f"创建标准 LLM 实例: {mode}")

        return cls._cache[mode]

    @classmethod
    def _create_api_llm(cls) -> BaseChatModel:
        """创建 API 模型（标准 ChatOpenAI 接口）"""
        api_config = LLM_CONFIG["api"]

        return ChatOpenAI(
            model=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            temperature=api_config["temperature"],
            max_tokens=api_config["max_tokens"],
            streaming=True,  # 启用流式
            timeout=api_config.get("timeout", 60),
        )

    @classmethod
    def _create_local_llm(cls) -> BaseChatModel:
        """创建本地模型（标准 HuggingFacePipeline + ChatHuggingFace 接口）"""
        local_config = LLM_CONFIG["local"]

        # 使用 HuggingFacePipeline 包装 transformers 模型
        pipe = pipeline(
            "text-generation",
            model=local_config["model_path"],
            torch_dtype=torch.float16 if local_config.get("torch_dtype") == "float16" else torch.float32,
            device_map=local_config.get("device_map", "auto"),
            max_new_tokens=local_config.get("max_new_tokens", 512),
            temperature=local_config.get("temperature", 0.7),
            do_sample=local_config.get("do_sample", True),
            top_p=local_config.get("top_p", 0.9),
            top_k=local_config.get("top_k", 50),
            trust_remote_code=True,
        )

        # 包装成 LangChain LLM
        hf_llm = HuggingFacePipeline(pipeline=pipe)

        # 转换为 ChatModel（支持标准 messages 接口）
        return ChatHuggingFace(llm=hf_llm)

    @classmethod
    def clear_cache(cls):
        """清理缓存（用于切换模型或释放显存）"""
        cls._cache.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        get_logger().info("LLM 缓存已清理")


# 便捷函数
def get_standard_llm(mode: str = None) -> BaseChatModel:
    """获取标准 LLM 实例（带缓存）"""
    return StandardLLM.create(mode)
```

### 2. 为什么不再手写 `LocalLLM/APILLM/LLMFactory`？

**已废弃的文件**（已从代码库删除）：
- `src/llm/base_llm.py` - 自定义抽象基类
- `src/llm/local_llm.py` - 本地模型实现
- `src/llm/api_llm.py` - API 模型实现  
- `src/llm/llm_factory.py` - 工厂模式

使用 `StandardLLM` 替代后：
1. **原生支持流式 (`llm.stream()`)**：不再需要自己写 `TextIteratorStreamer` 线程阻塞逻辑。
2. **兼容 LCEL**：可以直接放入 `prompt | llm | parser` 管道，不需要 `RunnableLambda` 包装。
3. **统一返回格式**：返回标准的 `AIMessage` 对象，而不是裸字符串。
4. **支持多厂商 API**：通过 `ChatOpenAI` 兼容 DeepSeek、Kimi、通义千问等所有 OpenAI 格式接口。

### 3. 模型配置 (config/model_config.py)

```python
# -*- coding: utf-8 -*-
"""模型配置模块"""

import os

# LLM 配置（双模架构）
LLM_CONFIG = {
    # 默认模式: "local" 或 "api"
    "default_mode": os.getenv("LLM_MODE", "local"),
    
    # 本地模型配置（Qwen/Llama 等）
    "local": {
        "model_path": os.getenv(
            "LOCAL_LLM_PATH",
            "/root/autodl-tmp/models/qwen/Qwen1.5-7B-Chat"
        ),
        "device": os.getenv("LOCAL_LLM_DEVICE", "cuda"),
        "device_map": os.getenv("LOCAL_LLM_DEVICE_MAP", "auto"),
        "torch_dtype": "float16",
        "max_new_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "512")),
        "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.7")),
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "system_prompt": """你是一位专业的中国现代史研究专家..."""
    },
    
    # API 模型配置（OpenAI/DeepSeek/Claude/通义千问/Kimi 等）
    "api": {
        "provider": os.getenv("API_PROVIDER", "deepseek"),
        "api_key": os.getenv("API_KEY", ""),
        "base_url": os.getenv("API_BASE_URL", "https://api.deepseek.com/v1"),
        "model": os.getenv("API_MODEL", "deepseek-chat"),
        "max_tokens": int(os.getenv("API_MAX_TOKENS", "512")),
        "temperature": float(os.getenv("API_TEMPERATURE", "0.7")),
        "timeout": 60,
        "retry": 3,
        "system_prompt": """你是一位专业的中国现代史研究专家..."""
    }
}

# 厂商特定的 API 配置
API_PROVIDER_CONFIG = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-3.5-turbo"
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
    },
    "claude": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-3-sonnet-20240229"
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4"
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k"
    }
}
```

### 4. 测试标准 LLM

```python
# test_standard_llm.py

from src.llm.standard_llm import get_standard_llm, StandardLLM
from langchain_core.messages import HumanMessage, SystemMessage

# 测试本地模型
print("=" * 50)
print("测试本地 LLM (Standard)")
print("=" * 50)

local_llm = get_standard_llm("local")
print(f"类型：{type(local_llm).__name__}")

messages = [
    SystemMessage(content="你是一位历史专家"),
    HumanMessage(content="王洪文是谁？请用一句话回答")
]

response = local_llm.invoke(messages)
print(f"回答：{response.content}")

# 测试 API 模型
print("\n" + "=" * 50)
print("测试 API LLM (Standard)")
print("=" * 50)

api_llm = get_standard_llm("api")
print(f"类型：{type(api_llm).__name__}")

response = api_llm.invoke(messages)
print(f"回答：{response.content}")

# 测试流式输出
print("\n" + "=" * 50)
print("测试流式输出")
print("=" * 50)

for chunk in local_llm.stream(messages):
    print(chunk.content, end="", flush=True)
```

### 5. 通义千问 API 配置示例

```bash
# 配置通义千问 (DashScope)
export API_PROVIDER=qwen
export API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export API_MODEL=qwen-turbo  # 或 qwen-plus, qwen-max
export API_KEY=your-dashscope-api-key
```

```python
# 使用通义千问
import os
os.environ["API_PROVIDER"] = "qwen"
os.environ["API_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["API_MODEL"] = "qwen-turbo"
os.environ["API_KEY"] = "your-key"

llm = get_standard_llm("api")
response = llm.invoke([HumanMessage(content="你好")])
print(response.content)
```

## ✅ 检查项

- [ ] 理解 `BaseChatModel` 相比自定义 `BaseLLM` 的优势。
- [ ] 能够通过 `get_standard_llm()` 获取标准 LLM 实例。
- [ ] 理解为什么 API 模式使用 `ChatOpenAI` 却能兼容 DeepSeek/Kimi/通义千问。
- [ ] 了解哪些旧文件已被废弃（`base_llm.py`、`local_llm.py`、`api_llm.py`、`llm_factory.py`）。

1. **抽象基类**: 定义统一接口，便于替换实现
2. **单例缓存**: 避免重复加载模型，节省内存
3. **指数退避**: API 失败重试策略

## ✅ 检查项

- [ ] 理解工厂模式
- [ ] 本地模型能正常推理
- [ ] API 模型配置正确

---

# 第 9 课：对话引擎与回答生成

## 🎯 目标

整合检索和生成，实现完整的对话流程

## ⏱️ 时间：5-6 小时

## 📖 理论学习 (1 小时)

### 对话引擎核心流程

```
1. 接收用户查询
         │
         ▼
2. 检索路由 (SearchRouter)
         │
         ├── FAQ_ONLY ──→ 直接返回 FAQ 答案
         │
         ├── HYBRID ──→ FAQ + 文档 ──→ 构建提示词 ──→ LLM
         │
         └── RAG ──→ 文档 ──→ 构建提示词 ──→ LLM
         │
         ▼
3. 生成回答 (ResponseGenerator)
         │
         ▼
4. 返回格式化结果
```

### 提示词构建策略

```python
# HYBRID 模式提示词结构

请参考以下资料回答问题：

【相关问答】
1. 问题：王洪文的生平
   答案：王洪文（1935 年 -1992 年），吉林长春人...

【相关文档片段】
1. 来源：王洪文传 第 15 页
   内容：1966 年，王洪文参与组织上海工人...

用户问题：王洪文在文革期间做了什么？

请基于以上资料回答。如果资料不足以回答，请明确说明。
```

### 数据流设计

| 组件 | 输入 | 输出 |
|------|------|------|
| SearchRouter | query | SearchContext |
| ResponseGenerator | query + SearchContext | answer |
| ChatEngine | query + history | {answer, sources, search_type, confidence} |

## 💻 代码复现

### 1. 回答生成器 (src/chat/response_generator.py)

```python
# -*- coding: utf-8 -*-
"""回答生成器"""

import logging
from typing import List
from src.llm.base_llm import BaseLLM
from src.retrieval.faq_retriever import FAQResult
from src.retrieval.doc_retriever import DocResult

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """回答生成器"""
    
    def __init__(self, llm: BaseLLM):
        """
        初始化
        
        Args:
            llm: LLM 实例
        """
        self.llm = llm
    
    def generate(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult]
    ) -> str:
        """
        生成回答
        
        Args:
            query: 用户查询
            faq_results: FAQ 检索结果
            doc_results: 文档检索结果
        
        Returns:
            str: 生成的回答
        """
        # 构建提示词
        prompt = self._build_prompt(query, faq_results, doc_results)
        
        # 调用 LLM
        messages = [
            {"role": "system", "content": "你是一个专业的中国历史研究助手。基于提供的参考资料回答用户问题。"},
            {"role": "user", "content": prompt}
        ]
        
        return self.llm.chat(messages)
    
    def _build_prompt(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult]
    ) -> str:
        """构建提示词"""
        prompt_parts = ["请参考以下资料回答问题：\n"]
        
        # 添加 FAQ 参考
        if faq_results:
            prompt_parts.append("\n【相关问答】")
            for i, r in enumerate(faq_results[:3], 1):
                prompt_parts.append(f"{i}. 问题：{r.question}")
                prompt_parts.append(f"   答案：{r.answer[:200]}...")
        
        # 添加文档参考
        if doc_results:
            prompt_parts.append("\n【相关文档片段】")
            for i, r in enumerate(doc_results[:3], 1):
                prompt_parts.append(f"{i}. 来源：{r.doc_name} 第{r.doc_page}页")
                prompt_parts.append(f"   内容：{r.content[:200]}...")
        
        prompt_parts.append(f"\n用户问题：{query}")
        prompt_parts.append("\n请基于以上资料回答。如果资料不足以回答，请明确说明。")
        
        return "\n".join(prompt_parts)
```

### 2. 对话引擎 (src/chat/chat_engine.py)

```python
# -*- coding: utf-8 -*-
"""对话引擎"""

import logging
from typing import List, Dict, Any

from src.retrieval.search_router import SearchRouter, SearchType
from src.llm.llm_factory import LLMFactory
from src.chat.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)


class ChatEngine:
    """
    对话引擎
    
    核心流程:
    1. 接收用户查询
    2. 检索相关内容 (FAQ/文档)
    3. 调用 LLM 生成回答
    4. 返回格式化结果
    """
    
    def __init__(self, llm_mode: str = None):
        """
        初始化对话引擎
        
        Args:
            llm_mode: LLM 模式，None 则使用配置默认值
        """
        self.search_router = SearchRouter()
        self.llm = LLMFactory.create_llm(llm_mode)
        self.response_gen = ResponseGenerator(self.llm)
        
        logger.info(f"对话引擎初始化完成：LLM={llm_mode or 'default'}")
    
    def chat(self, query: str, history: List[Dict] = None) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            query: 用户问题
            history: 对话历史（可选）
        
        Returns:
            {
                "answer": str,
                "sources": list,
                "search_type": str,
                "confidence": float
            }
        """
        # 1. 检索相关内容
        search_context = self.search_router.search(query)
        
        logger.info(
            f"检索完成：模式={search_context.search_type.value}, "
            f"置信度={search_context.confidence:.3f}, "
            f"FAQ={len(search_context.faq_results)}, "
            f"文档={len(search_context.doc_results)}"
        )
        
        # 2. 根据检索类型生成回答
        if search_context.search_type == SearchType.FAQ_ONLY:
            # 高置信度 FAQ，直接返回答案
            answer = search_context.faq_results[0].answer
            sources = [{
                "type": "faq",
                "question": search_context.faq_results[0].question,
                "confidence": search_context.faq_results[0].similarity
            }]
        
        else:
            # 需要 LLM 生成
            answer = self.response_gen.generate(
                query=query,
                faq_results=search_context.faq_results,
                doc_results=search_context.doc_results
            )
            
            # 构建来源信息
            sources = []
            for r in search_context.faq_results[:3]:
                sources.append({
                    "type": "faq",
                    "question": r.question,
                    "confidence": r.similarity
                })
            for r in search_context.doc_results[:3]:
                sources.append({
                    "type": "doc",
                    "source": r.doc_name,
                    "page": r.doc_page
                })
        
        # 3. 返回格式化结果
        return {
            "answer": answer,
            "sources": sources,
            "search_type": search_context.search_type.value,
            "confidence": search_context.confidence
        }


# 测试
if __name__ == "__main__":
    engine = ChatEngine(llm_mode="local")
    
    # 测试查询
    result = engine.chat("王洪文是谁？")
    
    print(f"回答：{result['answer']}")
    print(f"检索模式：{result['search_type']}")
    print(f"置信度：{result['confidence']:.3f}")
    print(f"来源数量：{len(result['sources'])}")
```

### 3. CLI 演示脚本 (src/chat/demo.py)

```python
#!/usr/bin/env python3
"""对话引擎 CLI 演示"""

import logging
from src.chat.chat_engine import ChatEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    print("=" * 60)
    print("Historical FAQ Bot - 对话演示")
    print("=" * 60)
    print("输入 'quit' 退出\n")
    
    # 初始化对话引擎
    engine = ChatEngine(llm_mode="local")
    
    # 对话循环
    while True:
        try:
            query = input("问：").strip()
            
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break
            
            # 处理查询
            result = engine.chat(query)
            
            # 显示结果
            print(f"\n答：{result['answer']}")
            print(f"\n[检索模式：{result['search_type']}, 置信度：{result['confidence']:.3f}]")
            print(f"[来源：{len(result['sources'])} 条]\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            print(f"错误：{e}\n")


if __name__ == "__main__":
    main()
```

## 📝 学习要点

1. **分层设计**: SearchRouter 负责检索，ResponseGenerator 负责生成
2. **统一返回格式**: 无论哪种模式，返回结构一致
3. **来源追踪**: 记录答案来源，便于追溯

## ✅ 检查项

- [ ] 对话引擎能正常工作
- [ ] 理解三种模式的返回差异
- [ ] CLI 演示能运行

---

# 第 10 课：FastAPI 后端服务开发

## 🎯 目标

实现 RESTful API 服务，支持 Web 前端调用

## ⏱️ 时间：4-5 小时

## 📖 理论学习 (45 分钟)

### 为什么用 FastAPI？

| 特性 | FastAPI | Flask | Django REST |
|------|---------|-------|-------------|
| **异步支持** | ✅ 原生 | ❌ 需扩展 | ⚠️ 部分 |
| **自动文档** | ✅ Swagger | ❌ 需扩展 | ⚠️ 需配置 |
| **数据验证** | ✅ Pydantic | ❌ 需扩展 | ✅ 内置 |
| **性能** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### API 设计原则

1. **RESTful 风格**: 使用 HTTP 方法表达操作
2. **统一响应格式**: 成功/失败结构一致
3. **错误处理**: 明确的错误码和消息
4. **版本管理**: URL 中包含版本号

### 核心接口设计

| 接口 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 健康检查 | GET | `/api/health` | 服务状态 |
| 查询接口 | POST | `/api/query` | 标准查询 |
| 流式查询 | POST | `/api/query/stream` | SSE 流式输出 |
| 服务信息 | GET | `/api/info` | 版本信息 |

## 💻 代码复现

### 1. FastAPI 主应用 (src/api/main.py)

```python
# -*- coding: utf-8 -*-
"""FastAPI 后端服务"""

import logging
import os
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.chat.chat_engine import ChatEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="Historical FAQ Bot API",
    description="基于 RAG 架构的历史人物问答系统",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应设置具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 初始化核心组件 ====================

# 从环境变量读取配置
llm_mode = os.getenv("LLM_MODE", "local")

logger.info(f"初始化服务：llm_mode={llm_mode}")

# 初始化对话引擎
chat_engine = ChatEngine(llm_mode=llm_mode)

# ==================== 请求/响应模型 ====================

class QueryRequest(BaseModel):
    """标准查询请求"""
    question: str
    history: List[Dict[str, str]] = None


class QueryResponse(BaseModel):
    """标准查询响应"""
    answer: str
    sources: List[Dict[str, Any]]
    search_type: str
    confidence: float


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    service: str
    version: str


# ==================== API 端点 ====================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        service="Historical FAQ Bot API",
        version="1.0.0"
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_faq(request: QueryRequest):
    """
    处理用户查询
    
    Args:
        request: 包含问题和对话历史的请求
    
    Returns:
        包含回答、来源、检索类型和置信度的响应
    """
    try:
        logger.info(f"收到查询：{request.question[:50]}...")
        
        result = chat_engine.chat(
            query=request.question,
            history=request.history
        )
        
        logger.info(
            f"查询完成，置信度：{result['confidence']:.2f}, "
            f"检索类型：{result.get('search_type', 'unknown')}"
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"查询错误：{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/info")
async def get_info():
    """获取服务信息"""
    return {
        "name": "Historical FAQ Bot",
        "version": "1.0.0",
        "description": "基于 RAG 架构的历史人物问答系统",
        "config": {
            "llm_mode": llm_mode
        }
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量读取启动配置
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"启动服务：{host}:{port}, reload={reload}")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )
```

### 2. 流式输出接口

```python
# 在 main.py 中添加

class StreamQueryRequest(BaseModel):
    """流式查询请求"""
    question: str


@app.post("/api/query/stream")
async def query_stream(request: StreamQueryRequest):
    """
    流式查询（SSE 输出）
    
    Args:
        request: 包含问题的请求
    
    Returns:
        Server-Sent Events 流
    """
    try:
        logger.info(f"收到流式查询：{request.question[:50]}...")
        
        async def generate():
            """生成 SSE 流"""
            # 这里需要 ChatEngine 支持异步流式生成
            # 简化示例：直接返回完整回答
            result = chat_engine.chat(request.question)
            
            # SSE 格式
            yield f"data: {result['answer']}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    except Exception as e:
        logger.error(f"流式查询错误：{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

### 3. 启动脚本 (scripts/start_server.py)

```python
#!/usr/bin/env python3
"""快速启动脚本"""

import os
import sys
import subprocess

def start_server():
    """启动服务"""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"启动 Historical FAQ Bot 服务...")
    print(f"地址：http://{host}:{port}")
    print(f"文档：http://{host}:{port}/docs")
    print()
    
    # 启动 uvicorn
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host", host,
        "--port", str(port),
        "--reload" if os.getenv("API_RELOAD") == "true" else "--no-reload"
    ]
    
    subprocess.run(cmd)


if __name__ == "__main__":
    start_server()
```

### 4. 测试 API

```python
# test_api.py

import requests

BASE_URL = "http://localhost:8000"

# 1. 健康检查
print("=" * 50)
print("测试健康检查")
print("=" * 50)

response = requests.get(f"{BASE_URL}/api/health")
print(f"状态：{response.json()['status']}")
print(f"服务：{response.json()['service']}")

# 2. 查询接口
print("\n" + "=" * 50)
print("测试查询接口")
print("=" * 50)

query_data = {
    "question": "王洪文是谁？",
    "history": []
}

response = requests.post(f"{BASE_URL}/api/query", json=query_data)
result = response.json()

print(f"回答：{result['answer'][:100]}...")
print(f"检索模式：{result['search_type']}")
print(f"置信度：{result['confidence']:.3f}")
print(f"来源数量：{len(result['sources'])}")

# 3. 服务信息
print("\n" + "=" * 50)
print("服务信息")
print("=" * 50)

response = requests.get(f"{BASE_URL}/api/info")
print(response.json())
```

### 5. 使用 curl 测试

```bash
# 健康检查
curl http://localhost:8000/api/health

# 查询接口
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "王洪文是谁？", "history": []}'

# 查看 API 文档
# 浏览器访问：http://localhost:8000/docs
```

## 📝 学习要点

1. **Pydantic 模型**: 自动验证请求数据
2. **异常处理**: 统一返回 HTTP 错误
3. **CORS 配置**: 允许前端跨域访问

## ✅ 检查项

- [ ] 服务能正常启动
- [ ] 健康检查接口正常
- [ ] 查询接口返回正确
- [ ] Swagger 文档可访问

---

## 📋 第 6-10 课总结

| 课时 | 核心产出 | 关键代码 |
|------|----------|----------|
| 第 6 课 | 文档检索模块 | `doc_retriever.py` |
| 第 7 课 | 混合检索路由 | `search_router.py` |
| 第 8 课 | LLM 双模架构 | `llm_factory.py` + 3 个实现 |
| 第 9 课 | 对话引擎 | `chat_engine.py` |
| 第 10 课 | FastAPI 服务 | `main.py` |

**下一步**: 第 11-15 课将实现 Web 前端、LangChain 集成、流式输出等高级功能
