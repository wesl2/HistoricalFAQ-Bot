# HistoricalFAQ-Bot v2.0 新功能说明

## 🎉 版本 2.0 重大更新

本次更新整合了 LangChain 生态，同时保留了原生实现的优势，实现了**双模架构**。

---

## ✨ 新增功能

### 1. 向量库持久化

**功能描述**：
- 向量数据自动保存到磁盘，服务重启后无需重新加载
- 支持 Chroma 和 FAISS 两种向量库

**使用方式**：
```python
from src.rag.advanced_retriever import get_advanced_retriever

retriever = get_advanced_retriever()
retriever.create_vectorstore(documents, vectorstore_type="chroma")
# 自动持久化到 ./vectorstore/chroma/
```

**配置文件**：
```bash
# .env
VECTORSTORE_PERSIST_DIR=./vectorstore
VECTORSTORE_TYPE=chroma
```

---

### 2. Streaming 流式输出

**功能描述**：
- 支持 SSE (Server-Sent Events) 流式输出
- 实时显示生成内容，提升用户体验

**API 接口**：
```bash
# 流式查询
POST /api/query/stream
Content-Type: application/json

{
    "question": "王洪文是谁？"
}

# 返回: text/event-stream
```

**前端接入示例**：
```javascript
const eventSource = new EventSource('/api/query/stream', {
    method: 'POST',
    body: JSON.stringify({question: '王洪文是谁？'})
});

eventSource.onmessage = (event) => {
    if (event.data === '[DONE]') {
        eventSource.close();
    } else {
        console.log(event.data); // 实时显示
    }
};
```

---

### 3. Multi-Query 检索

**功能描述**：
- 自动生成查询的多个变体，提升召回率
- 基于 LLM 生成语义相似的查询

**工作原理**：
```
用户查询: "王洪文的生平"
    ↓
LLM 生成变体:
    - "王洪文的生平"
    - "王洪文是谁，他的生平如何"
    - "介绍一下王洪文的历史"
    ↓
并行检索 → 合并去重 → 返回结果
```

**启用方式**：
```python
from src.rag.advanced_retriever import AdvancedRetriever

retriever = AdvancedRetriever(use_multi_query=True)
results = retriever.retrieve("查询", k=5)
```

---

### 4. BCE-Reranker 重排序

**功能描述**：
- 使用 BCE-Reranker 交叉编码器对检索结果重排序
- 显著提升结果相关性

**两阶段检索**：
```
第一阶段（召回）: 向量相似度检索 → Top 20
第二阶段（精排）: BCE-Reranker → Top 5
```

**配置**：
```python
# config/model_config.py
RERANKER_CONFIG = {
    "model_path": "/path/to/bce-reranker-base_v1",
    "device": "cuda",
    "top_n": 5,
    "use_fp16": True
}
```

**注意**：如果模型不存在，会自动降级为向量相似度排序。

---

### 5. Prompt 外部化

**功能描述**：
- 提示词模板存储在独立文件中
- 支持动态修改，无需重启服务

**文件结构**：
```
prompts/
├── rag_template.txt           # RAG 提示词
├── conversational_template.txt # 对话提示词
└── multi_query_template.txt    # Multi-query 提示词
```

**自定义提示词**：
```txt
# prompts/rag_template.txt
你是一位专业的历史研究专家...

参考资料：
{context}

问题：
{question}

要求：...
```

---

### 6. Callback 可观测性系统

**功能描述**：
- 自动记录 Chain 执行流程
- 统计 Token 使用量和成本
- 性能指标监控

**监控指标**：
```json
{
    "performance": {
        "chain_count": 10,
        "llm_count": 10,
        "avg_total_latency": 2.5,
        "avg_llm_latency": 1.8
    },
    "token_usage": {
        "total_tokens": 5000,
        "estimated_cost_usd": 0.01
    }
}
```

**API 接口**：
```bash
# 获取指标
GET /api/metrics

# 重置指标
POST /api/metrics/reset
```

**日志文件**：
```
logs/callback_YYYYMMDD.jsonl
```

---

## 🚀 快速开始

### 1. 启动服务（推荐）

```bash
# 启用所有功能（LangChain + 高级检索器）
python scripts/start_server.py

# 或者使用环境变量
USE_LANGCHAIN=true \
USE_ADVANCED_RETRIEVER=true \
python -m src.api.main
```

### 2. 启动服务（原生模式）

```bash
# 仅使用原生实现（无 LangChain）
python scripts/start_server.py --native
```

### 3. 测试新功能

```bash
# 运行功能测试
python scripts/test_new_features.py
```

---

## 📊 功能对比

| 功能 | v1.0 (原生) | v2.0 (LangChain) | 说明 |
|------|------------|-----------------|------|
| FAQ + RAG 检索 | ✅ | ✅ | v2.0 保留策略路由 |
| 向量库 | PostgreSQL/pgvector | Chroma/FAISS | v2.0 支持持久化 |
| Multi-Query | ❌ | ✅ | v2.0 新增 |
| Rerank | ❌ | ✅ | v2.0 新增 BCE-Reranker |
| 流式输出 | ❌ | ✅ | v2.0 新增 |
| 对话记忆 | ❌ | ✅ | v2.0 新增 |
| Agent | ❌ | ✅ | v2.0 新增 |
| 监控指标 | ❌ | ✅ | v2.0 新增 |
| Prompt 管理 | 硬编码 | 外部文件 | v2.0 更灵活 |
| Embedding 微调 | ✅ | ✅ | 两者都支持 |

---

## 🔧 环境变量配置

```bash
# 核心功能开关
USE_LANGCHAIN=true              # 启用 LangChain
USE_ADVANCED_RETRIEVER=true     # 启用高级检索器
RERANKER_ENABLED=true           # 启用 Reranker

# LangChain 配置
LANGCHAIN_CHAIN_TYPE=rag        # 链类型: rag, conversational
VECTORSTORE_TYPE=chroma         # 向量库类型: chroma, faiss
VECTORSTORE_PERSIST_DIR=./vectorstore

# 文档处理
DOCUMENT_CHUNK_SIZE=1000
DOCUMENT_CHUNK_OVERLAP=200

# 记忆配置
MEMORY_TYPE=buffer              # buffer 或 summary
MEMORY_ENABLED=true

# 服务配置
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 📁 新增文件结构

```
HistoricalFAQ-Bot/
├── prompts/                       # 新增：提示词模板
│   ├── rag_template.txt
│   ├── conversational_template.txt
│   └── multi_query_template.txt
├── src/
│   ├── rag/
│   │   ├── langchain_integration.py  # 更新：支持 Streaming
│   │   ├── advanced_retriever.py     # 新增：Multi-Query + Rerank
│   │   └── callbacks.py              # 新增：可观测性系统
│   └── chat/
│       └── chat_engine.py            # 更新：整合新功能
├── scripts/
│   ├── start_server.py               # 新增：快速启动脚本
│   └── test_new_features.py          # 新增：功能测试脚本
├── logs/                             # 新增：日志目录
└── vectorstore/                      # 新增：向量库持久化目录
```

---

## 🎯 适用场景推荐

### 场景 1：快速原型开发
**推荐配置**：`USE_LANGCHAIN=true`, `USE_ADVANCED_RETRIEVER=false`
- 快速搭建，配置简单
- 使用基础 RAG Chain

### 场景 2：生产环境
**推荐配置**：`USE_LANGCHAIN=true`, `USE_ADVANCED_RETRIEVER=true`
- 启用 Multi-Query + Rerank，提升准确性
- 向量库持久化，服务快速启动
- 监控指标，便于运维

### 场景 3：性能敏感
**推荐配置**：`USE_LANGCHAIN=false`
- 原生实现，延迟最低
- 手动优化每个环节

---

## 💡 最佳实践

### 1. 首次启动
```bash
# 1. 启动服务
python scripts/start_server.py

# 2. 运行测试
python scripts/test_new_features.py

# 3. 查看监控指标
curl http://localhost:8000/api/metrics
```

### 2. 向量库管理
```python
from src.rag.advanced_retriever import get_advanced_retriever

retriever = get_advanced_retriever()

# 添加新文档
retriever.add_documents(new_documents)

# 手动保存（通常自动保存）
retriever.vectorstore.persist()
```

### 3. 自定义提示词
```bash
# 1. 编辑 prompts/rag_template.txt
# 2. 重启服务（无需重新加载模型）
# 3. 新提示词立即生效
```

---

## 🔍 调试技巧

### 查看详细日志
```bash
# 设置日志级别
export LOG_LEVEL=DEBUG
python scripts/start_server.py
```

### 检查回调日志
```bash
# 实时查看执行流程
tail -f logs/callback_$(date +%Y%m%d).jsonl | jq .
```

### 测试特定功能
```python
# 测试高级检索器
from src.rag.advanced_retriever import get_advanced_retriever
retriever = get_advanced_retriever()
results = retriever.retrieve("测试查询")

# 测试流式输出
from src.chat.chat_engine import ChatEngine
engine = ChatEngine(use_langchain=True)
for chunk in engine.stream("测试"):
    print(chunk, end="")
```

---

## 📝 更新日志

### v2.0.0 (2024-03-25)
- ✨ 新增向量库持久化
- ✨ 新增 Streaming 流式输出
- ✨ 新增 Multi-Query 检索
- ✨ 新增 BCE-Reranker 重排序
- ✨ 新增 Prompt 外部化管理
- ✨ 新增 Callback 可观测性系统
- ✨ 新增 Agent 功能支持
- ✨ 新增对话记忆功能
- 🔧 重构 ChatEngine 支持双模架构
- 🔧 优化 API 接口，新增多个端点
- 📝 完善文档和测试脚本

---

## 🤝 贡献指南

欢迎提交 Issue 和 PR！

---

## 📄 许可证

MIT License
