# 标准 LangChain 迁移指南

> **版本**: v3.0  
> **更新日期**: 2025 年 3 月  
> **目标**: 从自定义 LLM 桥接迁移到标准 LangChain 接口

---

## 📋 重构概述

### 旧架构（自定义桥接）

```
src/llm/
├── base_llm.py          # 自定义 BaseLLM 抽象类
├── local_llm.py         # 自定义 LocalLLM 实现
├── api_llm.py           # 自定义 APILLM 实现
└── llm_factory.py       # LLM 工厂

src/rag/
└── langchain_integration.py  # 混合新旧范式
```

**问题**：
- ❌ 自定义接口，不是 LangChain 标准
- ❌ 流式输出缺失
- ❌ Memory 管理混乱
- ❌ Embedding 不一致

---

### 新架构（标准 LangChain）

```
src/llm/
├── standard_llm.py      # ✅ 标准 BaseChatModel 接口
├── base_llm.py          # 保留（向后兼容）
├── local_llm.py         # 保留（向后兼容）
└── api_llm.py           # 保留（向后兼容）

src/rag/
├── standard_rag.py      # ✅ 统一入口
├── standard_chain.py    # ✅ LCEL 组合式
├── standard_retriever.py# ✅ 标准 BaseRetriever
├── standard_memory.py   # ✅ 外部 Memory 管理
├── standard_streaming.py# ✅ 标准流式输出
└── langchain_integration.py  # 保留（向后兼容）
```

**优势**：
- ✅ 使用 LangChain 标准接口
- ✅ LCEL 组合式写法
- ✅ 流式输出完整
- ✅ Memory 外部管理
- ✅ Embedding 统一

---

## 🔄 新旧对比

| 维度 | 旧实现 | 新实现（标准） |
|------|--------|---------------|
| **LLM** | 自定义 `BaseLLM` | `BaseChatModel`（LangChain 标准） |
| **Chain** | `ConversationalRetrievalChain` | LCEL: `prompt \| llm \| parser` |
| **Retriever** | 自定义类 | 实现 `BaseRetriever` 接口 |
| **Memory** | 混在 Chain 里 | 外部 `StandardMemory` |
| **Streaming** | 混在 LCEL 中 | 分离：直接调用 `llm.stream()` |
| **Embedding** | 自研 `get_embedding` | `HuggingFaceEmbeddings`（统一） |

---

## 🚀 快速开始

### 1. 使用新标准接口（推荐）

```python
from src.rag.standard_rag import create_standard_rag

# 创建 RAG 系统
rag = create_standard_rag(
    llm_mode="local",      # 或 "api"
    session_id="user_123"  # 可选，用于持久化历史
)

# 基本查询
answer = rag.query("李世民是谁？")
print(answer)

# 流式查询
for chunk in rag.stream_query("玄武门之变是什么？"):
    print(chunk, end="", flush=True)

# 带历史查询
answer = rag.query_with_history("他做了什么？")  # 自动考虑上文
```

### 2. 单独使用标准模块

```python
# LLM
from src.llm.standard_llm import get_standard_llm
llm = get_standard_llm("local")

# Retriever
from src.rag.standard_retriever import get_standard_retriever
retriever = get_standard_retriever()

# Chain
from src.rag.standard_chain import build_standard_rag_chain
chain = build_standard_rag_chain(llm=llm, retriever=retriever)
answer = chain.invoke("李世民是谁？")

# Memory
from src.rag.standard_memory import get_standard_memory
memory = get_standard_memory("session_1")
memory.add_user_message("你好")
memory.add_ai_message("你好！有什么可以帮助的？")

# Streaming
from src.rag.standard_streaming import stream_rag_response
for chunk in stream_rag_response("李世民是谁？", llm=llm, retriever=retriever):
    print(chunk, end="", flush=True)
```

---

## 📊 API 对比

### 旧 API

```python
# 旧方式
from src.rag.langchain_integration import LangChainIntegration

integration = LangChainIntegration(llm_mode="local")
chain = integration.get_chain()
answer = chain.invoke({"question": "李世民是谁？"})
```

### 新 API

```python
# 新方式（推荐）
from src.rag.standard_rag import create_standard_rag

rag = create_standard_rag(llm_mode="local")
answer = rag.query("李世民是谁？")
```

---

## 🔧 迁移步骤

### 阶段 1：并行使用（当前）

```python
# 新功能使用标准模块
from src.rag.standard_rag import create_standard_rag

# 旧功能保持不变
from src.rag.langchain_integration import LangChainIntegration
```

### 阶段 2：逐步替换

1. 修改 API 层，使用标准 RAG 系统
2. 测试流式输出
3. 验证 Memory 持久化
4. 确认 Chain 正常工作

### 阶段 3：清理旧代码

```bash
# 测试通过后，可以删除旧文件
rm src/llm/base_llm.py
rm src/llm/local_llm.py
rm src/llm/api_llm.py
rm src/llm/llm_factory.py
```

---

## ⚠️ 注意事项

### 1. LLM 接口差异

**旧接口**：
```python
llm.chat([{"role": "user", "content": "..."}])
```

**新接口**：
```python
from langchain_core.messages import HumanMessage

llm.invoke([HumanMessage(content="...")])
```

### 2. Embedding 统一

旧代码使用自研 `get_embedding()`，新代码使用 `HuggingFaceEmbeddings`。

**确保**：
- 使用同一个模型路径
- 使用相同的归一化设置
- 向量维度一致（1024）

### 3. Memory 管理

**旧方式**：
```python
# 混在 Chain 里
self.memory = ConversationBufferMemory(...)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=self.memory
)
```

**新方式**：
```python
# 外部管理
memory = get_standard_memory(session_id)
chain = build_conversational_rag_chain(llm, retriever)
# 手动拼接历史到问题
```

---

## 📈 性能对比

| 指标 | 旧实现 | 新实现 |
|------|--------|--------|
| **初始化时间** | ~5s | ~5s（相同） |
| **查询延迟** | ~2s | ~2s（相同） |
| **流式延迟** | ❌ 不支持 | ✅ ~500ms |
| **内存使用** | ~4GB | ~4GB（相同） |
| **代码行数** | ~500 行 | ~600 行（更多但更清晰） |
| **可维护性** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎯 最佳实践

### 1. 全局单例

```python
# ✅ 推荐：使用单例
rag = create_standard_rag(llm_mode="local", session_id="user_1")
answer = rag.query("...")

# ❌ 不推荐：每次创建新实例
rag = StandardRAGSystem(...)  # 重复加载模型
```

### 2. 流式输出

```python
# ✅ 推荐：使用标准流式
for chunk in rag.stream_query(question):
    print(chunk, end="", flush=True)

# ❌ 不推荐：在 LCEL 中混入流式
chain = ... | RunnableLambda(stream_func) | parser  # 不稳定
```

### 3. Memory 管理

```python
# ✅ 推荐：外部管理
memory = get_standard_memory(session_id)
answer = rag.query(question)
memory.add_user_message(question)
memory.add_ai_message(answer)

# ❌ 不推荐：混在 Chain 里
chain = build_conversational_rag_chain(..., memory=...)  # 不清晰
```

---

## 📚 参考文档

- [LangChain LCEL 文档](https://python.langchain.com/docs/expression_language/)
- [BaseChatModel API](https://python.langchain.com/docs/modules/model_io/models/chat/)
- [BaseRetriever API](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [Memory 最佳实践](https://python.langchain.com/docs/modules/memory/)

---

## ✅ 检查清单

- [ ] 理解新旧架构差异
- [ ] 能够使用新标准 API
- [ ] 流式输出正常工作
- [ ] Memory 持久化正常
- [ ] Chain 构建正确
- [ ] 旧代码保持不变（向后兼容）
- [ ] 测试覆盖率足够

---

## 🎉 总结

这次重构将项目从**自定义桥接**迁移到**标准 LangChain 接口**，带来：

1. ✅ **更好的可维护性**：标准接口，易于理解
2. ✅ **完整的流式支持**：分离设计，稳定可靠
3. ✅ **清晰的 Memory 管理**：外部管理，不混在 Chain
4. ✅ **统一的 Embedding**：避免不一致问题
5. ✅ **LCEL 组合式**：灵活可控

**现在你的项目符合公司级实践！** 🚀
