# 对话历史持久化 - 方案 A 实现说明

> **方案 A**：使用 LangChain `PostgresChatMessageHistory` 原生支持（推荐，改动最小）

---

## 1. 方案对比

| 维度 | 方案 A (LangChain 原生) | 方案 B (手动存储) |
|------|----------------------|------------------|
| **表结构** | LangChain 标准格式 (`role`, `content`) | 自定义格式 (业务字段) |
| **实现难度** | 简单（2 行代码） | 较复杂（需写 SQL） |
| **数据内容** | 仅对话文本 | 对话文本 + 检索 ID + LLM 模式等 |
| **与 LangChain 集成** | 无缝 | 需要手动同步 |
| **适用场景** | 快速实现、生产环境 | 需要深度数据分析 |

---

## 2. 表结构变更

### 旧表结构（方案 B）
```sql
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    user_query TEXT NOT NULL,
    retrieved_faq_ids INTEGER[],    -- 业务字段
    retrieved_doc_ids INTEGER[],    -- 业务字段
    llm_response TEXT,
    llm_mode VARCHAR(20),           -- 业务字段
    created_at TIMESTAMP
);
```

### 新表结构（方案 A - LangChain 标准）
```sql
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL,        -- 'human' 或 'ai'
    content TEXT NOT NULL,            -- 消息内容
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 会话索引（提高查询性能）
CREATE INDEX idx_chat_history_session ON chat_history (session_id);
```

---

## 3. 代码实现

### 3.1 修改 `pg_schema.py`

```python
# 创建对话历史表（LangChain 标准格式）
cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {PG_CHAT_TABLE} (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(100) NOT NULL,
        role VARCHAR(20) NOT NULL,        -- 'human' 或 'ai'
        content TEXT NOT NULL,            -- 消息内容
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

# 创建会话索引，提高查询性能
cursor.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_chat_history_session 
    ON {PG_CHAT_TABLE} (session_id);
""")
```

### 3.2 修改 `chat_engine.py`

```python
import uuid
from langchain.memory import PostgresChatMessageHistory
from config.pg_config import PG_URL, PG_CHAT_TABLE

class ChatEngine:
    def __init__(self, 
                 llm_mode: str = None,
                 session_id: str = None,  # 新增参数
                 ...):
        
        # 会话 ID（用于持久化对话历史）
        self.session_id = session_id or str(uuid.uuid4())
        
        # ... 其他初始化代码 ...
        
    def chat(self, query: str) -> Dict:
        # ... 生成回答 ...
        
        return {
            "answer": answer,
            "session_id": self.session_id,  # 返回 session_id
            # ...
        }
```

### 3.3 修改 `langchain_integration.py`

```python
from langchain.memory import PostgresChatMessageHistory
from config.pg_config import PG_URL, PG_CHAT_TABLE

class LangChainIntegration:
    def __init__(self, 
                 llm_mode=None, 
                 use_advanced_retriever=True,
                 session_id: str = None):  # 新增参数
        
        self.session_id = session_id
        
        # 使用 PostgreSQL 持久化存储对话历史
        if session_id:
            self.memory = PostgresChatMessageHistory(
                connection_string=PG_URL,
                session_id=session_id,
                table_name=PG_CHAT_TABLE
            )
            logger.info(f"使用 PostgreSQL 持久化对话历史: {session_id}")
        else:
            # 降级为内存存储
            self.memory = ConversationBufferMemory(...)
            logger.warning("未提供 session_id，使用内存存储")
```

---

## 4. 使用方式

### 4.1 新对话（自动生成 session_id）

```python
from src.chat.chat_engine import ChatEngine

# 创建新对话引擎（自动生成 session_id）
engine = ChatEngine(llm_mode="local")

# 第一轮对话
result = engine.chat("王洪文是谁？")
print(f"Session ID: {result['session_id']}")  # 例如：550e8400-e29b-41d4-a716-446655440000

# 第二轮对话（自动带上下文）
result = engine.chat("他后来怎样了？")  # 能引用上文
```

### 4.2 继续已有对话（传入 session_id）

```python
# 使用已有 session_id 恢复对话
engine = ChatEngine(
    llm_mode="local",
    session_id="550e8400-e29b-41d4-a716-446655440000"
)

# 继续之前的对话
result = engine.chat("接着刚才的说")
```

### 4.3 API 接口使用

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: dict):
    query = request.get("query")
    session_id = request.get("session_id")  # 可选，不传则新建
    
    engine = ChatEngine(
        llm_mode="local",
        session_id=session_id
    )
    
    result = engine.chat(query)
    
    return {
        "answer": result["answer"],
        "session_id": result["session_id"]  # 返回给前端保存
    }
```

---

## 5. 数据库中的数据示例

```sql
-- 查看某会话的历史
SELECT * FROM chat_history 
WHERE session_id = '550e8400-e29b-41d4-a716-446655440000'
ORDER BY created_at;

-- 结果示例：
-- id | session_id | role   | content           | created_at
-- 1  | 550e84...  | human  | 王洪文是谁？       | 2026-04-04 10:00:00
-- 2  | 550e84...  | ai     | 王洪文（1935年...  | 2026-04-04 10:00:05
-- 3  | 550e84...  | human  | 他后来怎样了？     | 2026-04-04 10:01:00
-- 4  | 550e84...  | ai     | 他在1976年被...    | 2026-04-04 10:01:08
```

---

## 6. 常见问题

### Q1: 是否需要维护 SQL 里的对话历史表？

**A**: 表结构需要创建（`pg_schema.py` 会自动创建），但不需要手动维护数据。LangChain 的 `PostgresChatMessageHistory` 会自动处理：
- `INSERT`：发送消息时自动插入
- `SELECT`：读取历史时自动查询
- 无需手动编写 SQL

### Q2: 服务重启后对话历史会丢失吗？

**A**: **不会丢失**。方案 A 的核心优势就是**持久化**：
- 内存存储（`ConversationBufferMemory`）：重启丢失
- PostgreSQL 存储（`PostgresChatMessageHistory`）：重启保留

### Q3: 如何清理历史记录？

```python
# 删除特定会话
with get_cursor() as cursor:
    cursor.execute(
        "DELETE FROM chat_history WHERE session_id = %s",
        (session_id,)
    )

# 清理 30 天前的记录
cursor.execute(
    "DELETE FROM chat_history WHERE created_at < NOW() - INTERVAL '30 days'"
)
```

### Q4: 方案 A vs 方案 B 怎么选？

**选方案 A（当前实现）**：
- 快速上线，不想写复杂 SQL
- 主要需求是对话上下文恢复
- 不需要分析检索中间过程

**选方案 B**：
- 需要记录 `retrieved_faq_ids` 等业务字段
- 需要分析"为什么这个回答不好"（基于检索日志优化）
- 需要 A/B 测试不同检索策略的效果

---

## 7. 总结

**方案 A 核心特点**：
1. **极简实现**：2 行代码开启持久化
2. **LangChain 原生**：无缝集成，自动管理
3. **生产可用**：服务重启不丢失对话
4. **局限性**：只能存 `role` + `content`，丢失业务中间数据

**是否需要维护 SQL 表？**
- ✅ 需要创建表（`pg_schema.py` 已处理）
- ❌ 不需要手动维护数据（LangChain 自动处理）
- ✅ 可以查询/清理数据（直接 SQL 操作）
