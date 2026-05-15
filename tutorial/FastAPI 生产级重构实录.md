# FastAPI 生产级重构实录：从 v2.1 到 v3.0

> **目标**：把 tutorial 里的 FastAPI 最佳实践落地到项目中，让 API 层真正达到"可上线"标准。
> 
> **面试价值**：本文档覆盖的每一个改进点都是后端面试的高频考点。

---

## 一、重构前的问题诊断

### 1.1 原 main.py（v2.1）的问题清单

| # | 问题 | 面试追问 | 风险等级 |
|---|------|---------|---------|
| 1 | `except Exception` 一刀切 | "如果数据库挂了，用户看到什么？" | 🔴 P0 |
| 2 | 健康检查返回静态 JSON | "K8s 探针怎么判断服务真的健康？" | 🔴 P0 |
| 3 | 无请求 ID | "线上报错怎么定位是哪个用户的请求？" | 🟡 P1 |
| 4 | 无并发限流 | "1000 人同时访问，LLM API 会不会被限流？" | 🟡 P1 |
| 5 | SSE 格式不标准 | "前端怎么知道流式输出结束了？" | 🟡 P1 |
| 6 | CORS `allow_origins=["*"]` | "生产环境允许任何域名访问？" | 🟢 P2 |
| 7 | 无 lifespan 管理 | "服务重启时连接池泄漏怎么办？" | 🟢 P2 |
| 8 | 缺少请求耗时统计 | "哪个接口慢？" | 🟢 P2 |

**结论**：骨架有，但缺生产级的"血肉"——异常分层、可观测性、流量控制。

---

## 二、核心改动详解

### 改动 1：error_code 映射（P0）

**关键发现**：`ChatEngine.achat()` 内部已经 catch 了所有异常，不会向上抛出。它返回的是包含 `error_code` 字段的 dict：

```python
# ChatEngine.achat() 内部（不可改）
except DatabaseError as e:
    error_code = "DATABASE_ERROR"
    answer = "系统内部错误（数据库），请稍后再试。"
except RetrievalError as e:
    error_code = "RETRIEVAL_FAILED"
    # ... 降级到纯 LLM
except GenerationError as e:
    error_code = "GENERATION_FAILED"
    # ... FAQ 兜底
except Exception as e:
    error_code = "UNKNOWN_ERROR"
```

**原代码（问题）**：

```python
@app.post("/api/query")
async def query_faq(request: QueryRequest):
    try:
        engine = _get_chat_engine(session_id=request.session_id)
        result = await engine.achat(query=request.question)
        return QueryResponse(**result)
    except Exception as e:          # ← 永远不会触发，achat 已吞掉所有异常
        raise HTTPException(status_code=500, detail=str(e))
```

**新代码（error_code 映射）**：

```python
ERROR_CODE_MAP = {
    "DATABASE_ERROR": (503, "数据库服务暂不可用"),
    "RETRIEVAL_FAILED": (502, "检索服务异常"),
    "LLM_FAILED": (504, "AI 生成服务超时"),
    "GENERATION_FAILED": (500, "生成回答时出错"),
    "UNKNOWN_ERROR": (500, "系统出现未知错误"),
}

@app.post("/api/query")
async def query_faq(request, body):
    result = await engine.achat(query=body.question)
    error_code = result.get("error_code")
    if error_code:
        status_code, message = ERROR_CODE_MAP[error_code]
        return JSONResponse(status_code=status_code, content={
            "error": message,
            "detail": result.get("answer"),
            "error_code": error_code,
        })
    return QueryResponse(**result)
```

**为什么这样设计？**

| error_code | HTTP 状态码 | 语义 | 运维动作 |
|-----------|-----------|------|---------|
| `DATABASE_ERROR` | 503 | Service Unavailable | 检查 PostgreSQL 连接 |
| `RETRIEVAL_FAILED` | 502 | Bad Gateway | 检查向量索引/BM25 缓存 |
| `LLM_FAILED` | 504 | Gateway Timeout | 检查 DeepSeek API 状态 |

**面试考点**：
> Q：为什么不直接用 `@app.exception_handler` 捕获异常？
> 
> A：`ChatEngine.achat()` 内部已经 catch 了所有异常并返回 error_code，不会向上抛。API 层有两种选择：① 修改 ChatEngine（侵入底层），② 检查 error_code 映射状态码（非侵入）。选②是因为底层已经做了降级和容错，API 层只需要把错误信息翻译给客户端。
> 
> 但如果底层会抛异常（如其他团队的模块），异常处理器仍然是最佳实践。

---

### 改动 2：健康检查真实探测（P0）

**原代码（静态文本）**：

```python
@app.get("/api/health")
async def health_check():
    return HealthResponse(status="healthy", ...)  # ← 永远 healthy
```

**问题**：K8s 探针访问 `/api/health`，即使 PostgreSQL 挂了、DeepSeek API 不可达，也返回 200 OK。流量继续打进来，用户全是报错。

**新代码（实际探测）**：

```python
@app.get("/api/health")
async def health_check():
    checks = {}

    # 1. 检查数据库
    pg_ok = await asyncio.to_thread(check_pg_connection)
    checks["database"] = {"status": "ok" if pg_ok else "fail"}

    # 2. 检查 LLM API（轻量探测，不消耗 token）
    llm_ok = await StandardLLM.ahealth_check()
    checks["llm_api"] = {"status": "ok" if llm_ok else "fail"}

    overall = "healthy" if all(c["status"] == "ok" for c in checks.values()) else "degraded"
    return HealthResponse(status=overall, checks=checks, ...)
```

**返回示例**：

```json
{
  "status": "degraded",
  "checks": {
    "database": {"status": "ok", "detail": "PostgreSQL"},
    "llm_api": {"status": "fail", "detail": "Connection timeout"}
  }
}
```

**面试考点**：
> Q：健康检查为什么要探测依赖，而不是只检查自己进程存活？
> 
> A：进程活着不代表能提供服务。微服务的健康探针应该探测所有关键依赖（DB、缓存、下游 API），任一依赖失败就标记 degraded，让负载均衡器把流量切走。

---

### 改动 3：请求追踪 + 中间件（P1）

**原代码**：没有 request_id，日志是散乱的：

```
2025-01-01 10:00:01 收到查询 | session=abc | question=...
2025-01-01 10:00:02 查询完成 | session=abc | latency=1200ms
2025-01-01 10:00:03 收到查询 | session=xyz | question=...
2025-01-01 10:00:05 查询完成 | session=xyz | latency=2100ms
```

**问题**：如果 session 相同（如 `debug_session_001`），日志混在一起，无法区分是哪一次请求。

**新代码（中间件注入 request_id）**：

```python
@app.middleware("http")
async def add_request_context(request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    adapter = logging.LoggerAdapter(logger, {"request_id": request_id})
    adapter.info("→ 请求开始 | %s %s", request.method, request.url.path)

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

**日志效果**：

```
2025-01-01 10:00:01 [a3f7b2d1] → 请求开始 | POST /api/query
2025-01-01 10:00:02 [a3f7b2d1] ← 请求完成 | 200 | 1200ms
2025-01-01 10:00:03 [e8c9d4f2] → 请求开始 | POST /api/query
2025-01-01 10:00:05 [e8c9d4f2] ← 请求完成 | 200 | 2100ms
```

**面试考点**：
> Q：request_id 存在哪里？为什么不用全局变量？
> 
> A：存在 `request.state` 里，这是 FastAPI 的请求作用域存储。不用全局变量是因为 FastAPI 是异步并发的，全局变量会导致 request_id 被覆盖、串台。

---

### 改动 4：并发限流（P1）

**原代码**：无限制，高并发直接打爆 DeepSeek API。

**新代码（Semaphore 限流）**：

```python
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "10"))
_llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

@app.post("/api/query")
async def query_faq(request, body):
    engine = _get_chat_engine(request, session_id=body.session_id)
    async with _llm_semaphore:          # ← 最多 10 个并发 LLM 请求
        result = await engine.achat(query=body.question)
    return QueryResponse(**result)
```

**原理**：
- `asyncio.Semaphore(10)` = 同时最多 10 个协程进入 `async with` 块
- 第 11 个请求在 `async with` 外等待，不会阻塞事件循环（只是挂起）
- 保护的是**下游 LLM API**，不是 CPU

**面试考点**：
> Q：为什么不用 `asyncio.Lock()` 而用 `Semaphore`？
> 
> A：`Lock` 只允许 1 个协程通过，性能浪费。`Semaphore(N)` 允许 N 个并发，既保护下游不被打爆，又充分利用并发能力。这是"限流"不是"互斥"。

---

### 改动 5：SSE 标准化（P1）

**原代码（SSE 格式混乱）**：

```python
async for chunk in engine.astream(request.question):
    yield f"data: {chunk}\n\n"        # ← 只有 data，前端不知道何时结束
yield "data: [DONE]\n\n"
```

**问题**：前端需要字符串匹配 `[DONE]` 来判断结束，没有错误事件的语义。

**新代码（标准 SSE）**：

```python
async def generate():
    try:
        async with _llm_semaphore:
            async for chunk in engine.astream(body.question):
                yield f"event: message\ndata: {chunk}\n\n"
        yield "event: done\ndata: [DONE]\n\n"
    except Exception as exc:
        yield f"event: error\ndata: {error_msg}\n\n"
```

**SSE 事件语义**：

| event | 含义 | 前端处理 |
|-------|------|---------|
| `message` | 正常内容块 | `eventSource.onmessage` 或 `addEventListener('message', ...)` |
| `done` | 流式结束 | 关闭连接、显示"引用来源" |
| `error` | 生成异常 | 显示错误提示、保留已生成的内容 |

**前端对接示例**：

```javascript
const eventSource = new EventSource('/api/query/stream', {
  method: 'POST',
  body: JSON.stringify({question: "..."})
});

eventSource.addEventListener('message', (e) => {
  appendText(e.data);
});

eventSource.addEventListener('done', (e) => {
  eventSource.close();
  showCitations();
});

eventSource.addEventListener('error', (e) => {
  showError(e.data);
  eventSource.close();
});
```

**面试考点**：
> Q：SSE 和 WebSocket 有什么区别？为什么选 SSE？
> 
> A：SSE 是单向推送（服务器→客户端），基于 HTTP，自动重连，前端用 `EventSource` 一行代码搞定。WebSocket 是双向全双工，需要握手和状态管理。RAG 问答只需要服务器推流，SSE 更简单。

---

### 改动 6：Lifespan 生命周期管理（P2）

**原代码**：全局变量在模块导入时初始化，没有关闭逻辑。

```python
# 模块级全局变量，导入即初始化
_search_router = SearchRouter(...)
_response_gen = ResponseGenerator()
```

**问题**：
- 导入时初始化，如果配置错误，程序还没启动就挂了
- 没有关闭逻辑，HTTP client 连接池可能泄漏
- 单元测试时无法 mock（全局变量已经创建了）

**新代码（lifespan 上下文）**：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Lifespan] 启动...")
    app.state.search_router = SearchRouter(...)
    app.state.response_gen = ResponseGenerator()
    yield  # ← 服务运行期间
    logger.info("[Lifespan] 关闭：释放资源...")

app = FastAPI(lifespan=lifespan)
```

**面试考点**：
> Q：`lifespan` 和 `on_event("startup")` 有什么区别？
> 
> A：`on_event` 是旧 API，已被官方标记为弃用。`lifespan` 是 `asynccontextmanager`，支持 async 清理逻辑（如关闭 httpx.AsyncClient），且和测试框架兼容更好。

---

## 三、改动前后对比表

| 维度 | v2.1（重构前） | v3.0（重构后） | 面试关键词 |
|------|--------------|---------------|-----------|
| 异常处理 | `except Exception: 500` | 分层 503/502/504 | **错误码语义化** |
| 健康检查 | 静态 JSON | 探测 DB + LLM | **依赖探活** |
| 请求追踪 | 无 | request_id + 中间件 | **分布式追踪** |
| 并发控制 | 无 | Semaphore(10) | **下游限流** |
| SSE 格式 | `data: chunk` | `event: message/done/error` | **事件语义** |
| 生命周期 | 模块级全局变量 | `lifespan` 上下文 | **资源管理** |
| CORS | `allow_origins=["*"]` | 从环境变量读取 | **安全基线** |
| 日志 | 基础 logging | LoggerAdapter + request_id | **可观测性** |

---

## 四、面试速记卡

**Q：FastAPI 的异步是怎么工作的？**
> 单线程事件循环 + async/await。IO 操作时挂起当前协程，事件循环调度其他就绪协程。通过 `asyncio.to_thread()` 把同步操作（如 psycopg2）隔离到线程池，避免阻塞事件循环。

**Q：为什么 ChatEngine 不能全局单例？**
> ChatEngine 持有 `session_id` 和对话历史，是有状态的。全局单例会导致用户 A 的历史混入用户 B。SearchRouter / ResponseGenerator 无状态，可以全局复用。

**Q：限流用 Semaphore 还是 Lock？**
> Semaphore(N) 允许 N 个并发，适合保护下游 API 不被打爆。Lock 只允许 1 个，会严重降低吞吐量。

**Q：健康检查返回 degraded 有什么用？**
> 让 K8s / 负载均衡器感知到服务"活着但不能服务"，自动把流量切到健康实例，避免用户访问到故障节点。

---

## 五、启动验证

```bash
# 1. 启动服务
python src/api/main.py

# 2. 测试健康检查（应该返回 checks 详情）
curl http://localhost:8000/api/health

# 3. 测试标准问答（带 request_id 返回头）
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "唐太宗的用人之道"}'

# 4. 测试流式问答（SSE 格式）
curl -X POST http://localhost:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "玄武门之变"}'
```

---

*本文档配合 `src/api/main.py` v3.0 使用，建议与 `tutorial/第 6.8 课：数据清洗、引用验证与数字标号` 结合阅读，构成完整的"后端服务 → 数据层 → RAG 链路"面试叙事。*
