# -*- coding: utf-8 -*-
"""
对话引擎练习版（Practice）

只有注释、逻辑骨架和 TODO，所有具体实现留空。
目标：亲手把 ChatEngine 的每个方法填出来。

完成后应该能跑通：
  engine = ChatEngine(llm_mode="api")
  result = engine.chat("请介绍王洪文")
  print(result["answer"])

以及流式：
  async for chunk in engine.astream("请介绍王洪文"):
      print(chunk, end="", flush=True)
"""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Generator, AsyncGenerator, Optional

import psycopg2
from psycopg2 import sql as psycopg2_sql
from langchain_core.messages import AIMessage, HumanMessage

#TODO 1: 从正确路径导入这些依赖
from src.retrieval.search_router_practice import SearchRouter
from src.retrieval.faq_retriever_practice import FAQRetriever
from src.retrieval.doc_retriever_practice import DocRetriever
from src.llm.standard_llm_new import LLMError, LLMUnavailableError, StandardLLM
from src.chat.response_generator import ResponseGenerator
from src.vectorstore.pg_pool_practice import get_cursor,get_connection
from src.embedding.embedding_local_practice import get_embedding as _shared_get_embedding
from config.pg_config import PG_CHAT_TABLE


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 自定义异常（细分错误类型，方便上层处理）
# ---------------------------------------------------------------------------

class DatabaseError(Exception):
    """数据库操作异常（连接失败、表不存在、SQL 错误等）"""
    pass


class RetrievalError(Exception):
    """检索异常（向量检索失败、BM25 报错等）"""
    pass


class GenerationError(Exception):
    """生成异常（LLM 调用失败、超时等）"""
    pass


# ---------------------------------------------------------------------------
# 工具：给 AsyncIterator 加逐 chunk 超时（兼容 Python 3.10）
# ---------------------------------------------------------------------------

async def _aiter_with_timeout(
    async_iter: AsyncGenerator,
    timeout: float,
    timeout_msg: str = "\n[系统提示：生成超时，回答可能被截断]",
):
    """
    包装 AsyncGenerator，每次取 chunk 加超时。

    注意：不能用 asyncio.wait_for 直接包整个 AsyncGenerator，
    因为 wait_for 只接受 coroutine/Future/Task。
    这里对每个 __anext__() 调用包 wait_for，实现逐 chunk 超时。

    TODO 2: 实现逐 chunk 超时逻辑
      - 循环调用 await asyncio.wait_for(async_iter.__anext__(), timeout)
      - 如果 TimeoutError：yield timeout_msg 然后 break
      - 如果 StopAsyncIteration：break
      - 正常 chunk：yield chunk
    """
    # TODO: 实现
    pass


# ---------------------------------------------------------------------------
# ChatEngine
# ---------------------------------------------------------------------------

class ChatEngine:
    """
    对话引擎

    核心流程：
    1. 接收用户查询
    2. 从 PG 加载对话历史
    3. 检索相关内容（FAQ/文档）
    4. 调用 LLM 生成回答（带 history 注入）
    5. 保存本轮对话到 PG
    6. 返回结构化结果（含耗时、错误码）
    """

    # 默认 LLM 调用超时（秒）
    DEFAULT_LLM_TIMEOUT = 60

    def __init__(
        self,
        search_router: Optional[SearchRouter] = None,
        response_gen: Optional[ResponseGenerator] = None,
        llm_mode: str = None,
        session_id: str = None,
        history_limit: int = 10,
        llm_timeout: float = DEFAULT_LLM_TIMEOUT,
        enable_multi_query: bool = None,
    ):
        """
        初始化对话引擎

        Args:
            search_router: 检索路由器实例（依赖注入，可选）
            response_gen: 回答生成器实例（依赖注入，可选）
            llm_mode: LLM 模式（"local"/"api"），None 则用配置默认值
            session_id: 会话 ID，用于标识对话，None 则自动生成
            history_limit: 每次加载的历史消息条数（默认 10 条）
            llm_timeout: LLM 调用超时时间（秒，默认 60）
            enable_multi_query: 是否启用多查询扩展          
        TODO 3: 初始化成员变量
          - search_router：没有就调 _default_search_router() 构建默认的
          - response_gen：没有就 new 一个 ResponseGenerator(llm_mode=llm_mode)
          - llm_mode, session_id（没有就用 uuid.uuid4()）, history_limit, llm_timeout
          - 打 info 日志记录初始化参数
        """
        self.search_router = search_router or self._default_search_router(enable_multi_query)
        self.response_gen = response_gen or ResponseGenerator(llm_mode=llm_mode)
        self.llm_mode = llm_mode
        self.session_id = session_id or str(uuid.uuid4())
        self.history_limit = history_limit
        self.llm_timeout = llm_timeout
        self.enable_multi_query = enable_multi_query
        
        logger.info(
            "[ChatEngine] 初始化完成 | mode=%s | session=%s | mq=%s | history_limit=%d | llm_timeout=%.0fs",
            llm_mode or "default",
            self.session_id,
            enable_multi_query,
            history_limit,
            llm_timeout,
        )

    @staticmethod
    def _default_search_router(enable_multi_query: bool = False) -> SearchRouter:
        """
        构建默认的 SearchRouter（内部组装 FAQ + Doc 检索器）

        TODO 4: 组装检索器
          - FAQRetriever(top_k=3)
          - DocRetriever(top_k=10, use_bm25=True, fusion_method="rrf", rrf_k=60)
          - 返回 SearchRouter(faq_retriever=..., doc_retriever=...)
        """
        faq_retriever = FAQRetriever(top_k=3)
        doc_retriever = DocRetriever(top_k=10, 
                                    use_bm25=True,
                                    fusion_method="rrf",
                                    rrf_k=60,
                                    enable_multi_query=enable_multi_query,
                                    embedding_fn=_shared_get_embedding)
        return SearchRouter(faq_retriever=faq_retriever, doc_retriever=doc_retriever)

    # -----------------------------------------------------------------------
    # 对话历史读写（SQL 用 psycopg2.sql.Identifier 安全拼接表名）
    # -----------------------------------------------------------------------

    def _load_history(self) -> List[Dict[str, str]]:
        """
        从 PG 加载最近 N 条对话历史

        Returns:
            [{"role": "human"/"ai", "content": "..."}, ...]  按时间正序排列

        TODO 5: 实现历史加载
          - 用 get_cursor() 获取 cursor
          - SQL：SELECT role, content FROM {表名} WHERE session_id = %s ORDER BY created_at DESC LIMIT %s
          - 表名用 psycopg2_sql.Identifier(PG_CHAT_TABLE) 防注入
          - 参数：(self.session_id, self.history_limit)
          - fetchall() 后 reversed() 转成正序
          - psycopg2.Error 抛 DatabaseError
          - 其他异常：打 warning 日志，返回空列表
        """
        try:
            with get_cursor() as cursor:
                query = psycopg2_sql.SQL(
                    """
                    SELECT role , content FROM {}
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """
                ).format(psycopg2_sql.Identifier(PG_CHAT_TABLE))
                cursor.execute(query, (self.session_id, self.history_limit))
                rows = cursor.fetchall()
                history = [{"role": row[0], "content": row[1]} for row in reversed(rows)]
                logger.debug(
                    "[ChatEngine] 加载历史 | session=%s | 条数=%d",
                    self.session_id, len(history)
                )
                return history
        except psycopg2.Error as e:
            # 数据库错误：向上抛 DatabaseError，让上层决定怎么处理
            raise DatabaseError(f"加载历史失败: {e}") from e

    def _save_history(self, role: str, content: str) -> None:
        """
        保存单条消息到 PG

        Args:
            role: 'human' 或 'ai'
            content: 消息内容

        TODO 6: 实现历史保存
          - SQL：INSERT INTO {表名} (session_id, role, content) VALUES (%s, %s, %s)
          - 参数：(self.session_id, role, content)
          - psycopg2.Error 抛 DatabaseError
          - 其他异常：打 warning 日志
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    query = psycopg2_sql.SQL("""
                    INSERT INTO {} (session_id, role, content)
                    VALUES (%s, %s, %s)"""                        
                    ).format(psycopg2_sql.Identifier(PG_CHAT_TABLE))
                    cursor.execute(query, (self.session_id, role, content))
        except psycopg2.Error as e:
            logger.warning(f"保存历史时发生异常: {e}")
            raise DatabaseError(f"保存历史失败: {e}") from e

    def _history_to_messages(self, history: List[Dict[str, str]]) -> List[Any]:
        """
        将历史记录转换为 LangChain Message 列表

        Args:
            history: [{"role": "human"/"ai", "content": "..."}, ...]

        Returns:
            [HumanMessage, AIMessage, ...]

        TODO 7: 遍历 history，role == "human" 转 HumanMessage，role == "ai" 转 AIMessage
        """
        messages = []
        for item in history:
            role = item.get("role","").lower()
            content = item.get("content","").lower()
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai": 
                messages.append(AIMessage(content=content))
        return messages
        # -----------------------------------------------------------------------
        # P0: Query Rewriting（查询改写 / 指代消解）
        # -----------------------------------------------------------------------

    def _rewrite_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        【P0】根据对话历史改写当前查询，消除指代歧义（他/她/这/那）。

        Args:
            query: 当前用户查询
            history: 对话历史 [{"role": ..., "content": ...}, ...]

        Returns:
            改写后的查询。如果没有历史或无需改写，原样返回 query。

        TODO 8: 实现指代消解
          - 如果没有 history，直接返回 query
          - 取最近 3 轮历史，格式化成 "用户：xxx\nAI：xxx"
          - 构建 prompt，让 LLM 改写查询消除歧义
          - 调 StandardLLM.invoke(prompt) 获取改写结果
          - 如果失败或结果为空，回退到原 query
        """
        # TODO: 实现
        pass

    async def _arewrite_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        【P0】异步版本的 _rewrite_query。

        TODO 9: 实现异步指代消解
          - 逻辑和 _rewrite_query 一样，但调 StandardLLM.ainvoke()
        """
        if history is None or len(history) == 0:
            return query
        history_text = '\n'.join(
            [f"{'用户' if h['role'] == 'human' else 'AI'}：{h['content'][:100]}" 
             for h in history[-3:]]
        )
        prompt = f"""请根据以下对话历史，改写当前用户查询，消除指代歧义（如"他/她/它/这/那"等）。
                    如果当前查询已经明确具体，无需改写，请直接原样返回。
                    对话历史：
                    {history_text}
                    当前查询：{query}
                    改写后的查询（只输出改写结果，不要解释）："""
        try:
            resp = await StandardLLM.ainvoke(prompt, mode=self.llm_mode)
            rewritten = resp.content.strip()
            return rewritten if rewritten else query
        except Exception as e:  # ← 改成 Exception，不是 LLMError
            logger.warning(f"[ChatEngine] 查询改写失败: {e}")
            return query
    # -----------------------------------------------------------------------
    # 核心对话方法（同步）
    # -----------------------------------------------------------------------

    async def achat(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询（标准模式，带 history + 兜底 + 耗时统计）

        Returns:
            {
                "answer": str,
                "sources": list,
                "search_type": str,
                "confidence": float,
                "session_id": str,
                "error_code": str | None,
                "latency_ms": float,
            }

        TODO 10: 实现完整的 achat 流程（这是核心方法）
        """
        t0 = time.perf_counter()
        error_code = None
        answer = ""
        sources = []
        citations = [] 
        search_type = "unknown"
        confidence = 0.0
        search_context = None
        try:
             # 1. 加载历史（同步 DB → 线程池）
            history = await asyncio.to_thread(self._load_history)
            history_messages = self._history_to_messages(history)
            # 2. Query Rewriting（异步版本）
            rewritten_query = await self._arewrite_query(query, history)
            
            # 3. 检索 —— 这里接上 async！
            try:
                search_context = await self.search_router.asearch(rewritten_query)
                search_type = search_context.search_type.value
                confidence = search_context.confidence
            except Exception as e:
                raise RetrievalError(f"检索失败: {e}") from e
            # 4. 生成回答
            if search_context.faq_results and not search_context.doc_results:
                # 只有 FAQ 结果，直接返回 FAQ 答案
                answer = search_context.faq_results[0].answer
                sources = [{
                    "type": "faq",
                    "question": search_context.faq_results[0].question,
                    "confidence": search_context.faq_results[0].similarity
                }]        
        except DatabaseError as e:
            error_code = "DATABASE_ERROR"
            answer = "抱歉，系统遇到数据库问题，无法处理您的请求。请稍后再试。"
    # -----------------------------------------------------------------------
    # 流式（同步）
    # -----------------------------------------------------------------------

    def stream(self, query: str) -> Generator[str, None, None]:
        """
        同步流式生成回答（带 history + 异常收窄）

        Yields:
            生成的文本片段

        TODO 11: 实现同步流式
          - 加载历史（try/except DatabaseError → yield 错误提示 + return）
          - 【P0: Query Rewriting】调用 _rewrite_query(query, history) 改写查询
          - 检索（try/except → 降级到 None）——用改写后的 query 检索
          - 如果是 faq_only：yield 整段答案，记录到 chunks
          - 否则：
              - 调 self.response_gen.generate_stream(query, faq_results, doc_results, history_messages)
                逐 chunk yield
              - 每个 chunk yield 出去，同时 append 到 chunks
              - LLMError 捕获 → yield 错误提示
          - finally：
              - "".join(chunks) 得到完整回答
              - 【P0: Citation Tracking】调 extract_citations(full_answer) 解析引用（打日志即可）
              - 保存 history（human + full_answer）
              - 保存失败打 warning
        """
        # TODO: 实现
        pass

    # -----------------------------------------------------------------------
    # 流式（异步）—— 致命修复：同步调用包 asyncio.to_thread() + 逐 chunk 超时
    # -----------------------------------------------------------------------

    async def astream(self, query: str) -> AsyncGenerator[str, None]:
        """
        异步流式生成回答（带 history）

        致命修复：
        - _load_history / search_router.search 是同步的，必须包 to_thread
        - 不用 asyncio.wait_for 包整个 AsyncGenerator（会抛 TypeError）
        - 改用 _aiter_with_timeout 对每个 __anext__() 加超时

        TODO 12: 实现异步流式（最难的部分）

        步骤：
        1. 加载历史（同步操作 → asyncio.to_thread(self._load_history)）
           - DatabaseError → yield 错误提示 + return
        2. 转 history_messages
        3. 【P0: Query Rewriting】调用 _arewrite_query(query, history) 异步改写查询
        4. 检索（同步 → asyncio.to_thread(self.search_router.search, rewritten_query)）
           - 失败打 error 日志，search_context = None
        5. 如果是 faq_only：yield 整段答案，append 到 chunks
        6. 否则：
             - 用 self.response_gen.agenerate_stream(query, faq_results, doc_results, history_messages)
               得到 llm_stream
             - 用 _aiter_with_timeout(llm_stream, timeout=self.llm_timeout) 包装
             - async for chunk in 包装后的流：
                 yield chunk
                 append 到 chunks
             - 捕获 LLMError → yield 错误提示
        7. finally：
             - join chunks 得完整回答
             - 【P0: Citation Tracking】调 extract_citations(full_answer) 解析引用（打日志即可）
             - asyncio.to_thread(self._save_history, ...) 保存 human + ai
             - 失败打 warning
        """
        

    # -----------------------------------------------------------------------
    # 工具方法
    # -----------------------------------------------------------------------

    def clear_memory(self):
        """
        清除 LLM 缓存

        TODO 13: 调 StandardLLM.clear_cache() 并打 info 日志
        """
        # TODO: 实现
        pass


# 兼容旧版接口

def get_chat_engine(
    llm_mode: str = None,
    session_id: str = None,
    search_router=None,
    response_gen=None,
) -> "ChatEngine":
    """
    获取对话引擎实例（工厂函数）

    TODO 14: 返回 ChatEngine(...) 实例，把参数透传进去
    """
    # TODO: 实现
    pass
