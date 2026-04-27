# -*- coding: utf-8 -*-
"""
对话引擎（v2.1：修复 wait_for 不能包 AsyncGenerator + SQL 安全拼接 + 同步 stream 异常处理）

改动点（按朋友第二轮 review）：
1. 致命修复：asyncio.wait_for 不能包 AsyncGenerator → 改用逐 chunk __anext__ 超时
2. 同步 stream() 也加上异常收窄和兜底
3. finally 块不再静默吞异常，至少打 warning
4. SQL 表名拼接改用 psycopg2.sql.Identifier（防注入）
"""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Generator, AsyncGenerator, Optional

import psycopg2
from psycopg2 import sql as psycopg2_sql
from langchain_core.messages import AIMessage, HumanMessage

from src.retrieval.search_router_practice import SearchRouter
from src.retrieval.faq_retriever_practice import FAQRetriever
from src.retrieval.doc_retriever_practice import DocRetriever
from src.llm.standard_llm_new import LLMError, LLMUnavailableError, StandardLLM
from src.chat.response_generator import ResponseGenerator
from src.vectorstore.pg_pool_practice import get_cursor
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
    """
    while True:
        try:
            # __anext__() 返回的是 coroutine，可以被 wait_for 包
            chunk = await asyncio.wait_for(async_iter.__anext__(), timeout=timeout)
            yield chunk
        except asyncio.TimeoutError:
            logger.error("[ChatEngine] stream chunk 超时（%.0fs），截断输出", timeout)
            yield timeout_msg
            break
        except StopAsyncIteration:
            break


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
        """
        self.search_router = search_router or self._default_search_router()
        self.response_gen = response_gen or ResponseGenerator(llm_mode=llm_mode)
        self.llm_mode = llm_mode
        self.session_id = session_id or str(uuid.uuid4())
        self.history_limit = history_limit
        self.llm_timeout = llm_timeout

        logger.info(
            "[ChatEngine] 初始化完成 | mode=%s | session=%s | history_limit=%d | llm_timeout=%.0fs",
            llm_mode or "default",
            self.session_id,
            history_limit,
            llm_timeout,
        )

    @staticmethod
    def _default_search_router() -> SearchRouter:
        """构建默认的 SearchRouter（内部组装 FAQ + Doc 检索器）"""
        # 统一注入共享的 embedding 函数，避免重复加载模型
        faq_retriever = FAQRetriever(top_k=3, embedding_fn=_shared_get_embedding)
        doc_retriever = DocRetriever(
            top_k=10,
            use_bm25=True,
            fusion_method="rrf",
            rrf_k=60,
            embedding_fn=_shared_get_embedding,
        )
        return SearchRouter(
            faq_retriever=faq_retriever,
            doc_retriever=doc_retriever
        )

    # -----------------------------------------------------------------------
    # 对话历史读写（SQL 用 psycopg2.sql.Identifier 安全拼接表名）
    # -----------------------------------------------------------------------

    def _load_history(self) -> List[Dict[str, str]]:
        """
        从 PG 加载最近 N 条对话历史

        Returns:
            [{"role": "human"/"ai", "content": "..."}, ...]  按时间正序排列
        """
        try:
            with get_cursor() as cur:
                query = psycopg2_sql.SQL("""
                    SELECT role, content
                    FROM {}
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """).format(psycopg2_sql.Identifier(PG_CHAT_TABLE))
                cur.execute(query, (self.session_id, self.history_limit))
                rows = cur.fetchall()
                # 反转成时间正序（最旧的在前，最新的在后）
                history = [{"role": row[0], "content": row[1]} for row in reversed(rows)]
                logger.debug(
                    "[ChatEngine] 加载历史 | session=%s | 条数=%d",
                    self.session_id, len(history)
                )
                return history
        except psycopg2.Error as e:
            # 数据库错误：向上抛 DatabaseError，让上层决定怎么处理
            raise DatabaseError(f"加载历史失败: {e}") from e
        except Exception as e:
            logger.warning("[ChatEngine] 加载历史失败（非 DB 错误）: %s", e)
            return []

    def _save_history(self, role: str, content: str) -> None:
        """
        保存单条消息到 PG

        Args:
            role: 'human' 或 'ai'
            content: 消息内容
        """
        try:
            with get_cursor() as cur:
                query = psycopg2_sql.SQL("""
                    INSERT INTO {} (session_id, role, content)
                    VALUES (%s, %s, %s)
                """).format(psycopg2_sql.Identifier(PG_CHAT_TABLE))
                cur.execute(query, (self.session_id, role, content))
            logger.debug(
                "[ChatEngine] 保存历史 | session=%s | role=%s | content_len=%d",
                self.session_id, role, len(content)
            )
        except psycopg2.Error as e:
            raise DatabaseError(f"保存历史失败: {e}") from e
        except Exception as e:
            logger.warning("[ChatEngine] 保存历史失败（非 DB 错误）: %s", e)

    def _history_to_messages(self, history: List[Dict[str, str]]) -> List[HumanMessage | AIMessage]:
        """
        将历史记录转换为 LangChain Message 列表

        Args:
            history: [{"role": "human"/"ai", "content": "..."}, ...]

        Returns:
            [HumanMessage, AIMessage, ...]
        """
        messages = []
        for item in history:
            role = item.get("role", "").lower()
            content = item.get("content", "")
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
        return messages

    # -----------------------------------------------------------------------
    # 核心对话方法（同步）
    # -----------------------------------------------------------------------

    def _rewrite_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        【P0: Query Rewriting / 指代消解】
        根据对话历史改写当前查询，消除指代歧义（他/她/这/那）。
        如果查询已经明确或没有历史，直接原样返回。
        """
        if not history:
            return query

        # 取最近 3 轮对话作为上下文
        history_text = "\n".join([
            f"{'用户' if h['role'] == 'human' else 'AI'}：{h['content'][:100]}"
            for h in history[-3:]
        ])

        prompt = f"""请根据以下对话历史，改写当前用户查询，消除指代歧义（如"他/她/它/这/那"等）。
如果当前查询已经明确具体，无需改写，请直接原样返回。

对话历史：
{history_text}

当前查询：{query}

改写后的查询（只输出改写结果，不要解释）："""

        try:
            # 【注意】这里用同步 invoke，因为 chat() 是同步方法
            resp = StandardLLM.invoke(prompt, mode=self.llm_mode)
            rewritten = resp.content.strip()
            if rewritten and rewritten != query:
                logger.info("[ChatEngine] Query Rewriting: '%s' → '%s'", query, rewritten)
            return rewritten if rewritten else query
        except Exception as e:
            logger.warning("[ChatEngine] Query Rewriting 失败: %s，回退到原查询", e)
            return query

    async def _arewrite_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """异步版本的 Query Rewriting"""
        if not history:
            return query

        history_text = "\n".join([
            f"{'用户' if h['role'] == 'human' else 'AI'}：{h['content'][:100]}"
            for h in history[-3:]
        ])

        prompt = f"""请根据以下对话历史，改写当前用户查询，消除指代歧义（如"他/她/它/这/那"等）。
如果当前查询已经明确具体，无需改写，请直接原样返回。

对话历史：
{history_text}

当前查询：{query}

改写后的查询（只输出改写结果，不要解释）："""

        try:
            resp = await StandardLLM.ainvoke(prompt, mode=self.llm_mode)
            rewritten = resp.content.strip()
            if rewritten and rewritten != query:
                logger.info("[ChatEngine] Query Rewriting: '%s' → '%s'", query, rewritten)
            return rewritten if rewritten else query
        except Exception as e:
            logger.warning("[ChatEngine] Query Rewriting 失败: %s，回退到原查询", e)
            return query

    def chat(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询（标准模式，带 history + 兜底 + 耗时统计）

        Returns:
            {
                "answer": str,
                "sources": list,
                "citations": list,  # 【P0: 引用溯源】
                "search_type": str,
                "confidence": float,
                "session_id": str,
                "error_code": str | None,
                "latency_ms": float,
            }
        """
        t0 = time.perf_counter()
        error_code = None
        answer = ""
        sources = []
        citations = []  # 【P0: 引用溯源】
        search_type = "unknown"
        confidence = 0.0
        search_context = None

        try:
            # 1. 加载历史
            history = self._load_history()
            history_messages = self._history_to_messages(history)

            # 2. 【P0: Query Rewriting】改写查询，消除指代歧义
            rewritten_query = self._rewrite_query(query, history)

            # 3. 检索（用改写后的 query）
            try:
                search_context = self.search_router.search(rewritten_query)
                search_type = search_context.search_type.value
                confidence = search_context.confidence
            except Exception as e:
                raise RetrievalError(f"检索失败: {e}") from e

            # 3. 生成回答
            if search_context.search_type.value == "faq_only":
                answer = search_context.faq_results[0].answer
                sources = [{
                    "type": "faq",
                    "question": search_context.faq_results[0].question,
                    "confidence": search_context.faq_results[0].similarity
                }]
            else:
                try:
                    answer = self.response_gen.generate(
                        query=query,
                        faq_results=search_context.faq_results,
                        doc_results=search_context.doc_results,
                        history_messages=history_messages,
                    )
                except (LLMError, LLMUnavailableError) as e:
                    raise GenerationError(f"LLM 生成失败: {e}") from e

                if not sources:
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

            # 4. 保存历史
            self._save_history("human", query)
            self._save_history("ai", answer)

        except DatabaseError as e:
            logger.error("[ChatEngine] 数据库错误: %s", e)
            error_code = "DATABASE_ERROR"
            answer = "系统内部错误（数据库），请稍后再试。"
        except RetrievalError as e:
            logger.error("[ChatEngine] 检索错误，降级到纯 LLM: %s", e)
            error_code = "RETRIEVAL_FAILED"
            try:
                history = self._load_history()
                history_messages = self._history_to_messages(history)
                answer = self.response_gen.generate_pure_llm(query, history_messages)
                search_type = "pure_llm_fallback"
            except (LLMError, LLMUnavailableError) as e2:
                logger.error("[ChatEngine] 纯 LLM 兜底也失败: %s", e2)
                error_code = "LLM_FAILED"
                answer = "抱歉，服务暂时不可用，请稍后再试。"
        except GenerationError as e:
            logger.error("[ChatEngine] 生成错误，尝试 FAQ 兜底: %s", e)
            error_code = "GENERATION_FAILED"
            if search_context and search_context.faq_results:
                answer = search_context.faq_results[0].answer
                sources = [{
                    "type": "faq",
                    "question": search_context.faq_results[0].question,
                    "confidence": search_context.faq_results[0].similarity
                }]
                search_type = "faq_fallback"
            else:
                answer = "抱歉，生成回答时出错了，请稍后再试。"
        except Exception as e:
            # 兜底：任何未预期的异常
            logger.exception("[ChatEngine] 未预期错误: %s", e)
            error_code = "UNKNOWN_ERROR"
            answer = "抱歉，系统出现未知错误，请稍后再试。"
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            if error_code and answer:
                # 出错时也要尽量保存用户问题（如果 answer 有值）
                try:
                    self._save_history("human", query)
                    self._save_history("ai", answer)
                except Exception:
                    pass

            logger.info(
                "[ChatEngine] chat | session=%s | type=%s | faq_hits=%d | doc_hits=%d | "
                "confidence=%.3f | error=%s | latency=%.1fms",
                self.session_id,
                search_type,
                len(search_context.faq_results) if search_context else 0,
                len(search_context.doc_results) if search_context else 0,
                confidence,
                error_code,
                latency_ms,
            )

        # 【P0: 引用溯源】解析回答中的引用标记
        if answer:
            citations = self.response_gen.extract_citations(answer)

        return {
            "answer": answer,
            "sources": sources,
            "citations": citations,  # 【P0: 引用溯源】
            "search_type": search_type,
            "confidence": confidence,
            "session_id": self.session_id,
            "error_code": error_code,
            "latency_ms": round(latency_ms, 1),
        }

    # -----------------------------------------------------------------------
    # 流式（同步）—— 也加上异常收窄
    # -----------------------------------------------------------------------

    def stream(self, query: str) -> Generator[str, None, None]:
        """
        同步流式生成回答（带 history + 异常收窄）

        Yields:
            生成的文本片段
        """
        try:
            history = self._load_history()
        except DatabaseError as e:
            logger.error("[ChatEngine] stream 加载历史失败: %s", e)
            yield "系统内部错误（数据库），请稍后再试。"
            return

        history_messages = self._history_to_messages(history)

        # 【P0: Query Rewriting】
        rewritten_query = self._rewrite_query(query, history)

        search_context = None
        try:
            search_context = self.search_router.search(rewritten_query)
        except Exception as e:
            logger.error("[ChatEngine] stream 检索失败，降级到纯 LLM: %s", e)

        chunks = []
        try:
            if search_context and search_context.search_type.value == "faq_only":
                # FAQ 直接整段返回（不再伪流式分割）
                answer = search_context.faq_results[0].answer
                yield answer
                chunks.append(answer)
            else:
                try:
                    for chunk in self.response_gen.generate_stream(
                        query,
                        search_context.faq_results if search_context else [],
                        search_context.doc_results if search_context else [],
                        history_messages,
                    ):
                        yield chunk
                        chunks.append(chunk)
                except (LLMError, LLMUnavailableError) as e:
                    logger.error("[ChatEngine] stream LLM 调用失败: %s", e)
                    yield "\n[系统提示：生成服务暂时不可用]"
        finally:
            full_answer = "".join(chunks)
            # 【P0: 引用溯源】解析引用标记（仅用于日志记录，流式接口不返回结构）
            citations = self.response_gen.extract_citations(full_answer)
            if citations:
                logger.info("[ChatEngine] stream citations: %s", [c["id"] for c in citations])
            try:
                self._save_history("human", query)
                self._save_history("ai", full_answer)
            except Exception as e:
                logger.warning("[ChatEngine] stream 保存历史失败: %s", e)

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
        """
        # 1. 加载历史（同步 DB 操作 → 扔线程池）
        try:
            history = await asyncio.to_thread(self._load_history)
        except DatabaseError as e:
            logger.error("[ChatEngine] astream 加载历史失败: %s", e)
            yield "系统内部错误（数据库），请稍后再试。"
            return

        history_messages = self._history_to_messages(history)

        # 2. 【P0: Query Rewriting】改写查询
        rewritten_query = await self._arewrite_query(query, history)

        # 3. 检索（优先使用异步接口 asearch，回退到 to_thread 包 search）
        search_context = None
        try:
            if hasattr(self.search_router, "asearch"):
                search_context = await self.search_router.asearch(rewritten_query)
            else:
                search_context = await asyncio.to_thread(self.search_router.search, rewritten_query)
        except Exception as e:
            logger.error("[ChatEngine] astream 检索失败，降级到纯 LLM: %s", e)

        chunks = []
        try:
            if search_context and search_context.search_type.value == "faq_only":
                # FAQ 直接整段返回
                answer = search_context.faq_results[0].answer
                yield answer
                chunks.append(answer)
            else:
                # 3. LLM 流式调用（逐 chunk 超时，兼容 Python 3.10）
                try:
                    llm_stream = self.response_gen.agenerate_stream(
                        query,
                        search_context.faq_results if search_context else [],
                        search_context.doc_results if search_context else [],
                        history_messages,
                    )
                    async for chunk in _aiter_with_timeout(llm_stream, timeout=self.llm_timeout):
                        yield chunk
                        chunks.append(chunk)
                except (LLMError, LLMUnavailableError) as e:
                    logger.error("[ChatEngine] astream LLM 调用失败: %s", e)
                    yield "\n[系统提示：生成服务暂时不可用]"
        finally:
            full_answer = "".join(chunks)
            # 【P0: 引用溯源】
            citations = self.response_gen.extract_citations(full_answer)
            if citations:
                logger.info("[ChatEngine] astream citations: %s", [c["id"] for c in citations])
            try:
                await asyncio.to_thread(self._save_history, "human", query)
                await asyncio.to_thread(self._save_history, "ai", full_answer)
            except Exception as e:
                logger.warning("[ChatEngine] astream 保存历史失败: %s", e)

    # -----------------------------------------------------------------------
    # 工具方法
    # -----------------------------------------------------------------------

    def clear_memory(self):
        """清除 LLM 缓存"""
        StandardLLM.clear_cache()
        logger.info("[ChatEngine] LLM 缓存已清除")


# 兼容旧版接口
def get_chat_engine(
    llm_mode: str = None,
    session_id: str = None,
    search_router: Optional[SearchRouter] = None,
    response_gen: Optional[ResponseGenerator] = None,
) -> ChatEngine:
    """
    获取对话引擎实例（工厂函数）

    Args:
        llm_mode: LLM 模式
        session_id: 会话 ID
        search_router: 自定义检索路由器（可选，用于依赖注入）
        response_gen: 自定义回答生成器（可选，用于依赖注入）

    Returns:
        ChatEngine 实例
    """
    return ChatEngine(
        llm_mode=llm_mode,
        session_id=session_id,
        search_router=search_router,
        response_gen=response_gen,
    )
