# -*- coding: utf-8 -*-
"""
标准 LangChain 接口（统一入口）

公司级实践：清晰的 API，标准接口
使用 PostgreSQL + pgvector 作为向量存储
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from typing import Iterator, AsyncIterator, List, Dict, Any

# 导入标准模块（PGVector 版）
from src.llm.standard_llm import get_standard_llm, StandardLLM
from src.rag.standard_retriever import get_pgvector_retriever, PGVectorRetriever
from src.rag.standard_chain import (
    build_standard_rag_chain,
    build_conversational_rag_chain,
    build_agent_chain,
    format_docs,
)
from src.rag.standard_memory import get_standard_memory, StandardMemory, clear_memory
from src.rag.standard_streaming import stream_rag_response, astream_rag_response


class StandardRAGSystem:
    """
    标准 RAG 系统（统一入口）

    公司实践：
    - 使用 LangChain 标准接口
    - LCEL 组合式写法
    - 清晰的流式输出
    - 外部 Memory 管理
    """

    def __init__(
        self,
        llm_mode: str = None,
        session_id: str = None,
        use_postgres_memory: bool = True,
        retriever_search_type: str = "hybrid",  # "vector", "faq", "hybrid"
    ):
        """
        初始化标准 RAG 系统

        Args:
            llm_mode: LLM 模式 ("local" 或 "api")
            session_id: 会话 ID
            use_postgres_memory: 是否使用 PostgreSQL 持久化
            retriever_search_type: 检索类型 ("vector", "faq", "hybrid")
        """
        self.llm = get_standard_llm(llm_mode)
        self.retriever = get_pgvector_retriever(search_type=retriever_search_type)
        self.session_id = session_id
        self.memory = None

        if session_id:
            self.memory = get_standard_memory(session_id, use_postgres_memory)

    def query(self, question: str) -> str:
        """
        标准 RAG 查询

        Args:
            question: 用户问题

        Returns:
            str: AI 回答
        """
        chain = build_standard_rag_chain(
            llm=self.llm,
            retriever=self.retriever,
        )

        result = chain.invoke(question)

        # 保存历史
        if self.memory:
            self.memory.add_user_message(question)
            self.memory.add_ai_message(result)

        return result

    def query_with_history(self, question: str) -> str:
        """
        带对话历史的查询（标准 Conversational RAG）

        使用 MessagesPlaceholder 传递结构化历史消息，
        而不是拼接字符串，确保 LCEL 链正确工作。

        Args:
            question: 用户问题

        Returns:
            str: AI 回答
        """
        if not self.memory:
            return self.query(question)

        chain = build_conversational_rag_chain(
            llm=self.llm,
            retriever=self.retriever,
            use_history_rewrite=True,
        )

        # 修复：传递标准的 dict，包含 history messages list 和 question
        history_messages = self.memory.get_messages()
        result = chain.invoke({
            "history": history_messages,
            "question": question,
        })

        # 保存历史
        self.memory.add_user_message(question)
        self.memory.add_ai_message(result)

        return result

    def stream_query(self, question: str) -> Iterator[str]:
        """
        流式查询

        Args:
            question: 用户问题

        Yields:
            str: 每次生成的文本块
        """
        for chunk in stream_rag_response(
            query=question,
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
        ):
            yield chunk

    def add_documents(self, documents: List[Document]):
        """
        添加文档到向量库

        Args:
            documents: 文档列表
        """
        self.retriever.add_documents(documents)

    def clear_memory(self):
        """清空对话历史"""
        if self.memory:
            self.memory.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """获取系统指标（可扩展）"""
        return {
            "llm_mode": self.llm.__class__.__name__,
            "retriever_type": self.retriever.__class__.__name__,
            "memory_size": len(self.memory) if self.memory else 0,
        }


# 便捷函数
def create_standard_rag(
    llm_mode: str = None,
    session_id: str = None,
    use_postgres_memory: bool = True,
    retriever_search_type: str = "hybrid",
) -> StandardRAGSystem:
    """创建标准 RAG 系统实例（使用关键字参数防止错位）"""
    return StandardRAGSystem(
        llm_mode=llm_mode,
        session_id=session_id,
        use_postgres_memory=use_postgres_memory,
        retriever_search_type=retriever_search_type,
    )
