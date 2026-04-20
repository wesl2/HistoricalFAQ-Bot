# -*- coding: utf-8 -*-
"""
对话引擎

整合检索和生成，提供完整的对话能力
支持：
- 原生双模检索（FAQ + RAG）
- LangChain 集成
- 高级检索（Multi-Query + Rerank）
- 流式输出
- 对话记忆
- Agent 功能
"""

import logging
import uuid
from typing import List, Dict, Any, Generator, AsyncGenerator

from langchain_core.documents import Document
from langchain_community.chat_message_histories import PostgresChatMessageHistory

from src.retrieval.search_router import SearchRouter
from src.retrieval.faq_retriever_practice import FAQRetriever
from src.retrieval.doc_retriever_practice import DocRetriever
from config.pg_config import PG_URL, PG_CHAT_TABLE
from src.llm.llm_factory import LLMFactory
from src.chat.response_generator import ResponseGenerator
from src.rag.langchain_integration import LangChainIntegration
from src.rag.advanced_retriever import get_advanced_retriever
from src.rag.callbacks import get_callback_manager
from config.model_config import LANGCHAIN_CONFIG

logger = logging.getLogger(__name__)


class ChatEngine:
    """
    对话引擎
    
    核心流程：
    1. 接收用户查询
    2. 检索相关内容（FAQ/文档）
    3. 调用 LLM 生成回答
    4. 返回格式化结果
    """
    
    def __init__(self, 
                 llm_mode: str = None, 
                 use_langchain: bool = True,
                 use_advanced_retriever: bool = True,
                 chain_type: str = None,
                 session_id: str = None):
        """
        初始化对话引擎
        
        Args:
            llm_mode: LLM 模式，None 则使用配置默认值
            use_langchain: 是否使用 LangChain 集成
            use_advanced_retriever: 是否使用高级检索器
            chain_type: 链类型 (rag, conversational, conversational_retrieval)
            session_id: 会话ID，用于持久化对话历史，None则自动生成
        """
        # 依赖注入：在外部创建检索器实例，传入 SearchRouter
        # 这样 ChatEngine 控制用什么检索器，SearchRouter 只负责路由逻辑
        faq_retriever = FAQRetriever(top_k=3)
        doc_retriever = DocRetriever(
            top_k=10,
            use_bm25=True,
            fusion_method="rrf",
            rrf_k=60
        )
        self.search_router = SearchRouter(
            faq_retriever=faq_retriever,
            doc_retriever=doc_retriever
        )
        self.llm = LLMFactory.create_llm(llm_mode)
        self.response_gen = ResponseGenerator(self.llm)
        
        self.use_langchain = use_langchain
        self.use_advanced_retriever = use_advanced_retriever
        self.chain_type = chain_type or LANGCHAIN_CONFIG["default_chain_type"]
        
        # 会话ID（用于持久化对话历史）
        self.session_id = session_id or str(uuid.uuid4())
        
        # LangChain 集成
        self.langchain = None
        if use_langchain:
            self.langchain = LangChainIntegration(
                llm_mode=llm_mode,
                use_advanced_retriever=use_advanced_retriever,
                session_id=self.session_id  # 传入 session_id 用于持久化
            )
        
        # 高级检索器（原生方式）
        self.advanced_retriever = None
        if use_advanced_retriever and not use_langchain:
            self.advanced_retriever = get_advanced_retriever()
        
        # 缓存检索器
        self._retrievers = {}
        
        logger.info(
            f"对话引擎初始化完成: "
            f"LLM={llm_mode or 'default'}, "
            f"LangChain={use_langchain}, "
            f"AdvancedRetriever={use_advanced_retriever}, "
            f"Chain={self.chain_type}, "
            f"Session={self.session_id}"
        )
    
    def _build_documents(self, faq_results, doc_results) -> List[Document]:
        """
        构建 LangChain 文档
        
        Args:
            faq_results: FAQ 搜索结果
            doc_results: 文档搜索结果
            
        Returns:
            文档列表
        """
        docs_content = []
        for r in faq_results[:3]:
            content = f"问题: {r.question}\n答案: {r.answer}"
            docs_content.append(Document(
                page_content=content,
                metadata={"type": "faq", "source": r.question}
            ))
        
        for r in doc_results[:3]:
            content = f"来源: {r.doc_name}\n内容: {r.content}"
            docs_content.append(Document(
                page_content=content,
                metadata={"type": "doc", "source": r.doc_name, "page": r.doc_page}
            ))
        
        return docs_content
    
    def _get_or_create_retriever(self, documents: List[Document]):
        """
        获取或创建检索器（带缓存）
        
        Args:
            documents: 文档列表
            
        Returns:
            检索器实例
        """
        cache_key = f"{len(documents)}_{hash(str([d.page_content[:50] for d in documents]))}"
        
        if cache_key not in self._retrievers:
            if self.langchain:
                vectorstore = self.langchain.create_vectorstore(documents)
                self._retrievers[cache_key] = self.langchain.create_retriever(vectorstore)
        
        return self._retrievers.get(cache_key)
    
    def chat(self, query: str, history: List[Dict] = None) -> Dict[str, Any]:
        """
        处理用户查询（标准模式）
        
        Args:
            query: 用户问题
            history: 对话历史（可选）
            
        Returns:
            {
                "answer": str,
                "sources": list,
                "search_type": str,
                "confidence": float,
                "langchain_used": bool,
                "chain_type": str
            }
        """
        # 1. 检索相关内容（原生检索策略）
        search_context = self.search_router.search(query)
        
        # 2. 根据检索类型生成回答
        if search_context.search_type.value == "faq_only":
            # 高置信度 FAQ，直接返回答案
            answer = search_context.faq_results[0].answer
            sources = [{
                "type": "faq",
                "question": search_context.faq_results[0].question,
                "confidence": search_context.faq_results[0].similarity
            }]
            
        else:
            # 需要 LLM 生成
            if self.use_langchain and self.langchain:
                answer = self._langchain_generate(
                    query=query,
                    faq_results=search_context.faq_results,
                    doc_results=search_context.doc_results,
                    history=history
                )
            else:
                # 原生方式
                answer = self.response_gen.generate(
                    query=query,
                    faq_results=search_context.faq_results,
                    doc_results=search_context.doc_results
                )
            
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
            
            # 使用高级检索器的记忆功能
            if history and self.use_langchain and self.langchain:
                self._save_history_to_memory(history)
        
        return {
            "answer": answer,
            "sources": sources,
            "search_type": search_context.search_type.value,
            "confidence": search_context.confidence,
            "session_id": self.session_id,  # 返回 session_id
            "langchain_used": self.use_langchain,
            "chain_type": self.chain_type if self.use_langchain else None
        }
    
    def _langchain_generate(self, 
                           query: str,
                           faq_results,
                           doc_results,
                           history: List[Dict] = None) -> str:
        """
        使用 LangChain 生成回答
        
        Args:
            query: 查询
            faq_results: FAQ 结果
            doc_results: 文档结果
            history: 历史记录
            
        Returns:
            生成的回答
        """
        # 构建文档
        documents = self._build_documents(faq_results, doc_results)
        
        # 获取检索器
        retriever = self._get_or_create_retriever(documents)
        
        # 根据链类型选择生成方式
        if self.chain_type == "conversational":
            chain = self.langchain.create_conversational_chain(retriever)
            answer = chain.invoke(query)
            
        elif self.chain_type == "conversational_retrieval":
            chain = self.langchain.create_conversational_retrieval_chain(retriever)
            result = chain({"question": query})
            answer = result["answer"]
            
        else:  # rag
            chain = self.langchain.create_rag_chain(retriever)
            answer = chain.invoke(query)
        
        # 保存上下文
        self.langchain.save_context(query, answer)
        
        return answer
    
    def _save_history_to_memory(self, history: List[Dict]):
        """
        保存历史记录到记忆
        
        Args:
            history: 对话历史
        """
        if not self.langchain:
            return
        
        for item in history:
            if "role" in item and "content" in item:
                if item["role"] == "user":
                    self.langchain.save_context(item["content"], "")
                elif item["role"] == "assistant":
                    self.langchain.save_context("", item["content"])
    
    def stream(self, query: str) -> Generator[str, None, None]:
        """
        流式生成回答
        
        Args:
            query: 查询文本
            
        Yields:
            生成的文本片段
        """
        if not self.use_langchain or not self.langchain:
            raise ValueError("流式输出需要 LangChain 支持")
        
        # 使用原生检索策略
        search_context = self.search_router.search(query)
        
        if search_context.search_type.value == "faq_only":
            # FAQ 直接返回（非流式，但模拟流式）
            answer = search_context.faq_results[0].answer
            # 按句子分割，模拟流式
            import re
            sentences = re.split(r'([。！？；\n])', answer)
            buffer = ""
            for s in sentences:
                buffer += s
                if s in ['。', '！', '？', '；', '\n']:
                    yield buffer
                    buffer = ""
            if buffer:
                yield buffer
        else:
            # 使用 LangChain 流式生成
            documents = self._build_documents(
                search_context.faq_results,
                search_context.doc_results
            )
            retriever = self._get_or_create_retriever(documents)
            
            for chunk in self.langchain.stream(query):
                yield chunk
    
    async def astream(self, query: str) -> AsyncGenerator[str, None]:
        """
        异步流式生成回答
        
        Args:
            query: 查询文本
            
        Yields:
            生成的文本片段
        """
        if not self.use_langchain or not self.langchain:
            raise ValueError("流式输出需要 LangChain 支持")
        
        async for chunk in self.langchain.astream(query):
            yield chunk
    
    def clear_memory(self):
        """清除对话记忆"""
        if self.langchain:
            self.langchain.clear_memory()
            logger.info("LangChain 对话记忆已清除")
        
        # 清除缓存的检索器
        self._retrievers.clear()
        logger.info("检索器缓存已清除")
    
    def add_tool(self, name: str, func, description: str):
        """
        添加工具
        
        Args:
            name: 工具名称
            func: 工具函数
            description: 工具描述
            
        Returns:
            工具实例
        """
        if self.langchain:
            return self.langchain.create_tool(name, func, description)
        return None
    
    def create_agent(self, tools: list):
        """
        创建智能体
        
        Args:
            tools: 工具列表
            
        Returns:
            智能体实例
        """
        if self.langchain:
            return self.langchain.create_agent(tools)
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取监控指标
        
        Returns:
            监控指标字典
        """
        if self.langchain:
            return self.langchain.get_metrics()
        return {
            "performance": {"note": "LangChain 未启用"},
            "token_usage": {"note": "LangChain 未启用"}
        }
    
    def reset_metrics(self):
        """重置监控指标"""
        if self.langchain:
            self.langchain.reset_metrics()
            logger.info("监控指标已重置")


# 兼容旧版接口
def get_chat_engine(llm_mode: str = None, use_langchain: bool = True) -> ChatEngine:
    """
    获取对话引擎实例（工厂函数）
    
    Args:
        llm_mode: LLM 模式
        use_langchain: 是否使用 LangChain
        
    Returns:
        ChatEngine 实例
    """
    return ChatEngine(llm_mode=llm_mode, use_langchain=use_langchain)
