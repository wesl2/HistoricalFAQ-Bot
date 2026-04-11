# -*- coding: utf-8 -*-
"""
RAG 模块

标准 LangChain 接口（PGVector 版）
使用 PostgreSQL + pgvector 作为向量存储
"""

# 标准模块（推荐使用）
from .standard_rag import create_standard_rag, StandardRAGSystem
from .standard_chain import build_standard_rag_chain, build_conversational_rag_chain
from .standard_retriever import get_pgvector_retriever, PGVectorRetriever
from .standard_memory import get_standard_memory, StandardMemory
from .standard_streaming import stream_rag_response, astream_rag_response

__all__ = [
    # 统一入口
    'create_standard_rag',
    'StandardRAGSystem',
    # Chain
    'build_standard_rag_chain',
    'build_conversational_rag_chain',
    # Retriever（PGVector）
    'get_pgvector_retriever',
    'PGVectorRetriever',
    # Memory
    'get_standard_memory',
    'StandardMemory',
    # Streaming
    'stream_rag_response',
    'astream_rag_response',
]
