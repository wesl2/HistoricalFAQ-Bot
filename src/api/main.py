# -*- coding: utf-8 -*-
"""
FastAPI 后端服务（v2.1：按请求隔离 session，全局复用核心组件）

改动点：
1. 去掉全局 ChatEngine 单例，避免 session_id 串台
2. SearchRouter / ResponseGenerator 全局复用，避免 BM25 重复加载
3. 前端传入 session_id，不传则后端自动生成
4. 标准查询走 achat()，享受并行检索收益
"""

import logging
import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.chat.chat_engine import ChatEngine
from src.chat.response_generator import ResponseGenerator
from src.retrieval.search_router_practice import SearchRouter
from src.retrieval.faq_retriever_practice import FAQRetriever
from src.retrieval.doc_retriever_practice import DocRetriever
from src.embedding.embedding_local_practice import get_embedding as _shared_get_embedding

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="Historical FAQ Bot API",
    description="基于 RAG 架构的历史人物问答系统",
    version="2.1.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 全局复用核心组件 ====================
# SearchRouter 和 ResponseGenerator 无状态、线程安全，只初始化一次

logger.info("初始化全局核心组件...")

_faq_retriever = FAQRetriever(top_k=3, embedding_fn=_shared_get_embedding)
_doc_retriever = DocRetriever(
    top_k=10,
    use_bm25=True,
    fusion_method="rrf",
    rrf_k=60,
    embedding_fn=_shared_get_embedding,
)
_search_router = SearchRouter(
    faq_retriever=_faq_retriever,
    doc_retriever=_doc_retriever
)
_response_gen = ResponseGenerator()

logger.info("全局核心组件初始化完成")


def _get_chat_engine(session_id: Optional[str] = None) -> ChatEngine:
    """
    工厂函数：每次请求创建新的 ChatEngine（隔离 session），
    但注入全局复用的核心组件。
    """
    return ChatEngine(
        search_router=_search_router,
        response_gen=_response_gen,
        session_id=session_id,
    )


# ==================== 请求/响应模型 ====================

class QueryRequest(BaseModel):
    """标准查询请求"""
    question: str
    session_id: Optional[str] = None  # 前端传入，继续历史对话；不传则开启新对话


class StreamQueryRequest(BaseModel):
    """流式查询请求"""
    question: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """标准查询响应"""
    answer: str
    sources: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    search_type: str
    confidence: float
    session_id: str
    error_code: Optional[str] = None
    latency_ms: float


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    service: str
    version: str
    features: List[str]


# ==================== API 端点 ====================

@app.post("/api/query", response_model=QueryResponse)
async def query_faq(request: QueryRequest):
    """
    处理用户查询（异步标准模式，带 session 隔离）

    Args:
        request: 包含问题和可选 session_id 的请求

    Returns:
        包含回答、来源、检索类型、session_id 的响应
    """
    try:
        logger.info("收到查询 | session=%s | question=%s...",
                    request.session_id or "(new)", request.question[:50])

        engine = _get_chat_engine(session_id=request.session_id)
        result = await engine.achat(query=request.question)

        logger.info("查询完成 | session=%s | type=%s | confidence=%.2f | latency=%.1fms",
                    result["session_id"], result.get("search_type", "unknown"),
                    result["confidence"], result["latency_ms"])
        return QueryResponse(**result)

    except Exception as e:
        logger.error("查询错误: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(request: StreamQueryRequest):
    """
    流式查询（SSE 输出）

    Args:
        request: 包含问题和可选 session_id 的请求

    Returns:
        Server-Sent Events 流
    """
    try:
        logger.info("收到流式查询 | session=%s | question=%s...",
                    request.session_id or "(new)", request.question[:50])

        engine = _get_chat_engine(session_id=request.session_id)

        async def generate():
            """生成 SSE 流"""
            try:
                async for chunk in engine.astream(request.question):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error("流式生成错误: %s", e)
                yield f"data: [ERROR] {str(e)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        logger.error("流式查询错误: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    features = [
        "FAQ 检索",
        "文档检索 (RAG)",
        "BM25 + 向量混合检索",
        "RRF 融合排序",
        "Query Rewriting (指代消解)",
        "本地/API LLM 支持",
        "流式输出",
        "引用溯源",
    ]

    return HealthResponse(
        status="healthy",
        service="Historical FAQ Bot API",
        version="2.1.0",
        features=features
    )


@app.get("/api/info")
async def get_info():
    """获取服务信息"""
    return {
        "name": "Historical FAQ Bot",
        "version": "2.1.0",
        "description": "基于 RAG 架构的历史人物问答系统",
        "config": {
            "llm_mode": os.getenv("LLM_MODE", "local"),
        }
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    logger.info("启动服务: %s:%d, reload=%s", host, port, reload)

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )
