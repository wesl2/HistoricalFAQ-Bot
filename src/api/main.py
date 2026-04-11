# -*- coding: utf-8 -*-
"""
FastAPI后端服务

提供历史问答API接口，支持：
- 标准 RAG 查询
- 流式输出
- Agent 功能
- 监控指标
"""

import logging
import os
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.chat.chat_engine import ChatEngine
from src.tools.tools import Tools

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Historical FAQ Bot API",
    description="基于RAG架构的历史人物问答系统（支持LangChain集成、流式输出、Agent）",
    version="2.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应设置具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 初始化核心组件 ====================

# 从环境变量读取配置
use_langchain = os.getenv("USE_LANGCHAIN", "true").lower() == "true"
use_advanced_retriever = os.getenv("USE_ADVANCED_RETRIEVER", "true").lower() == "true"

logger.info(f"初始化服务: use_langchain={use_langchain}, use_advanced_retriever={use_advanced_retriever}")

# 初始化对话引擎
chat_engine = ChatEngine(
    use_langchain=use_langchain,
    use_advanced_retriever=use_advanced_retriever
)

# 初始化工具和智能体
agent = None
if use_langchain:
    try:
        tools_instance = Tools(chat_engine)
        tools = [
            chat_engine.add_tool(
                name=tool["name"], 
                func=tool["func"], 
                description=tool["description"]
            )
            for tool in tools_instance.get_all_tools()
        ]
        agent = chat_engine.create_agent(tools)
        logger.info("Agent 初始化完成")
    except Exception as e:
        logger.error(f"Agent 初始化失败: {e}")
        agent = None

# ==================== 请求/响应模型 ====================

class QueryRequest(BaseModel):
    """标准查询请求"""
    question: str
    history: List[Dict[str, str]] = None
    stream: bool = False  # 是否使用流式输出


class StreamQueryRequest(BaseModel):
    """流式查询请求"""
    question: str


class AgentRequest(BaseModel):
    """Agent 请求"""
    question: str


class QueryResponse(BaseModel):
    """标准查询响应"""
    answer: str
    sources: List[Dict[str, Any]]
    search_type: str
    confidence: float
    langchain_used: bool = False
    chain_type: str = None


class AgentResponse(BaseModel):
    """Agent 响应"""
    answer: str


class MetricsResponse(BaseModel):
    """监控指标响应"""
    performance: Dict[str, Any]
    token_usage: Dict[str, Any]


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
    处理用户查询（标准模式）
    
    Args:
        request: 包含问题和对话历史的请求
        
    Returns:
        包含回答、来源、检索类型和置信度的响应
    """
    try:
        logger.info(f"收到查询: {request.question[:50]}...")
        
        # 如果请求流式输出，但客户端不支持，降级为标准输出
        if request.stream and not use_langchain:
            logger.warning("流式输出需要 LangChain 支持，降级为标准输出")
        
        result = chat_engine.chat(
            query=request.question,
            history=request.history
        )
        
        logger.info(f"查询完成，置信度: {result['confidence']:.2f}, 检索类型: {result.get('search_type', 'unknown')}")
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"查询错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(request: StreamQueryRequest):
    """
    流式查询（SSE 输出）
    
    Args:
        request: 包含问题的请求
        
    Returns:
        Server-Sent Events 流
    """
    if not use_langchain:
        raise HTTPException(
            status_code=400, 
            detail="流式输出需要 LangChain 支持，请设置 USE_LANGCHAIN=true"
        )
    
    try:
        logger.info(f"收到流式查询: {request.question[:50]}...")
        
        async def generate():
            """生成 SSE 流"""
            try:
                async for chunk in chat_engine.astream(request.question):
                    # SSE 格式
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"流式生成错误: {e}")
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
        logger.error(f"流式查询错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent", response_model=AgentResponse)
async def query_agent(request: AgentRequest):
    """
    处理需要工具调用的复杂查询
    
    Args:
        request: 包含问题的请求
        
    Returns:
        包含回答的响应
    """
    if not use_langchain:
        raise HTTPException(
            status_code=400, 
            detail="Agent 功能需要 LangChain 支持"
        )
    
    if agent is None:
        raise HTTPException(
            status_code=503, 
            detail="Agent 未初始化，请检查配置"
        )
    
    try:
        logger.info(f"收到 Agent 查询: {request.question[:50]}...")
        result = agent.run(request.question)
        logger.info("Agent 查询完成")
        return AgentResponse(answer=result)
        
    except Exception as e:
        logger.error(f"Agent 查询错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    获取系统监控指标
    
    Returns:
        性能指标和 Token 使用统计
    """
    try:
        if not use_langchain:
            return MetricsResponse(
                performance={"note": "LangChain 未启用"},
                token_usage={"note": "LangChain 未启用"}
            )
        
        metrics = chat_engine.get_metrics()
        return MetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"获取指标错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/metrics/reset")
async def reset_metrics():
    """
    重置监控指标
    
    Returns:
        重置结果
    """
    try:
        if use_langchain:
            chat_engine.reset_metrics()
        return {"status": "success", "message": "指标已重置"}
        
    except Exception as e:
        logger.error(f"重置指标错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memory/clear")
async def clear_memory():
    """
    清除对话记忆
    
    Returns:
        清除结果
    """
    try:
        chat_engine.clear_memory()
        logger.info("对话记忆已清除")
        return {"status": "success", "message": "对话记忆已清除"}
        
    except Exception as e:
        logger.error(f"清除记忆错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口
    """
    features = [
        "FAQ检索",
        "文档检索(RAG)",
        "混合检索",
        "本地/API LLM支持"
    ]
    
    if use_langchain:
        features.extend([
            "LangChain 集成",
            "对话记忆",
            "Agent 功能",
            "工具调用",
            "流式输出",
            "监控指标"
        ])
    
    if use_advanced_retriever:
        features.extend([
            "Multi-Query 检索",
            "BCE-Reranker 重排序",
            "向量库持久化"
        ])
    
    return HealthResponse(
        status="healthy",
        service="Historical FAQ Bot API",
        version="2.0.0",
        features=features
    )


@app.get("/api/info")
async def get_info():
    """
    获取服务信息
    """
    return {
        "name": "Historical FAQ Bot",
        "version": "2.0.0",
        "description": "基于RAG架构的历史人物问答系统（支持LangChain集成）",
        "config": {
            "use_langchain": use_langchain,
            "use_advanced_retriever": use_advanced_retriever,
            "llm_mode": os.getenv("LLM_MODE", "local"),
            "chain_type": os.getenv("LANGCHAIN_CHAIN_TYPE", "rag")
        }
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量读取启动配置
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"启动服务: {host}:{port}, reload={reload}")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )
