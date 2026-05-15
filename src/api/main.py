# -*- coding: utf-8 -*-
"""
FastAPI 后端服务（v3.0：生产级 RAG API）

核心改进：
1. lifespan 上下文：优雅管理组件生命周期
2. 错误码映射：根据 ChatEngine 返回的 error_code 返回对应 HTTP 状态码
3. 请求追踪：中间件注入 request_id，日志全链路可追踪
4. 健康检查：实际探测 DB + LLM 连接
5. 限流保护：asyncio.Semaphore 防止并发打爆 LLM 后端
6. SSE 标准化：规范 event / data / id 格式

注意：ChatEngine.achat() 内部已 catch 所有异常，返回 error_code 字段。
API 层不再依赖异常处理器，而是检查 result["error_code"] 映射状态码。
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 动态导入 ingest_documents 脚本（避免直接依赖其副作用）
import importlib.util
_ingest_spec = importlib.util.spec_from_file_location(
    "_ingest_documents",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "ingest_documents.py")
)
_ingest_module = importlib.util.module_from_spec(_ingest_spec)
import sys
sys.modules["_ingest_documents"] = _ingest_module
_ingest_spec.loader.exec_module(_ingest_module)

from src.chat.chat_engine import ChatEngine
from src.chat.response_generator import ResponseGenerator
from src.retrieval.search_router_practice import SearchRouter
from src.retrieval.faq_retriever_practice import FAQRetriever
from src.retrieval.doc_retriever_practice import DocRetriever
from src.llm.standard_llm_new import StandardLLM
from src.vectorstore.pg_pool_practice import check_connection as check_pg_connection, get_connection
from src.embedding.embedding_local_practice import get_embedding as _shared_get_embedding
from config.pg_config import PG_DOC_TABLE

# =============================================================================
# 配置
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 限流：同时最多 N 个 LLM 请求在跑，防止打爆 DeepSeek API
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "10"))
_llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

# error_code → HTTP 状态码 映射（与 ChatEngine 内部错误码对齐）
ERROR_CODE_MAP = {
    "DATABASE_ERROR": (503, "数据库服务暂不可用"),
    "RETRIEVAL_FAILED": (502, "检索服务异常"),
    "LLM_FAILED": (504, "AI 生成服务超时"),
    "GENERATION_FAILED": (500, "生成回答时出错"),
    "UNKNOWN_ERROR": (500, "系统出现未知错误"),
}

# =============================================================================
# Lifespan：优雅管理生命周期
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    启动：初始化全局无状态组件
    关闭：释放资源
    """
    logger.info("[Lifespan] 启动：初始化全局组件...")

    app.state.faq_retriever = FAQRetriever(top_k=3, embedding_fn=_shared_get_embedding)
    app.state.doc_retriever = DocRetriever(
        top_k=10, use_bm25=True, fusion_method="rrf", rrf_k=60,
        embedding_fn=_shared_get_embedding,
    )
    app.state.search_router = SearchRouter(
        faq_retriever=app.state.faq_retriever,
        doc_retriever=app.state.doc_retriever
    )
    app.state.response_gen = ResponseGenerator()

    logger.info("[Lifespan] 全局组件初始化完成 | LLM并发限流=%d", LLM_CONCURRENCY)

    yield

    logger.info("[Lifespan] 关闭：释放资源...")
    logger.info("[Lifespan] 资源释放完成")


# =============================================================================
# FastAPI 应用
# =============================================================================

app = FastAPI(
    title="Historical Chat Bot API",
    description="基于 RAG 架构的历史人物问答系统（PDR语义切分 + BM25混合检索 + 引用校验）",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 中间件：请求追踪 + 耗时统计
# =============================================================================

@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """每个请求注入 request_id，并记录耗时"""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    adapter = logging.LoggerAdapter(logger, {"request_id": request_id})
    t0 = time.perf_counter()
    adapter.info("→ 请求开始 | %s %s", request.method, request.url.path)

    try:
        response = await call_next(request)
        latency = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency:.1f}ms"
        adapter.info("← 请求完成 | %d | %.1fms", response.status_code, latency)
        return response
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        adapter.error("✕ 请求异常 | %.1fms | %s", latency, e, exc_info=True)
        raise


# =============================================================================
# 工厂函数
# =============================================================================

def _get_chat_engine(request: Request, session_id: Optional[str] = None) -> ChatEngine:
    """每次请求创建新的 ChatEngine（隔离 session），注入全局复用组件"""
    return ChatEngine(
        search_router=request.app.state.search_router,
        response_gen=request.app.state.response_gen,
        session_id=session_id,
    )


# =============================================================================
# 请求/响应模型
# =============================================================================

class QueryRequest(BaseModel):
    """标准查询请求"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID，不传则开启新对话")


class StreamQueryRequest(BaseModel):
    """流式查询请求"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID，不传则开启新对话")


class CitationItem(BaseModel):
    """引用项"""
    id: str
    type: str
    doc_name: Optional[str] = None
    content: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None


class QueryResponse(BaseModel):
    """标准查询响应"""
    answer: str
    sources: List[Dict[str, Any]]
    citations: List[CitationItem]
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
    checks: Dict[str, Any]
    features: List[str]


# =============================================================================
# 工具函数
# =============================================================================

def _map_error_code(error_code: Optional[str]) -> tuple:
    """
    将 ChatEngine 返回的 error_code 映射为 HTTP 状态码和友好提示。
    与 ChatEngine 内部错误码完全对齐。
    """
    if not error_code:
        return 200, None
    return ERROR_CODE_MAP.get(error_code, (500, "服务异常"))


# =============================================================================
# API 端点
# =============================================================================

@app.post("/api/query", response_model=QueryResponse, tags=["对话"], summary="标准问答")
async def query_faq(request: Request, body: QueryRequest):
    """
    标准问答接口（非流式）。

    ChatEngine 内部已处理所有异常，返回 error_code 字段。
    API 层检查 error_code 并映射为对应 HTTP 状态码。
    """
    engine = _get_chat_engine(request, session_id=body.session_id)

    async with _llm_semaphore:
        result = await engine.achat(query=body.question)

    error_code = result.get("error_code")
    if error_code:
        status_code, message = _map_error_code(error_code)
        logger.warning("[API] 业务错误 | code=%s | status=%d | session=%s",
                       error_code, status_code, result.get("session_id"))
        return JSONResponse(
            status_code=status_code,
            content={
                "error": message,
                "detail": result.get("answer", message),
                "error_code": error_code,
                "session_id": result.get("session_id"),
            }
        )

    return QueryResponse(**result)


@app.post("/api/query/stream", tags=["对话"], summary="流式问答 (SSE)")
async def query_stream(request: Request, body: StreamQueryRequest):
    """
    流式问答接口（SSE 输出）。

    ChatEngine.astream() 内部已 catch 异常，yield 错误文本。
    API 层将错误文本包装为 SSE error 事件。
    """
    engine = _get_chat_engine(request, session_id=body.session_id)

    async def generate() -> AsyncGenerator[str, None]:
        try:
            async with _llm_semaphore:
                async for chunk in engine.astream(body.question):
                    # ChatEngine 可能在出错时 yield 错误文本（而非抛异常）
                    # 这里统一包装为 message 事件
                    # 特殊标记 __CITATIONS__ 表示引用后处理结果
                    if chunk.startswith("__CITATIONS__"):
                        payload = chunk[len("__CITATIONS__"):]
                        yield f"event: citations\ndata: {payload}\n\n"
                    else:
                        yield f"event: message\ndata: {chunk}\n\n"

            yield "event: done\ndata: [DONE]\n\n"

        except Exception as exc:
            # 极少数情况下 astream 本身抛异常（如 Semaphore 异常）
            logger.error("[StreamError] %s", exc, exc_info=True)
            error_msg = "服务内部错误，请稍后重试"
            yield f"event: error\ndata: {error_msg}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/health", response_model=HealthResponse, tags=["运维"], summary="健康检查")
async def health_check():
    """
    健康检查（实际探测依赖服务）。
    - database: PostgreSQL 连接检查
    - llm_api: DeepSeek API 可达性检查
    """
    checks = {}

    try:
        pg_ok = await asyncio.to_thread(check_pg_connection)
        checks["database"] = {"status": "ok" if pg_ok else "fail", "detail": "PostgreSQL"}
    except Exception as e:
        checks["database"] = {"status": "fail", "detail": str(e)}

    try:
        llm_ok = await StandardLLM.ahealth_check()
        checks["llm_api"] = {"status": "ok" if llm_ok else "fail", "detail": "DeepSeek API"}
    except Exception as e:
        checks["llm_api"] = {"status": "fail", "detail": str(e)}

    overall = "healthy" if all(c["status"] == "ok" for c in checks.values()) else "degraded"

    features = [
        "FAQ 检索", "文档检索 (RAG)", "BM25 + 向量混合检索",
        "RRF 融合排序", "Query Rewriting (指代消解)",
        "本地/API LLM 支持", "流式输出 (SSE)", "引用溯源与校验",
        "PDR 语义切分", "对话历史持久化",
    ]

    return HealthResponse(
        status=overall,
        service="Historical Chat Bot API",
        version="3.0.0",
        checks=checks,
        features=features,
    )


@app.get("/api/info", tags=["运维"], summary="服务信息")
async def get_info():
    """获取服务基本信息"""
    return {
        "name": "Historical Chat Bot",
        "version": "3.0.0",
        "description": "基于 RAG 架构的历史人物问答系统",
        "stack": {
            "llm": os.getenv("LLM_MODE", "local"),
            "embedding": "BGE-M3 (1024d)",
            "retrieval": "BM25 + Vector (RRF fusion)",
            "database": "PostgreSQL + pgvector",
        },
        "limits": {
            "llm_concurrency": LLM_CONCURRENCY,
            "max_question_length": 2000,
        }
    }


# =============================================================================
# 文档列表
# =============================================================================

def _get_doc_names_sync() -> List[str]:
    """同步查询数据库中已入库的文档名称列表"""
    names = []
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT doc_name FROM {PG_DOC_TABLE} WHERE doc_name IS NOT NULL ORDER BY doc_name"
                )
                rows = cur.fetchall()
                names = [row[0] for row in rows if row[0]]
    except Exception as e:
        logger.error("[Documents] 查询文档列表失败: %s", e)
    return names


@app.get("/api/documents", tags=["文档"], summary="已入库文档列表")
async def get_documents():
    """获取当前已解析入库的所有文档名称列表"""
    names = await asyncio.to_thread(_get_doc_names_sync)
    return {"documents": names, "count": len(names)}


# =============================================================================
# 文档上传
# =============================================================================

@app.post("/api/upload", tags=["文档"], summary="上传 EPUB 文档")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    上传 EPUB 文档并自动解析入库。
    支持 .epub 格式，入库后会自动刷新 BM25 索引。
    同名文档会先删除旧记录再插入新记录（幂等）。
    """
    # 检查文件类型
    if not file.filename or not file.filename.lower().endswith('.epub'):
        return JSONResponse(
            status_code=400,
            content={"error": "仅支持 EPUB 格式"}
        )
    
    # 生成安全的文件名（保留中文、字母、数字、连字符）
    import re
    safe_name = re.sub(r'[^\w\u4e00-\u9fff\-]', '_', os.path.splitext(file.filename)[0])
    temp_path = f"data/raw/{safe_name}.epub"
    os.makedirs("data/raw", exist_ok=True)
    
    # 保存上传的文件
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        logger.info("[Upload] 文件已保存: %s | 大小: %d bytes", temp_path, len(content))
    except Exception as e:
        logger.error("[Upload] 保存文件失败: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": "保存文件失败", "detail": str(e)}
        )
    
    # 调用入库逻辑（在线程池中执行，避免阻塞事件循环）
    try:
        result = await asyncio.to_thread(
            _ingest_module.process_single_file,
            temp_path,
            chunk_size=1024,
            chunk_overlap=128,
            append=True,
            hash_cache={},  # 空缓存，强制重新处理
        )
        chunks_count = result[0] if isinstance(result, tuple) else result
        logger.info("[Upload] 入库完成: %s | chunks=%d", safe_name, chunks_count)
    except Exception as e:
        logger.error("[Upload] 入库失败: %s", e, exc_info=True)
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(
            status_code=500,
            content={"error": "文档解析失败", "detail": str(e)}
        )
    
    # 刷新 BM25 索引
    try:
        request.app.state.doc_retriever.bm25_retriever.refresh_index()
        logger.info("[Upload] BM25 索引已刷新")
    except Exception as e:
        logger.warning("[Upload] BM25 刷新失败: %s", e)
    
    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return {
        "status": "success",
        "doc_name": safe_name,
        "chunks_count": chunks_count,
        "message": f"成功导入 {chunks_count} 个文档块" if chunks_count > 0 else "文档已是最新，无新内容导入"
    }


# =============================================================================
# 静态文件：serve frontend.html
# =============================================================================

# 挂载项目根目录，让前端可以直接访问
# 访问路径：http://localhost:8000/frontend.html
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# =============================================================================
# 启动入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    workers = int(os.getenv("API_WORKERS", "1"))

    logger.info("启动服务: %s:%d | reload=%s | workers=%d", host, port, reload, workers)

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
    )
