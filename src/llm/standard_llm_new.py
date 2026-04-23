# -*- coding: utf-8 -*-
"""
生产级 LLM 封装 v3（OpenAI 协议统一入口 + 自动降级）

修复清单（第二轮 Code Review）：
1. ✅ Semaphore 延迟初始化（loop-local），避免跨 event loop 隐患
2. ✅ HTTP Client 复用（httpx.Client / AsyncClient）
3. ✅ 重试白名单扩充：加入 httpx 底层异常（ReadTimeout / ConnectError）
4. ✅ 文档修正：缓存的是 HTTP client，不是模型权重
5. ✅ 自动降级：主模型挂了自动切备用（local → api → 抛异常）
"""

import asyncio
import logging
import os
import threading
from typing import AsyncIterator, List, Optional, Union

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.model_config_practice import API_PROVIDER_CONFIG, LLM_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """LLM 调用异常基类"""
    pass


class LLMUnavailableError(LLMError):
    """LLM 服务不可用（健康检查失败或全部降级失败）"""
    pass


# ---------------------------------------------------------------------------
# 重试判断：白名单机制（仅对可重试异常重试）
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """
    判断异常是否可重试。
    
    白名单：
    - OpenAI SDK: APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
    - httpx 底层: ReadTimeout, ConnectError, ReadError
    """
    # 直接类型匹配
    if isinstance(exc, (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
    )):
        return True
    
    # 检查异常链：某些情况下 OpenAI SDK 会包装 httpx 异常
    cause = getattr(exc, '__cause__', None) or getattr(exc, '__context__', None)
    if cause and isinstance(cause, (
        httpx.ReadTimeout,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
    )):
        return True
    
    return False


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class StandardLLM:
    """
    生产级 LLM 工厂（OpenAI 协议统一入口 + 自动降级）

    核心设计：
    1. 所有模型统一走 OpenAI API 协议（本地 vLLM/Ollama/云端）
    2. 线程安全 + loop-local 并发限流 + 精细化重试
    3. 内置自动降级：主模型挂了自动切备用
    4. FastAPI 等 async 框架主推 ainvoke / astream / abatch

    降级链（可配置）：
        local 挂了 → api
        api 挂了   → 无退路，直接抛异常
    """

    # -----------------------------------------------------------------------
    # 缓存 & 锁
    # -----------------------------------------------------------------------
    
    # ChatOpenAI HTTP client 缓存（轻量对象，不是模型权重）
    _cache: dict[str, BaseChatModel] = {}
    _sync_lock = threading.Lock()   # create() 线程锁
    
    # loop-local Semaphore：避免跨 event loop 隐患
    _sem_cache: dict = {}           # loop_id -> Semaphore

    # HTTP Client 复用
    _http_client: Optional[httpx.Client] = None
    _async_http_client: Optional[httpx.AsyncClient] = None

    # 降级链配置
    _FALLBACK_CHAIN = {
        "local": "api",   # 本地模型挂了切云端 API
        "api": None,      # 云端 API 没退路
    }

    # -----------------------------------------------------------------------
    # Semaphore（loop-local）
    # -----------------------------------------------------------------------

    @classmethod
    def _get_sem(cls) -> asyncio.Semaphore:
        """
        获取当前 event loop 的 Semaphore。
        
        避免类变量 Semaphore 跨 loop 使用的隐患。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 不在 async 环境，返回一个 dummy（异步调用不会走到这里）
            return asyncio.Semaphore(999999)
        
        loop_id = id(loop)
        if loop_id not in cls._sem_cache:
            cls._sem_cache[loop_id] = asyncio.Semaphore(
                int(os.getenv("LLM_MAX_CONCURRENCY", "100"))
            )
        return cls._sem_cache[loop_id]

    # -----------------------------------------------------------------------
    # HTTP Client（复用连接池）
    # -----------------------------------------------------------------------

    @classmethod
    def _get_http_client(cls) -> httpx.Client:
        """获取同步 HTTP Client（复用连接池）"""
        if cls._http_client is None:
            cls._http_client = httpx.Client(timeout=5)
        return cls._http_client

    @classmethod
    async def _get_async_http_client(cls) -> httpx.AsyncClient:
        """获取异步 HTTP Client（复用连接池）"""
        if cls._async_http_client is None:
            cls._async_http_client = httpx.AsyncClient(timeout=5)
        return cls._async_http_client

    # -----------------------------------------------------------------------
    # 工厂方法
    # -----------------------------------------------------------------------

    @classmethod
    def create(cls, mode: Optional[str] = None) -> BaseChatModel:
        """
        同步入口（线程安全）。

        缓存的是 ChatOpenAI HTTP client（轻量对象），不是模型权重。
        模型权重由后端的 vLLM/Ollama/云端服务托管。
        """
        mode = mode or LLM_CONFIG["default_mode"]

        if mode in cls._cache:
            return cls._cache[mode]

        # 双重检查锁定（DCL）
        with cls._sync_lock:
            if mode not in cls._cache:
                cfg = cls._resolve_config(mode)
                cls._cache[mode] = cls._build_llm(cfg)
                logger.info(
                    "[LLM] Client 创建成功 | mode=%s | model=%s | base_url=%s",
                    mode,
                    cfg["model"],
                    cfg.get("base_url", "default"),
                )
        return cls._cache[mode]

    @classmethod
    async def acreate(cls, mode: Optional[str] = None) -> BaseChatModel:
        """
        异步入口（协程安全，不阻塞事件循环）。
        """
        mode = mode or LLM_CONFIG["default_mode"]

        if mode in cls._cache:
            return cls._cache[mode]

        # ChatOpenAI 初始化很轻，扔线程池避免阻塞事件循环
        return await asyncio.to_thread(cls.create, mode)

    # -----------------------------------------------------------------------
    # 同步调用（兼容旧代码，生产不推荐在 async 路由里用）
    # -----------------------------------------------------------------------

    @classmethod
    def invoke(
        cls,
        messages: Union[str, List[BaseMessage]],
        mode: Optional[str] = None,
        **kwargs,
    ) -> AIMessage:
        llm = cls.create(mode)
        messages = cls._normalize_messages(messages)
        return llm.invoke(messages, **kwargs)

    @classmethod
    def stream(
        cls,
        messages: Union[str, List[BaseMessage]],
        mode: Optional[str] = None,
        **kwargs,
    ):
        llm = cls.create(mode)
        messages = cls._normalize_messages(messages)
        yield from llm.stream(messages, **kwargs)

    # -----------------------------------------------------------------------
    # 核心异步调用（带限流 + 重试 + 自动降级）
    # -----------------------------------------------------------------------

    @classmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    async def _ainvoke_core(
        cls,
        messages: List[BaseMessage],
        mode: str,
        **kwargs,
    ) -> AIMessage:
        """
        核心调用（不带降级逻辑）。
        
        限流 + 重试在这个层级实现。
        """
        llm = await cls.acreate(mode)

        async with cls._get_sem():
            return await llm.ainvoke(messages, **kwargs)

    @classmethod
    async def ainvoke(
        cls,
        messages: Union[str, List[BaseMessage]],
        mode: Optional[str] = None,
        fallback_mode: Optional[str] = None,
        **kwargs,
    ) -> AIMessage:
        """
        异步调用（生产标准入口）+ 自动降级。

        降级逻辑：
        1. 先尝试主模型（mode）
        2. 若主模型彻底不可用（非白名单异常，如 401/404/连接失败），自动切 fallback
        3. fallback 也失败则抛 LLMUnavailableError

        Args:
            fallback_mode: 显式指定降级目标；None 则使用内置降级链
        """
        mode = mode or LLM_CONFIG["default_mode"]
        messages = cls._normalize_messages(messages)

        try:
            return await cls._ainvoke_core(messages, mode, **kwargs)
        except Exception as e:
            # 白名单异常已经被 @retry 处理过（重试 3 次仍失败）
            # 走到这里说明：非白名单异常 或 重试耗尽
            fb = fallback_mode or cls._FALLBACK_CHAIN.get(mode)
            if fb:
                logger.warning(
                    "[LLM] mode=%s 调用失败（%s: %s），降级到 %s",
                    mode,
                    type(e).__name__,
                    e,
                    fb,
                )
                try:
                    return await cls._ainvoke_core(messages, fb, **kwargs)
                except Exception as e2:
                    raise LLMUnavailableError(
                        f"主模型 {mode} 和降级模型 {fb} 均不可用"
                    ) from e2
            raise

    @classmethod
    async def astream(
        cls,
        messages: Union[str, List[BaseMessage]],
        mode: Optional[str] = None,
        fallback_mode: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[BaseMessage]:
        """
        异步流式（带降级）。
        """
        mode = mode or LLM_CONFIG["default_mode"]
        messages = cls._normalize_messages(messages)

        try:
            llm = await cls.acreate(mode)
            async with cls._get_sem():
                async for chunk in llm.astream(messages, **kwargs):
                    yield chunk
        except Exception as e:
            fb = fallback_mode or cls._FALLBACK_CHAIN.get(mode)
            if fb:
                logger.warning(
                    "[LLM] mode=%s 流式失败，降级到 %s",
                    mode,
                    fb,
                )
                llm = await cls.acreate(fb)
                async with cls._get_sem():
                    async for chunk in llm.astream(messages, **kwargs):
                        yield chunk
            else:
                raise

    @classmethod
    async def abatch(
        cls,
        messages_list: List[List[BaseMessage]],
        mode: Optional[str] = None,
        fallback_mode: Optional[str] = None,
        **kwargs,
    ) -> List[AIMessage]:
        """
        异步批量（带降级）。
        """
        mode = mode or LLM_CONFIG["default_mode"]

        try:
            llm = await cls.acreate(mode)
            async with cls._get_sem():
                return await llm.abatch(messages_list, **kwargs)
        except Exception as e:
            fb = fallback_mode or cls._FALLBACK_CHAIN.get(mode)
            if fb:
                logger.warning(
                    "[LLM] mode=%s 批量失败，降级到 %s",
                    mode,
                    fb,
                )
                llm = await cls.acreate(fb)
                async with cls._get_sem():
                    return await llm.abatch(messages_list, **kwargs)
            raise

    # -----------------------------------------------------------------------
    # 健康检查（轻量化：复用 HTTP Client）
    # -----------------------------------------------------------------------

    @classmethod
    def health_check(cls, mode: Optional[str] = None) -> bool:
        """
        同步健康检查：优先 HTTP /v1/models 探测，不占用 GPU。
        """
        mode = mode or LLM_CONFIG["default_mode"]
        cfg = cls._resolve_config(mode)
        base_url = cfg.get("base_url", "")

        # 1. 优先 HTTP 轻量探测
        if base_url:
            try:
                client = cls._get_http_client()
                resp = client.get(
                    f"{base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {cfg['api_key']}"},
                )
                if resp.status_code == 200:
                    return True
            except Exception:
                pass

        # 2. Fallback：最小化推理请求
        try:
            llm = cls.create(mode)
            llm.invoke(
                [HumanMessage(content="ping")],
                max_tokens=1,
                temperature=0,
                timeout=5,
            )
            return True
        except Exception as e:
            logger.warning("[LLM] 健康检查失败: %s", e)
            return False

    @classmethod
    async def ahealth_check(cls, mode: Optional[str] = None) -> bool:
        """
        异步健康检查：使用 AsyncClient。
        """
        mode = mode or LLM_CONFIG["default_mode"]
        cfg = cls._resolve_config(mode)
        base_url = cfg.get("base_url", "")

        # 1. 优先 HTTP 轻量探测
        if base_url:
            try:
                client = await cls._get_async_http_client()
                resp = await client.get(
                    f"{base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {cfg['api_key']}"},
                )
                if resp.status_code == 200:
                    return True
            except Exception:
                pass

        # 2. Fallback：扔线程池执行同步 health_check
        return await asyncio.to_thread(cls.health_check, mode)

    # -----------------------------------------------------------------------
    # 工具方法
    # -----------------------------------------------------------------------

    @classmethod
    def clear_cache(cls) -> None:
        """清理 Client 缓存"""
        cls._cache.clear()
        logger.info("[LLM] Client 缓存已清理")

    # -----------------------------------------------------------------------
    # 内部实现
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_messages(
        messages: Union[str, List[BaseMessage]]
    ) -> List[BaseMessage]:
        if isinstance(messages, str):
            return [HumanMessage(content=messages)]
        return messages

    @classmethod
    def _resolve_config(cls, mode: str) -> dict:
        if mode == "api":
            cfg = LLM_CONFIG["api"].copy()
            provider = cfg.get("provider", "deepseek")
            provider_defaults = API_PROVIDER_CONFIG.get(provider, {})
            if not cfg.get("base_url"):
                cfg["base_url"] = provider_defaults.get("base_url", "")
            if not cfg.get("model"):
                cfg["model"] = provider_defaults.get("model", "unknown")
            return cfg

        local_cfg = LLM_CONFIG["local"]
        return {
            "model": os.getenv("LOCAL_LLM_MODEL_NAME", "local-model"),
            "api_key": os.getenv("LOCAL_LLM_API_KEY", "dummy"),
            "base_url": os.getenv(
                "LOCAL_LLM_BASE_URL",
                "http://localhost:11434/v1",
            ),
            "temperature": float(
                os.getenv("LOCAL_LLM_TEMPERATURE", local_cfg.get("temperature", 0.7))
            ),
            "max_tokens": int(
                os.getenv("LOCAL_LLM_MAX_TOKENS", local_cfg.get("max_new_tokens", 512))
            ),
            "timeout": int(os.getenv("LOCAL_LLM_TIMEOUT", "120")),
            "max_retries": int(os.getenv("LOCAL_LLM_MAX_RETRIES", "2")),
        }

    @classmethod
    def _build_llm(cls, cfg: dict) -> BaseChatModel:
        return ChatOpenAI(
            model=cfg["model"],
            api_key=cfg["api_key"],
            base_url=cfg.get("base_url"),
            temperature=cfg.get("temperature", 0.7),
            max_tokens=cfg.get("max_tokens", 512),
            timeout=cfg.get("timeout", 60),
            max_retries=cfg.get("max_retries", 2),
            streaming=True,
        )


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def get_llm(mode: Optional[str] = None) -> BaseChatModel:
    return StandardLLM.create(mode)


async def get_llm_async(mode: Optional[str] = None) -> BaseChatModel:
    return await StandardLLM.acreate(mode)


# ---------------------------------------------------------------------------
# 使用示例
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def demo():
        # 1. 异步调用（带自动降级）
        #os.environ["LLM_MODE"] = "api"
        #os.environ["API_PROVIDER"] = "moonshot"
        #os.environ["API_KEY"] = "sk-ki...LsePT"


        resp = await StandardLLM.ainvoke("李世民的丰功伟绩", mode="api")
        print(f"[回答] {resp.content[:]}...")

        # 2. 流式
        print("[流式] ", end="", flush=True)
        async for chunk in StandardLLM.astream("你好"):
            print(chunk.content, end="", flush=True)
        print()

        # 3. 健康检查
        ok = await StandardLLM.ahealth_check()
        print(f"[健康] {'正常' if ok else '异常'}")

        StandardLLM.clear_cache()

    asyncio.run(demo())
