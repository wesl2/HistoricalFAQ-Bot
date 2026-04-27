# -*- coding: utf-8 -*-
"""
生产级 LLM 封装（练习版）

目标：实现一个统一 OpenAI 协议、支持异步 + 限流 + 重试 + 自动降级的 LLM 工厂。

【练习说明】
本文件只包含注释、TODO 提示和空函数框架。
你需要根据 TODO 中的提示，一步步补全代码。
每个 TODO 都标注了难度（⭐ 简单 / ⭐⭐ 中等 / ⭐⭐⭐ 困难）。

【核心概念】
1. OpenAI 协议统一：本地模型通过 vLLM/Ollama 提供 API，代码只用 ChatOpenAI
2. 异步非阻塞：FastAPI 环境下必须用 ainvoke，避免阻塞事件循环
3. 限流：asyncio.Semaphore 防止打爆后端推理服务
4. 重试：tenacity 库实现指数退避，只对网络超时等可重试异常生效
5. 自动降级：主模型挂了自动切备用模型（如 local → api）


关于协程：
"1000 个协程被包装成 1000 个 Task，由 Event Loop 在同一个线程里调度。
Event Loop 通过监听 IO Event 来决定该唤醒哪个 Task。"



【参考原脚本】
原脚本路径：/root/autodl-tmp/HistoricalFAQ-Bot/src/llm/standard_llm_new.py
"""

# =============================================================================
# TODO 1: ⭐ 导入依赖
# =============================================================================
# 提示：
# 1. asyncio, os, threading, typing（标准库）
# 2. httpx（HTTP 客户端，健康检查用）
# 3. langchain_core（BaseChatModel, AIMessage, BaseMessage, HumanMessage）
# 4. langchain_openai（ChatOpenAI）
# 5. openai 异常类型（APIConnectionError, APITimeoutError, InternalServerError, RateLimitError）
# 6. tenacity（retry, retry_if_exception, stop_after_attempt, wait_exponential）
# 7. 项目配置（API_PROVIDER_CONFIG, LLM_CONFIG）
#
# 思考：为什么用 httpx 而不用 requests？
#   - httpx 同时支持 sync 和 async（AsyncClient），API 统一
#   - requests 没有官方 async 支持

# TODO 1.1: 在这里写下所有 import
# ...
import asyncio,os,threading
from typing import Optional, AsyncIterator, List, Union
import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI #实现的BaseChatModel
from langchain_core.language_models import BaseChatModel
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from pathlib import Path
import logging,sys,os
project_root = Path(__file__).parent.parent.parent  # 根据文件层级调整
sys.path.insert(0,str(project_root))
from config.model_config_practice import API_PROVIDER_CONFIG, LLM_CONFIG
from concurrent.futures import ThreadPoolExecutor
'''
BaseChatModel 只定义"聊天模型应该有什么方法"，比如：
invoke(messages) → 同步调用
ainvoke(messages) → 异步调用
stream(messages) → 流式输出
它不涉及任何具体协议格式。
'''
# =============================================================================
# TODO 2: ⭐ 自定义异常
# =============================================================================
# 提示：
#   class LLMError(Exception): ...
#   class LLMUnavailableError(LLMError): ...
#
# 思考：为什么需要自定义异常？
#   1. 调用方可以用 except LLMError 捕获所有 LLM 相关错误
#   2. 和 LangChain/OpenAI 的内置异常区分开，便于上游处理
logger = logging.getLogger(__name__)
# TODO 2.1: 定义 LLMError（基类）

# TODO 2.2: 定义 LLMUnavailableError（服务不可用）
class LLMError(Exception):
    """LLM 调用异常基类"""
    pass


class LLMUnavailableError(LLMError):
    """LLM 服务不可用（健康检查失败或全部降级失败）"""
    pass


# =============================================================================
# TODO 3: ⭐⭐ 重试白名单判断函数
# =============================================================================
# 目标：判断一个异常是否"可重试"
#
# 白名单（可重试）：
#   - APIConnectionError / APITimeoutError：网络超时/断连
#   - RateLimitError：429 限流
#   - InternalServerError：503/502 服务端内部错误
#   - httpx.ReadTimeout / ConnectError / ReadError：底层网络异常
#
# 技巧：某些情况下 OpenAI SDK 会把 httpx 异常包装成 __cause__ 或 __context__
#      需要检查异常链才能发现底层原因。
#
# 函数签名：
#   def _is_retryable(exc: BaseException) -> bool:
#
# TODO 3.1: 先判断 exc 是否属于白名单类型（直接 isinstance）
# TODO 3.2: 再检查 exc.__cause__ 和 exc.__context__ 是否属于 httpx 异常
# TODO 3.3: 都不匹配返回 False
def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, 
                        RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, (httpx.ReadTimeout, httpx.ConnectError, 
                        httpx.ReadError, httpx.WriteError)):
        return True
    return False


# =============================================================================
# TODO 4: ⭐⭐⭐ 主类 StandardLLM
# =============================================================================
# 这是本文件的核心，你需要实现一个完整的工厂类。

class StandardLLM:
    """
    生产级 LLM 工厂（OpenAI 协议统一入口 + 自动降级）
    
    设计原则：
    1. 所有模型统一走 OpenAI API 协议（本地 vLLM/Ollama/云端）
    2. 线程安全 + loop-local 并发限流 + 精细化重试
    3. 内置自动降级：主模型挂了自动切备用
    4. FastAPI 等 async 框架主推 ainvoke / astream / abatch
    
    降级链（可配置）：
        local 挂了 → api
        api 挂了   → 无退路，直接抛异常
    """
    
    # -------------------------------------------------------------------------
    # TODO 4.1: ⭐ 类变量定义
    # -------------------------------------------------------------------------
    # 提示：
    #   _cache: dict[str, BaseChatModel] = {}           # ChatOpenAI client 缓存
    #   _sync_lock = threading.Lock()                   # create() 线程锁
    #   _sem_cache: dict = {}                           # loop_id -> Semaphore
    #   _http_client: Optional[httpx.Client] = None     # 同步 HTTP Client
    #   _async_http_client: Optional[httpx.AsyncClient] = None  # 异步 HTTP Client
    #   _FALLBACK_CHAIN = {"local": "api", "api": None}  # 降级链
    #
    # 思考：
    #   - 为什么缓存的是 ChatOpenAI client 而不是模型权重？
    #     因为模型权重由后端 vLLM/Ollama/云端托管，client 只是 HTTP 客户端。
    #   - 为什么不用 asyncio.Lock() 保护 _cache？
    #     create() 是同步方法，可能被多线程调用，要用 threading.Lock。
    #     acreate() 是异步方法，但内部通过 asyncio.to_thread 调用 create，
    #     所以线程锁已经够用了。
    _cache:dict[str,BaseChatModel] = {} #"api":ChatOpenAI(...)实例
    _sync_lock = threading.Lock()
    _sem_cache:dict = {}
    _http_client: Optional[httpx.Client] = None
    _async_http_client: Optional[httpx.AsyncClient] = None
    _FALLBACK_CHAIN = {"local": "api", "api": None}
    # -------------------------------------------------------------------------
    # TODO 4.2: ⭐⭐ Semaphore（loop-local）
    # -------------------------------------------------------------------------
    # 目标：获取当前 event loop 的 Semaphore，避免跨 loop 隐患。
    #
    # 实现思路：
    #   1. 调用 asyncio.get_running_loop() 获取当前 loop
    #   2. 用 id(loop) 作为 key，在 _sem_cache 里查找
    #   3. 如果不存在，创建一个 asyncio.Semaphore(max_concurrency) 并存入
    #   4. 返回 Semaphore
    #
    # 边界情况：
    #   - 如果不在 async 环境（RuntimeError），返回一个 dummy Semaphore(999999)
    #
    # 类方法签名：
    #   @classmethod
    #   def _get_sem(cls) -> asyncio.Semaphore:
    #
    # 提示：用 os.getenv("LLM_MAX_CONCURRENCY", "100") 读取并发上限
    @classmethod
    def _get_sem(cls) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.Semaphore(999999)
        loop_id = id(loop)
        if loop_id not in cls._sem_cache:
            cls._sem_cache[loop_id] = asyncio.Semaphore(
                int(os.getenv("LLM_MAX_CONCURRENCY", "100"))
            )  #限制并发量
        return cls._sem_cache[loop_id]
    # -------------------------------------------------------------------------
    # TODO 4.3: ⭐⭐ HTTP Client（复用连接池）
    # -------------------------------------------------------------------------
    # 目标：创建并复用 httpx Client，避免每次健康检查都新建 TCP 连接。
    #
    # 实现思路：
    #   - 同步版本：检查 _http_client 是否为 None，是则创建 httpx.Client(timeout=5)
    #   - 异步版本：检查 _async_http_client 是否为 None，是则创建 httpx.AsyncClient(timeout=5)
    #
    # 类方法签名：
    #   @classmethod
    #   def _get_http_client(cls) -> httpx.Client:
    #
    #   @classmethod
    #   async def _get_async_http_client(cls) -> httpx.AsyncClient:
    @classmethod
    def _get_http_client(cls) -> httpx.Client:
        if cls._http_client is None:
            cls._http_client = httpx.Client(timeout=5)
        return cls._http_client
    
    @classmethod
    def _get_async_http_client(cls) -> httpx.AsyncClient:
        if cls._async_http_client is None:
            cls._async_http_client = httpx.AsyncClient(timeout=5)
        return cls._async_http_client

    # -------------------------------------------------------------------------
    # TODO 4.4: ⭐⭐ 工厂方法 create()（线程安全）
    # -------------------------------------------------------------------------
    # 目标：创建 ChatOpenAI 实例，带缓存和线程锁。
    #
    # 实现思路（双重检查锁定 DCL）：
    #   1. mode = mode or LLM_CONFIG["default_mode"]
    #   2. 快速路径：如果 mode 在 _cache 中，直接返回
    #   3. 慢速路径：加 threading.Lock
    #   4. 再次检查 _cache（双重检查）
    #   5. 调用 _resolve_config(mode) 获取配置
    #   6. 调用 _build_llm(cfg) 创建 ChatOpenAI 实例
    #   7. 存入 _cache 并记录日志
    #   8. 返回实例
    #
    # 思考：为什么叫"双重检查"？因为第 2 步和第 4 步都检查了 _cache，
    #      避免多个线程同时进入慢速路径时重复创建。
    #
    # 类方法签名：
    #   @classmethod
    #   def create(cls, mode: Optional[str] = None) -> BaseChatModel:
    @classmethod
    def create(cls,mode: Optional[str] = None) -> BaseChatModel:
        mode = mode or LLM_CONFIG["default_mode"]
        if mode in cls._cache:
            return cls._cache[mode]
        with cls._sync_lock: #threading.Lock() 线程锁，保护下面的代码块 同步的
            if mode not in cls._cache: #双重检查锁定（DCL）
                cfg = cls._resolve_config(mode)
                cls._cache[mode] = cls._build_llm(cfg)
                logger.info(
                    "[LLM] Client 创建成功 | mode=%s | model=%s | base_url=%s",
                    mode,
                    cfg["model"],
                    cfg.get("base_url", "default"),
                )
        return cls._cache[mode]
    # -------------------------------------------------------------------------
    # TODO 4.5: ⭐⭐ 工厂方法 acreate()（异步入口）
    # -------------------------------------------------------------------------
    # 目标：异步获取/创建 LLM 实例，不阻塞事件循环。
    #
    # 实现思路：
    #   1. 检查 _cache，已存在直接返回
    #   2. 如果不存在，用 asyncio.to_thread(cls.create, mode) 扔到线程池执行
    #   3. 返回结果
    #
    # 思考：为什么不直接用 asyncio.Lock？
    #   因为 create() 内部已经有 threading.Lock 了，
    #   acreate() 只需要确保    #       "异步调用时，create() 的同步锁不会阻塞事件循环"——因为 to_thread 把锁操作放到了后台线程。
    #
    # 类方法签名：
    #   @classmethod
    #   async def acreate(cls, mode: Optional[str] = None) -> BaseChatModel:

    @classmethod
    async def acreate(cls, mode: Optional[str] = None) -> BaseChatModel:
        mode = mode or LLM_CONFIG["default_mode"]
        if mode in cls._cache:
            return cls._cache[mode]
        
        return await asyncio.to_thread(cls.create, mode)
        #e = ThreadPoolExecutor(max_workers=10)
        #return await asyncio.get_running_loop().run_in_executor(e, cls.create, mode)

    # -------------------------------------------------------------------------
    # TODO 4.6: ⭐ 同步调用（兼容旧代码）
    # -------------------------------------------------------------------------
    # 目标：invoke() 和 stream()，供非 async 环境使用。
    #
    # invoke 实现思路：
    #   1. llm = cls.create(mode)
    #   2. 如果 messages 是 str，包装成 [HumanMessage(content=messages)]
    #   3. return llm.invoke(messages, **kwargs)
    #
    # stream 实现思路类似，但返回 generator。
    #
    # 类方法签名：
    #   @classmethod
    #   def invoke(cls, messages, mode=None, **kwargs) -> AIMessage:
    @classmethod
    def invoke(cls,messages=Union[str, List[BaseMessage]],mode=None,**kwargs
               ) -> AIMessage:
        llm = cls.create(mode)
        messages = cls._normalize_messages(messages)
        return llm.invoke(messages, **kwargs)

    #   @classmethod
    #   def stream(cls, messages, mode=None, **kwargs):

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
    # -------------------------------------------------------------------------
    # TODO 4.7: ⭐⭐⭐ 核心异步调用（最复杂）
    # -------------------------------------------------------------------------
    # 你需要实现三个互相配合的方法：_ainvoke_core, ainvoke
    #
    # ---- 4.7.1 _ainvoke_core：带限流 + 重试的"裸"调用 ----
    #
    # 装饰器（tenacity.retry）：
    #   - stop=stop_after_attempt(3)                  # 最多重试 3 次
    #   - wait=wait_exponential(multiplier=1, min=1, max=10)  # 指数退避
    #   - retry=retry_if_exception(_is_retryable)     # 只对白名单异常重试
    #   - reraise=True                                # 重试耗尽后把原异常抛出去
    #
    # 方法体：
    #   1. llm = await cls.acreate(mode)
    #   2. async with cls._get_sem():                 # 限流
    #   3. return await llm.ainvoke(messages, **kwargs)
    #
    # 类方法签名：
    #   @classmethod
    #   @retry(...)
    #   async def _ainvoke_core(cls, messages, mode, **kwargs) -> AIMessage:
    #
    @classmethod
    @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)   #  注意：流式是不能retry的，因为无法判断是否成功
    async def _ainvoke_core(cls,messages,mode,**kwargs) -> AIMessage:
      llm = await cls.acreate(mode)  
      async with cls._get_sem():
          return await llm.ainvoke(messages, **kwargs) #限制最大并发量

    # ---- 4.7.2 ainvoke：核心 + 自动降级 ----
    #
    # 实现思路：
    #   1. mode = mode or LLM_CONFIG["default_mode"]
    #   2. messages = cls._normalize_messages(messages)
    #   3. try:
    #          return await cls._ainvoke_core(messages, mode, **kwargs)
    #      except Exception as e:
    #          # 白名单异常已经被 @retry 处理过（重试 3 次仍失败）
    #          # 走到这里说明：非白名单异常 或 重试耗尽
    #          fb = fallback_mode or cls._FALLBACK_CHAIN.get(mode)
    #          if fb:
    #              logger.warning("...降级到 %s", fb)
    #              try:
    #                  return await cls._ainvoke_core(messages, fb, **kwargs)
    #              except Exception as e2:
    #                  raise LLMUnavailableError(f"主模型 {mode} 和降级模型 {fb} 均不可用") from e2
    #          raise
    #
    # 参数 fallback_mode：显式指定降级目标，None 则用内置降级链
    #
    # 类方法签名：
    #   @classmethod
    #   async def ainvoke(cls, messages, mode=None, fallback_mode=None, **kwargs) -> AIMessage:
    @classmethod
    async def ainvoke(cls,messages,mode=None,fallback_mode=None,**kwargs) ->AIMessage:
        """
        异步调用 LLM
        
        Args:
            messages: 聊天消息列表
            mode: LLM 模式
            fallback_mode: 降级模式
            kwargs: 其他参数
        
        Returns:
            AIMessage: LLM 输出
        """
        mode = mode or LLM_CONFIG["default_mode"]
        messages = cls._normalize_messages(messages)
        try:
            return await cls._ainvoke_core(messages,mode,**kwargs)
        except Exception as e:
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
            #到最后一步才raise抛出异常 前面做了所有情况的处理
            '''
            | 写法                        | 行为              | 原始异常还在吗？          |
            | ------------------------- | --------------- | ----------------- |
            | `raise`                   | 原样抛出当前异常        | ✅ 完全保留            |
            | `raise NewError()`        | 抛出新异常           | ❌ 丢失              |
            | `raise NewError() from e` | 抛出新异常，把 `e` 挂上去 | ✅ 保留在 `__cause__` |

            '''
    # -------------------------------------------------------------------------
    # TODO 4.8: ⭐⭐⭐ 异步流式 astream（带降级）
    # -------------------------------------------------------------------------
    # 目标：和 ainvoke 类似，但返回 AsyncIterator，用于 SSE 推送。
    #
    # 难点：流式 generator 里做 try/except/降级 比较 tricky。
    # 建议：外层 try 捕获，如果失败则重新 create fallback 模型的 client，
    #      然后重新走 stream。
    #
    # 类方法签名：
    #   @classmethod
    #   async def astream(cls, messages, mode=None, fallback_mode=None, **kwargs) -> AsyncIterator[BaseMessage]:
    @classmethod
    async def astream(cls,messages,mode= None,fallback_mode = None,**kwargs):
        mode = mode or LLM_CONFIG["default_mode"]
        messages = cls._normalize_messages(messages)
        try:
            llm = await cls.acreate(mode)
            async with cls._get_sem():
                async for chunk in llm.astream(messages,**kwargs):
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
                    async for chunk in llm.astream(messages,**kwargs):
                        yield chunk
            else:
                raise
    # TODO 4.9: ⭐⭐ 异步批量 abatch（带降级）
    # -------------------------------------------------------------------------
    # 目标：批量发送多个对话，底层由 LangChain 自动 batching。
    #
    # 实现思路和 ainvoke 类似，只是调用的是 llm.abatch()。
    #
    # 类方法签名：
    #   @classmethod
    #   async def abatch(cls, messages_list, mode=None, fallback_mode=None, **kwargs) -> List[AIMessage]:
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
    
    # -------------------------------------------------------------------------
    # TODO 4.10: ⭐⭐ 健康检查（轻量化）
    # -------------------------------------------------------------------------
    # 目标：探测 LLM 服务是否可用，优先不占用 GPU。
    #
    # sync 版本 health_check：
    #   1. 获取 base_url
    #   2. 优先 HTTP GET /v1/models（用 _get_http_client()）
    #   3. 如果 HTTP 失败，fallback 到最小推理（max_tokens=1, temperature=0, timeout=5）
    #
    # async 版本 ahealth_check：
    #   1. 优先 HTTP GET /v1/models（用 _get_async_http_client()）
    #   2. fallback 到 asyncio.to_thread(cls.health_check, mode)
    #
    # 类方法签名：
    #   @classmethod
    #   def health_check(cls, mode=None) -> bool:
    #
    #   @classmethod
    #   async def ahealth_check(cls, mode=None) -> bool:
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

    # -------------------------------------------------------------------------
    # TODO 4.11: ⭐ 清理缓存
    # -------------------------------------------------------------------------
    # 目标：clear_cache()，清空 _cache。
    #
    # 类方法签名：
    #   @classmethod
    #   def clear_cache(cls) -> None:
    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
    # -------------------------------------------------------------------------
    # TODO 4.12: ⭐ 内部工具方法
    # -------------------------------------------------------------------------
    # _normalize_messages(messages):
    #   - 如果 messages 是 str，返回 [HumanMessage(content=messages)]
    #   - 否则原样返回
    @classmethod
    def _normalize_messages(cls,
        messages: Union[str, List[BaseMessage]]
    ) -> List[BaseMessage]:
        if isinstance(messages, str):
            return [HumanMessage(content=messages)]
        else:
            return messages
    # _resolve_config(mode):
    #   - mode == "api"：从 LLM_CONFIG["api"] 读取，补全 base_url
    #   - mode == "local"：从环境变量 / LLM_CONFIG["local"] 读取，统一为 OpenAI 格式
    #   - 注意：统一使用 max_tokens，不再混用 max_new_tokens
    @classmethod
    def _resolve_config(cls,mode:str) -> dict:
        if mode == "api":
            cfg = LLM_CONFIG["api"].copy()
            if not cfg.get("base_url"):
                provider_name = cfg.get("provider","deepseek")
                provider_defaults = API_PROVIDER_CONFIG.get(provider_name,{})
                cfg["base_url"] = provider_defaults.get("base_url","")  
                cfg["model"] = cfg.get("model") or provider_defaults.get("model", "unknown")
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
    # _build_llm(cfg):
    #   - 用 cfg 里的参数创建 ChatOpenAI 实例
    #   - 记得 streaming=True

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



# =============================================================================
# TODO 5: ⭐ 便捷函数
# =============================================================================
# 提示：
#   def get_llm(mode=None) -> BaseChatModel:
#       return StandardLLM.create(mode)
#
#   async def get_llm_async(mode=None) -> BaseChatModel:
#       return await StandardLLM.acreate(mode)

def get_llm(mode: Optional[str] = None) -> BaseChatModel:
    return StandardLLM.create(mode)


async def get_llm_async(mode: Optional[str] = None) -> BaseChatModel:
    return await StandardLLM.acreate(mode)
# =============================================================================
# TODO 6: ⭐ 使用示例
# =============================================================================
# 在 if __name__ == "__main__": 里写一个 demo 函数：
#   1. StandardLLM.ainvoke("请简要介绍王洪文")
#   2. StandardLLM.astream("你好") 流式输出
#   3. StandardLLM.ahealth_check()
#   4. StandardLLM.clear_cache()


# =============================================================================
# 练习检查清单
# =============================================================================
"""
□ 正确导入所有依赖（asyncio, httpx, langchain, openai, tenacity）
□ 定义 LLMError 和 LLMUnavailableError
□ _is_retryable 正确判断白名单异常（含异常链检查）
□ StandardLLM 类变量定义正确（_cache, _sync_lock, _sem_cache, _http_client...）
□ _get_sem 是 loop-local（延迟初始化，id(loop) 做 key）
□ _get_http_client / _get_async_http_client 复用连接池
□ create() 使用双重检查锁定（DCL），线程安全
□ acreate() 用 asyncio.to_thread，不阻塞事件循环
□ invoke / stream 实现正确，兼容 str 输入
□ _ainvoke_core 带 @retry 装饰器（正确配置 stop/wait/retry/reraise）
□ _ainvoke_core 内部用 async with cls._get_sem() 限流
□ ainvoke 实现自动降级（try → fallback → LLMUnavailableError）
□ astream 支持降级（generator 里重新 yield fallback 结果）
□ abatch 支持降级
□ health_check 优先 HTTP /v1/models，fallback 最小推理
□ ahealth_check 使用 AsyncClient
□ _normalize_messages 把 str 转成 HumanMessage 列表
□ _resolve_config 正确处理 api/local 两种 mode
□ _build_llm 返回配置好的 ChatOpenAI
□ 便捷函数 get_llm / get_llm_async 正常可用
□ 测试代码能运行并打印结果
"""
