# -*- coding: utf-8 -*-
"""
标准 LangChain LLM 封装（公司级实践）

使用 LangChain 标准接口（BaseChatModel），不是自定义桥接
支持本地模型和 API 模型
"""

from typing import Optional, Iterator
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import pipeline, TextIteratorStreamer
import torch
from threading import Thread

from config.model_config_practice import LLM_CONFIG

logger = None  # 延迟导入，避免循环依赖


def get_logger():
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    return logger


class StandardLLM:
    """
    标准 LangChain LLM 工厂
    返回真正的 BaseChatModel，可在 LCEL 中直接使用
    """

    # 全局缓存（避免重复加载模型）
    _cache = {}

    @classmethod
    def create(cls, mode: str = None) -> BaseChatModel:
        """
        创建标准 LangChain ChatModel

        Args:
            mode: "local" 或 "api"，None 则使用默认配置

        Returns:
            BaseChatModel: 标准 LangChain 聊天模型
        """
        mode = mode or LLM_CONFIG["default_mode"]

        if mode not in cls._cache:
            if mode == "api":
                cls._cache[mode] = cls._create_api_llm()
            else:
                cls._cache[mode] = cls._create_local_llm()
            get_logger().info(f"创建标准 LLM 实例: {mode}")

        return cls._cache[mode]

    @classmethod
    def _create_api_llm(cls) -> BaseChatModel:
        """创建 API 模型（标准 ChatOpenAI 接口）"""
        api_config = LLM_CONFIG["api"]

        return ChatOpenAI(
            model=api_config["model"],
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            temperature=api_config["temperature"],
            max_tokens=api_config["max_tokens"],
            streaming=True,  # 启用流式
            timeout=api_config.get("timeout", 60),
        )

    @classmethod
    def _create_local_llm(cls) -> BaseChatModel:
        """创建本地模型（标准 HuggingFacePipeline + ChatHuggingFace 接口）"""
        local_config = LLM_CONFIG["local"]

        # 使用 HuggingFacePipeline 包装 transformers 模型
        pipe = pipeline(
            "text-generation",
            model=local_config["model_path"],
            torch_dtype=torch.float16 if local_config.get("torch_dtype") == "float16" else torch.float32,
            device_map=local_config.get("device_map", "auto"),
            max_new_tokens=local_config.get("max_new_tokens", 512),
            temperature=local_config.get("temperature", 0.7),
            do_sample=local_config.get("do_sample", True),
            top_p=local_config.get("top_p", 0.9),
            top_k=local_config.get("top_k", 50),
            trust_remote_code=True,
        )

        # 包装成 LangChain LLM
        hf_llm = HuggingFacePipeline(pipeline=pipe)

        # 转换为 ChatModel（支持标准 messages 接口）
        return ChatHuggingFace(llm=hf_llm)

    @classmethod
    def clear_cache(cls):
        """清理缓存（用于切换模型或释放显存）"""
        cls._cache.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        get_logger().info("LLM 缓存已清理")


# 便捷函数
def get_standard_llm(mode: str = None) -> BaseChatModel:
    """获取标准 LLM 实例（带缓存）"""
    return StandardLLM.create(mode)
