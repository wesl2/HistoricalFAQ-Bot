# -*- coding: utf-8 -*-
"""
LLM 模块

生产级 LLM 封装（StandardLLM）
"""

from .standard_llm_new import StandardLLM, get_llm, get_llm_async

__all__ = ['StandardLLM', 'get_llm', 'get_llm_async']
