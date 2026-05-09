#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Query 检索扩展（LLM-based Query Expansion）

功能：
1. 用 LLM 将用户原始 query 扩展为多个语义变体
2. 对每个变体执行检索
3. 合并去重结果，提升召回率

开关控制：
- 通过环境变量 MULTI_QUERY_ENABLED=true/false 全局开关
- 通过 DocRetriever 构造函数参数 enable_multi_query 实例级开关

成本分析（以 DeepSeek 为例）：
- Query Expansion: 1 次 LLM 调用，~300 tokens，~2000ms
- 3 个扩展 query 并行检索：Embedding 80ms + 向量检索 50ms（并行后实际 +50ms）
- 总时间增加：~2.5s
- Token 成本增加：prompt 长度 × 3（因为召回文档变多了）
"""

import os
import logging
from typing import List
from langchain_core.messages import HumanMessage

from src.llm.standard_llm_new import StandardLLM

logger = logging.getLogger(__name__)

# 全局开关（环境变量控制）
MULTI_QUERY_ENABLED = os.getenv("MULTI_QUERY_ENABLED", "false").lower() == "true"

# 默认扩展 query 数量
MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "3"))

# Prompt 模板（让 LLM 生成多角度查询）
MULTI_QUERY_PROMPT_TEMPLATE = """你是一个专业的历史研究检索助手。
请根据用户的问题，生成 {n} 个不同角度的查询变体，帮助从多个维度检索相关资料。

要求：
1. 每个变体聚焦不同的子主题或关键词
2. 保持中文，使用历史领域专业术语
3. 只输出查询列表，每行一个，不要解释

用户问题：{query}

查询变体："""


async def expand_query(query: str, n: int = None) -> List[str]:
    """
    用 LLM 扩展 query 为多角度查询变体

    Args:
        query: 原始用户查询
        n: 生成变体数量（默认读环境变量 MULTI_QUERY_COUNT）

    Returns:
        扩展后的查询列表（包含原始 query 在第一位）
    """
    n = n or MULTI_QUERY_COUNT
    prompt = MULTI_QUERY_PROMPT_TEMPLATE.format(query=query, n=n)

    try:
        resp = await StandardLLM.ainvoke(prompt, mode="api")
        raw = resp.content.strip()

        # 解析 LLM 输出：每行一个查询
        variants = []
        for line in raw.split("\n"):
            line = line.strip()
            # 去除序号标记（如 "1."、"-"、"*"）
            line = line.lstrip("0123456789.-* ")
            if line and len(line) > 3:
                variants.append(line)

        # 去重并保留原始 query
        seen = {query}
        result = [query]
        for v in variants:
            if v not in seen:
                seen.add(v)
                result.append(v)

        # 限制数量
        result = result[: n + 1]
        logger.info(f"[MultiQuery] 扩展完成 | 原始: '{query}' → {len(result)} 个变体")
        return result

    except Exception as e:
        logger.warning(f"[MultiQuery] LLM 扩展失败，回退到原始 query: {e}")
        return [query]


def expand_query_sync(query: str, n: int = None) -> List[str]:
    """
    同步版本的 query 扩展（用于 sync 场景）
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # 如果在 async 环境，调度到线程池
        return asyncio.run_coroutine_threadsafe(expand_query(query, n), loop).result()
    except RuntimeError:
        # 不在 async 环境，直接创建新 loop
        return asyncio.run(expand_query(query, n))
