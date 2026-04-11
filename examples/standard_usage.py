#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准 LangChain 使用示例

展示如何使用新创建的标准模块
"""

import sys
import os

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.rag.standard_rag import create_standard_rag, StandardRAGSystem
from src.rag.standard_streaming import stream_rag_response


def example_1_basic_query():
    """示例 1: 基本查询"""
    print("=" * 60)
    print("示例 1: 基本查询")
    print("=" * 60)

    rag = create_standard_rag(llm_mode="local")

    question = "李世民是谁？"
    print(f"\n用户: {question}")

    answer = rag.query(question)
    print(f"\nAI: {answer[:200]}...")


def example_2_streaming_query():
    """示例 2: 流式查询"""
    print("\n" + "=" * 60)
    print("示例 2: 流式查询")
    print("=" * 60)

    rag = create_standard_rag(llm_mode="local")

    question = "玄武门之变是什么？"
    print(f"\n用户: {question}")
    print("\nAI: ", end="", flush=True)

    for chunk in rag.stream_query(question):
        print(chunk, end="", flush=True)

    print()  # 换行


def example_3_conversation():
    """示例 3: 多轮对话"""
    print("\n" + "=" * 60)
    print("示例 3: 多轮对话")
    print("=" * 60)

    rag = create_standard_rag(
        llm_mode="local",
        session_id="test_session_1"
    )

    questions = [
        "李世民是谁？",
        "他是怎么当上皇帝的？",
        "贞观之治有哪些成就？",
    ]

    for question in questions:
        print(f"\n用户: {question}")
        answer = rag.query_with_history(question)
        print(f"AI: {answer[:150]}...")

    # 清空历史
    rag.clear_memory()
    print("\n对话历史已清空")


def example_4_standard_modules():
    """示例 4: 单独使用标准模块"""
    print("\n" + "=" * 60)
    print("示例 4: 单独使用标准模块")
    print("=" * 60)

    # 1. 获取 LLM
    from src.llm.standard_llm import get_standard_llm
    llm = get_standard_llm("local")
    print(f"\nLLM 类型: {llm.__class__.__name__}")

    # 2. 获取 Retriever
    from src.rag.standard_retriever import get_standard_retriever
    retriever = get_standard_retriever()
    print(f"Retriever 类型: {retriever.__class__.__name__}")

    # 3. 构建 Chain
    from src.rag.standard_chain import build_standard_rag_chain
    chain = build_standard_rag_chain(llm=llm, retriever=retriever)
    print(f"Chain 类型: {chain.__class__.__name__}")

    # 4. 获取 Memory
    from src.rag.standard_memory import get_standard_memory
    memory = get_standard_memory("test_session_2")
    print(f"Memory 类型: {memory.__class__.__name__}")

    print("\n所有标准模块初始化成功！")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("标准 LangChain 使用示例")
    print("=" * 60)

    # 运行示例（按需注释掉）
    try:
        # example_1_basic_query()
        # example_2_streaming_query()
        # example_3_conversation()
        example_4_standard_modules()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)
