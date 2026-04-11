# -*- coding: utf-8 -*-
"""
标准流式输出实现（公司级实践）

流式与 LCEL 分离，直接调用 llm.stream()
"""

from typing import Iterator, AsyncIterator
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

from src.llm.standard_llm import get_standard_llm
from src.rag.standard_retriever import PGVectorRetriever, get_pgvector_retriever
from src.rag.standard_chain import format_docs
from src.rag.standard_memory import StandardMemory

logger = None


def get_logger():
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    return logger


def stream_rag_response(
    query: str,
    llm: BaseChatModel = None,
    retriever: PGVectorRetriever = None,
    memory: StandardMemory = None,
    system_prompt: str = None,
) -> Iterator[str]:
    """
    标准流式 RAG 响应

    公司实践：
    - 流式不经过 LCEL，直接调用 llm.stream()
    - 检索 -> 拼接 -> 提示词 -> 流式生成

    Yields:
        str: 每次生成的文本块
    """
    llm = llm or get_standard_llm()
    retriever = retriever or get_pgvector_retriever()

    # 1. 检索文档
    docs = retriever.invoke(query)
    context = format_docs(docs)

    # 2. 获取对话历史（如果有）
    history_str = ""
    if memory:
        history_str = memory.get_history_string()

    # 3. 构建系统提示词
    if system_prompt is None:
        system_prompt = """你是一位专业的历史研究专家。
请基于提供的参考资料回答用户问题。

要求：
1. 回答必须基于提供的资料，不要编造
2. 保持客观中立的历史态度
3. 如果资料不足以回答，请明确说明"根据现有资料无法确定"
4. 适当引用资料来源"""

    full_system_prompt = f"""{system_prompt}

参考资料：
{context}"""

    if history_str:
        full_system_prompt += f"\n\n对话历史：\n{history_str}"

    # 4. 构建消息
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=full_system_prompt),
        HumanMessage(content=query)
    ]

    # 5. 流式生成（标准做法：直接调用 llm.stream）
    full_response = ""
    for chunk in llm.stream(messages):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        full_response += content
        yield content

    # 6. 保存历史（如果有 memory）
    if memory:
        memory.add_user_message(query)
        memory.add_ai_message(full_response)

    get_logger().info(f"流式响应完成: {len(full_response)} 字符")


async def astream_rag_response(
    query: str,
    llm: BaseChatModel = None,
    retriever: PGVectorRetriever = None,
    memory: StandardMemory = None,
) -> AsyncIterator[str]:
    """
    异步流式 RAG 响应
    """
    llm = llm or get_standard_llm()
    retriever = retriever or get_pgvector_retriever()

    # 1. 检索文档
    docs = await retriever.ainvoke(query)
    context = format_docs(docs)

    # 2. 获取历史
    history_str = ""
    if memory:
        history_str = memory.get_history_string()

    # 3. 构建消息
    from langchain_core.messages import HumanMessage, SystemMessage

    system_content = f"""你是一位专业的历史研究专家。
请基于提供的参考资料回答用户问题。

参考资料：
{context}"""

    if history_str:
        system_content += f"\n\n对话历史：\n{history_str}"

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=query)
    ]

    # 4. 异步流式生成
    full_response = ""
    async for chunk in llm.astream(messages):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        full_response += content
        yield content

    # 5. 保存历史
    if memory:
        memory.add_user_message(query)
        memory.add_ai_message(full_response)
