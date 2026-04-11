# -*- coding: utf-8 -*-
"""
标准 LangChain LCEL Chain 构建（公司级实践）

使用 LCEL 组合式写法，不使用 ConversationalRetrievalChain
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from operator import itemgetter
from typing import List, Dict, Any, Optional

from src.llm.standard_llm import get_standard_llm
from src.rag.standard_retriever import get_pgvector_retriever, PGVectorRetriever
from config.model_config import LANGCHAIN_CONFIG

logger = None


def get_logger():
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    return logger


def format_docs(docs: List[Document]) -> str:
    """格式化检索结果为上下文字符串"""
    if not docs:
        return "无相关参考资料"

    return "\n\n".join(
        f"来源：{doc.metadata.get('source', doc.metadata.get('doc_name', '未知'))}\n"
        f"内容：{doc.page_content}"
        for doc in docs
    )


def load_prompt_template(prompt_type: str = "rag") -> ChatPromptTemplate:
    """
    加载提示词模板（从文件或默认）
    """
    from pathlib import Path

    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    prompt_file = prompts_dir / f"{prompt_type}_template.txt"

    if prompt_file.exists():
        template = prompt_file.read_text(encoding="utf-8")
        get_logger().info(f"加载提示词模板：{prompt_file}")
        return ChatPromptTemplate.from_template(template)
    else:
        get_logger().warning(f"提示词模板不存在：{prompt_file}，使用默认")
        return _get_default_prompt()


def _get_default_prompt() -> ChatPromptTemplate:
    """默认 RAG 提示词"""
    return ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的历史研究专家。
请基于提供的参考资料回答用户问题。

要求：
1. 回答必须基于提供的资料，不要编造
2. 保持客观中立的历史态度
3. 如果资料不足以回答，请明确说明"根据现有资料无法确定"
4. 适当引用资料来源

参考资料：
{context}"""),
        ("human", "{question}")
    ])


def build_standard_rag_chain(
    llm: BaseChatModel = None,
    retriever: PGVectorRetriever = None,
) -> Any:
    """
    标准 RAG Chain（LCEL 写法）

    流程：question -> retriever -> format_docs -> prompt -> llm -> parser -> answer

    公司实践：使用 LCEL 组合式，清晰可控
    """
    llm = llm or get_standard_llm()
    retriever = retriever or get_pgvector_retriever()

    # 构建提示词
    prompt = _get_default_prompt()

    # LCEL Chain（标准写法）
    rag_chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    get_logger().info("标准 RAG Chain 构建完成")
    return rag_chain


def build_conversational_rag_chain(
    llm: BaseChatModel = None,
    retriever: PGVectorRetriever = None,
    use_history_rewrite: bool = True,
) -> Any:
    """
    带对话历史的 RAG Chain

    公司实践：不使用 ConversationalRetrievalChain，自己组合

    流程：
    1. 问题重写（考虑历史）-> rewritten_question
    2. 检索文档 -> context
    3. 生成回答 -> answer
    """
    llm = llm or get_standard_llm()
    retriever = retriever or get_pgvector_retriever()

    if use_history_rewrite:
        # 1. 问题重写 Prompt
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个问题重写助手。
基于对话历史，将用户问题重写为独立的查询，以便检索。
如果问题已经独立，不需要重写。

示例：
历史：用户问"李世民是谁？"，AI 回答"..."
新问题："他做了什么？"
重写："李世民做了什么？"

只输出重写后的问题，不要其他内容。"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        # 2. RAG Prompt
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的历史研究专家。
请基于提供的参考资料回答用户问题。

参考资料：
{context}"""),
            ("human", "{rewritten_question}")
        ])

        # 3. 构建 Chain
        rewrite_chain = (
            rewrite_prompt
            | llm
            | StrOutputParser()
        )

        # 修复：使用 itemgetter 确保 retriever 收到 rewritten_question 字符串
        rag_chain = (
            RunnableParallel({
                "rewritten_question": rewrite_chain,
                "context": itemgetter("rewritten_question") | retriever | format_docs,
                "history": RunnablePassthrough()
            })
            | rag_prompt
            | llm
            | StrOutputParser()
        )
    else:
        # 简单版本：不重写，直接拼接历史到问题
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的历史研究专家。
请基于提供的参考资料和对话历史回答用户问题。

参考资料：
{context}

对话历史：
{history}"""),
            ("human", "{question}")
        ])

        rag_chain = (
            RunnableParallel({
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "history": RunnablePassthrough()
            })
            | prompt
            | llm
            | StrOutputParser()
        )

    get_logger().info("带对话历史的 RAG Chain 构建完成")
    return rag_chain


def build_agent_chain(
    llm: BaseChatModel = None,
    tools: List = None,
    agent_type: str = "zero-shot-react-description",
) -> Any:
    """
    Agent Chain（工具调用）

    公司实践：使用 LangChain Agent，支持工具调用
    """
    try:
        from langchain.agents import initialize_agent, AgentType
    except ImportError:
        raise NotImplementedError(
            "Agent 功能在 LangChain 1.x 中已重构，"
            "请使用 langchain_core 的 create_react_agent 等 API"
        )

    llm = llm or get_standard_llm()

    agent = initialize_agent(
        tools=tools or [],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    get_logger().info(f"Agent Chain 构建完成：{agent_type}")
    return agent
