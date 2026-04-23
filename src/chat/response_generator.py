# -*- coding: utf-8 -*-
"""
回答生成器（适配 StandardLLM）

基于检索结果和 LLM 生成最终回答。
支持注入对话历史（history messages）。
"""

import logging
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.llm.standard_llm_new import StandardLLM
from src.retrieval.faq_retriever_practice import FAQResult
from src.retrieval.doc_retriever_practice import DocResult

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """回答生成器"""

    def __init__(self, llm_mode: str = None):
        """
        初始化

        Args:
            llm_mode: LLM 模式（"local"/"api"），None 则用配置默认值
        """
        self.llm_mode = llm_mode

    def generate(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult],
        history_messages: Optional[List[BaseMessage]] = None,
    ) -> str:
        """
        生成回答

        Args:
            query: 用户查询
            faq_results: FAQ 检索结果
            doc_results: 文档检索结果
            history_messages: 对话历史消息（可选）

        Returns:
            str: 生成的回答
        """
        prompt = self.build_prompt(query, faq_results, doc_results)

        messages: List[BaseMessage] = [
            SystemMessage(
                content="你是一位专业的中国现代史研究专家。"
                        "基于提供的参考资料回答用户问题，不要编造。"
                        "如果资料不足以回答，请明确说明。"
            ),
        ]

        # 注入历史对话（如果有）
        if history_messages:
            messages.extend(history_messages)

        messages.append(HumanMessage(content=prompt))

        try:
            resp = StandardLLM.invoke(messages, mode=self.llm_mode)
            return resp.content
        except Exception as e:
            logger.error("[ResponseGenerator] LLM 调用失败: %s", e)
            return "抱歉，生成回答时出错了，请稍后再试。"

    def generate_pure_llm(self, query: str, history_messages: Optional[List[BaseMessage]] = None) -> str:
        """
        纯 LLM 生成（不走检索，用于兜底/错误恢复）

        Args:
            query: 用户查询
            history_messages: 对话历史消息（可选）

        Returns:
            str: 生成的回答
        """
        messages: List[BaseMessage] = [
            SystemMessage(
                content="你是一位专业的中国现代史研究专家。"
                        "请基于你的知识回答用户问题，保持客观中立。"
                        "如果不确定，请明确说明。"
            ),
        ]
        if history_messages:
            messages.extend(history_messages)
        messages.append(HumanMessage(content=query))

        try:
            resp = StandardLLM.invoke(messages, mode=self.llm_mode)
            return resp.content
        except Exception as e:
            logger.error("[ResponseGenerator] 纯 LLM 兜底调用失败: %s", e)
            return "抱歉，服务暂时不可用，请稍后再试。"

    def build_prompt(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult]
    ) -> str:
        """
        构建提示词

        将检索结果拼接成结构化 prompt，供 LLM 参考。
        """
        prompt_parts = ["请参考以下资料回答问题：\n"]

        if faq_results:
            prompt_parts.append("\n【相关问答】")
            for i, r in enumerate(faq_results[:3], 1):
                prompt_parts.append(f"{i}. 问题：{r.question}")
                prompt_parts.append(f"   答案：{r.answer[:200]}...")

        if doc_results:
            prompt_parts.append("\n【相关文档片段】")
            for i, r in enumerate(doc_results[:3], 1):
                prompt_parts.append(f"{i}. 来源：{r.doc_name} 第{r.doc_page}页")
                prompt_parts.append(f"   内容：{r.content[:200]}...")

        prompt_parts.append(f"\n用户问题：{query}")
        prompt_parts.append("\n请基于以上资料回答。如果资料不足以回答，请明确说明。")

        return "\n".join(prompt_parts)
