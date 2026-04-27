# -*- coding: utf-8 -*-
"""
回答生成器（适配 StandardLLM）

基于检索结果和 LLM 生成最终回答。
支持注入对话历史（history messages）。
"""

import logging
from typing import List, Optional, Generator, AsyncGenerator, Dict

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
        self._system_prompt = (
            "你是一位专业的中国历史的研究专家。"
            "基于提供的参考资料回答用户问题，不要编造。"
            "如果资料不足以回答，请明确说明。"
        )
        self._system_prompt_pure = (
            "你是一位专业的中国历史的研究专家。"
            "请基于你的知识回答用户问题，保持客观中立。"
            "如果不确定，请明确说明。"
        )
        # Prompt 组装配置
        self.max_faq_in_prompt = 3
        self.max_doc_in_prompt = 5
        from langchain_text_splitters import TokenTextSplitter
        self._text_splitter = TokenTextSplitter(
            model_name="gpt-3.5-turbo",
            chunk_size=200,
            chunk_overlap=0,
        )
        self._last_source_map = {}

    def _build_messages(
        self,
        prompt: str,
        history_messages: Optional[List[BaseMessage]] = None,
        pure: bool = False,
    ) -> List[BaseMessage]:
        """构建消息列表（同步/异步共用）"""
        system = self._system_prompt_pure if pure else self._system_prompt
        messages: List[BaseMessage] = [SystemMessage(content=system)]
        if history_messages:
            messages.extend(history_messages)
        messages.append(HumanMessage(content=prompt))
        return messages

    def generate(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult],
        history_messages: Optional[List[BaseMessage]] = None,
    ) -> str:
        """
        生成回答（同步）

        Args:
            query: 用户查询
            faq_results: FAQ 检索结果
            doc_results: 文档检索结果
            history_messages: 对话历史消息（可选）

        Returns:
            str: 生成的回答
        """
        prompt = self.build_prompt(query, faq_results, doc_results)
        messages = self._build_messages(prompt, history_messages)
        try:
            resp = StandardLLM.invoke(messages, mode=self.llm_mode)
            return resp.content
        except Exception as e:
            logger.error("[ResponseGenerator] LLM 调用失败: %s", e)
            return "抱歉，生成回答时出错了，请稍后再试。"

    async def agenerate(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult],
        history_messages: Optional[List[BaseMessage]] = None,
    ) -> str:
        """
        生成回答（异步）

        Args:
            query: 用户查询
            faq_results: FAQ 检索结果
            doc_results: 文档检索结果
            history_messages: 对话历史消息（可选）

        Returns:
            str: 生成的回答
        """
        prompt = self.build_prompt(query, faq_results, doc_results)
        messages = self._build_messages(prompt, history_messages)
        try:
            resp = await StandardLLM.ainvoke(messages, mode=self.llm_mode)
            return resp.content
        except Exception as e:
            logger.error("[ResponseGenerator] LLM 异步调用失败: %s", e)
            return "抱歉，生成回答时出错了，请稍后再试。"

    def generate_stream(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult],
        history_messages: Optional[List[BaseMessage]] = None,
    ) -> Generator[str, None, None]:
        """
        流式生成回答（同步）

        Yields:
            生成的文本片段
        """
        prompt = self.build_prompt(query, faq_results, doc_results)
        messages = self._build_messages(prompt, history_messages)
        for chunk in StandardLLM.stream(messages, mode=self.llm_mode):
            yield chunk.content

    async def agenerate_stream(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult],
        history_messages: Optional[List[BaseMessage]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式生成回答（异步）

        Yields:
            生成的文本片段
        """
        prompt = self.build_prompt(query, faq_results, doc_results)
        messages = self._build_messages(prompt, history_messages)
        async for chunk in StandardLLM.astream(messages, mode=self.llm_mode):
            yield chunk.content

    def generate_pure_llm(self, query: str, history_messages: Optional[List[BaseMessage]] = None) -> str:
        """
        纯 LLM 生成（不走检索，用于兜底/错误恢复）

        Args:
            query: 用户查询
            history_messages: 对话历史消息（可选）

        Returns:
            str: 生成的回答
        """
        messages = self._build_messages(query, history_messages, pure=True)
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

        【2024-04-21 修改说明】
        1. 将硬编码的 [:3] 改为可配置参数（max_faq_in_prompt / max_doc_in_prompt）
        2. 将 [:200] 字符粗暴截断改为 TokenTextSplitter 精确按 token 截断
        3. 【P0: 引用溯源】给每个来源分配编号 [1]、[2]...，要求 LLM 在回答中标注引用
        """
        # 【P0: 引用溯源】重置来源映射表
        self._last_source_map = {}
        idx = 1

        prompt_parts = ["请参考以下资料回答问题，并在相关陈述后标注引用来源（格式如 [1]、[2]）。\n"]

        if faq_results:
            prompt_parts.append("\n【相关问答】")
            for r in faq_results[:self.max_faq_in_prompt]:
                answer_snippet = (
                    self._text_splitter.split_text(r.answer)[0] if r.answer else ""
                )
                prompt_parts.append(f"[{idx}] 问题：{r.question}")
                prompt_parts.append(f"    答案：{answer_snippet}...")
                self._last_source_map[str(idx)] = {
                    "type": "faq",
                    "question": r.question,
                    "answer": r.answer,
                }
                idx += 1

        if doc_results:
            prompt_parts.append("\n【相关文档片段】")
            for r in doc_results[:self.max_doc_in_prompt]:
                content_snippet = (
                    self._text_splitter.split_text(r.content)[0] if r.content else ""
                )
                prompt_parts.append(f"[{idx}] 来源：{r.doc_name} 第{r.doc_page}页")
                prompt_parts.append(f"    内容：{content_snippet}...")
                self._last_source_map[str(idx)] = {
                    "type": "doc",
                    "doc_name": r.doc_name,
                    "page": r.doc_page,
                    "content": r.content,
                }
                idx += 1

        prompt_parts.append(f"\n用户问题：{query}")
        prompt_parts.append("\n请基于以上资料回答，并标注引用来源。如果资料不足以回答，请明确说明。")

        return "\n".join(prompt_parts)

    def extract_citations(self, answer: str) -> List[Dict]:
        """
        【P0: 引用溯源】从 LLM 回答中提取 [1]、[2] 等引用标记，关联到原始来源。

        Args:
            answer: LLM 生成的回答文本

        Returns:
            [{"id": "1", "type": "faq"/"doc", ...}, ...]
        """
        import re
        citations = []
        seen = set()
        for match in re.finditer(r'\[(\d+)\]', answer):
            cid = match.group(1)
            if cid not in seen and cid in self._last_source_map:
                seen.add(cid)
                citations.append({"id": cid, **self._last_source_map[cid]})
        return citations
