# -*- coding: utf-8 -*-
"""
回答生成器

基于检索结果和 LLM 生成最终回答
"""

import logging
from typing import List, Dict
from src.llm.base_llm import BaseLLM
from src.retrieval.faq_retriever import FAQResult
from src.retrieval.doc_retriever import DocResult

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """回答生成器"""
    
    def __init__(self, llm: BaseLLM):
        """
        初始化
        
        Args:
            llm: LLM 实例
        """
        self.llm = llm
    
    def generate(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult]
    ) -> str:
        """
        生成回答
        
        Args:
            query: 用户查询
            faq_results: FAQ 检索结果
            doc_results: 文档检索结果
            
        Returns:
            str: 生成的回答
        """
        # 构建提示词
        prompt = self._build_prompt(query, faq_results, doc_results)
        
        # 调用 LLM
        messages = [
            {"role": "system", "content": "你是一个专业的中国历史研究助手。基于提供的参考资料回答用户问题。"},
            {"role": "user", "content": prompt}
        ]
        
        return self.llm.chat(messages)
    
    def _build_prompt(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult]
    ) -> str:
        """构建提示词"""
        prompt_parts = ["请参考以下资料回答问题：\n"]
        
        # 添加 FAQ 参考
        if faq_results:
            prompt_parts.append("\n【相关问答】")
            for i, r in enumerate(faq_results[:3], 1):
                prompt_parts.append(f"{i}. 问题：{r.question}")
                prompt_parts.append(f"   答案：{r.answer[:200]}...")
        
        # 添加文档参考
        if doc_results:
            prompt_parts.append("\n【相关文档片段】")
            for i, r in enumerate(doc_results[:3], 1):
                prompt_parts.append(f"{i}. 来源：{r.doc_name} 第{r.doc_page}页")
                prompt_parts.append(f"   内容：{r.content[:200]}...")
        
        prompt_parts.append(f"\n用户问题：{query}")
        prompt_parts.append("\n请基于以上资料回答。如果资料不足以回答，请明确说明。")
        
        return "\n".join(prompt_parts)
