# -*- coding: utf-8 -*-
"""检索策略路由器"""

import logging
from enum import Enum
from typing import List, NamedTuple, Optional
from dataclasses import dataclass, field
from .faq_retriever import FAQResult, FAQRetriever
from .doc_retriever import DocResult, DocRetriever

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """检索类型枚举"""
    FAQ_ONLY = "faq_only"      # 仅 FAQ
    HYBRID = "hybrid"          # 混合检索
    RAG = "rag"                # 纯文档检索


@dataclass
class SearchContext:
    """检索上下文"""
    faq_results: List[FAQResult] = field(default_factory=list)
    doc_results: List[DocResult] = field(default_factory=list)
    search_type: SearchType = SearchType.RAG
    confidence: float = 0.0


class SearchRouter:
    """检索路由器"""
    
    def __init__(
        self,
        high_threshold: float = 0.90,
        low_threshold: float = 0.85
    ):
        """
        初始化路由器
        
        Args:
            high_threshold: 高置信度阈值 (> 直接返回 FAQ)
            low_threshold: 低置信度阈值 (< 转 RAG)
        """
        self.faq_retriever = FAQRetriever(top_k=3)
        self.doc_retriever = DocRetriever(top_k=10)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
    
    def search(self, query: str) -> SearchContext:
        """
        检索路由主入口
        
        Args:
            query: 用户查询
        
        Returns:
            SearchContext: 包含检索结果和路由决策
        """
        # 1. FAQ 检索
        faq_results = self.faq_retriever.retrieve(query)
        
        # 2. 根据 FAQ 置信度决策
        if not faq_results:
            # 无 FAQ 匹配，直接走 RAG
            doc_results = self.doc_retriever.retrieve(query)
            return SearchContext(
                faq_results=[],
                doc_results=doc_results,
                search_type=SearchType.RAG,
                confidence=0.0
            )
        
        # 获取最高相似度
        max_similarity = faq_results[0].similarity
        
        # 3. 高置信度：直接返回 FAQ 答案
        if max_similarity >= self.high_threshold:
            logger.info(f"高置信度 ({max_similarity:.3f})，使用 FAQ_ONLY 模式")
            return SearchContext(
                faq_results=faq_results,
                doc_results=[],
                search_type=SearchType.FAQ_ONLY,
                confidence=max_similarity
            )
        
        # 4. 低置信度或无匹配：检索文档
        doc_results = self.doc_retriever.retrieve(query)
        
        if not faq_results:
            # 纯 RAG
            return SearchContext(
                faq_results=[],
                doc_results=doc_results,
                search_type=SearchType.RAG,
                confidence=0.0
            )
        
        # 5. 融合模式
        logger.info(f"中置信度 ({max_similarity:.3f})，使用 HYBRID 模式")
        return SearchContext(
            faq_results=faq_results,
            doc_results=doc_results,
            search_type=SearchType.HYBRID,
            confidence=max_similarity
        )
