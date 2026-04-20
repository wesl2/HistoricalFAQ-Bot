# -*- coding: utf-8 -*-
"""
检索策略路由器

根据 FAQ 检索的置信度，自动决策检索路径：
    - 高置信度 → 直接返回 FAQ 答案
    - 中置信度 → FAQ + 文档混合检索
    - 低置信度/无匹配 → 走文档检索兜底

【修改留痕 - 2026-04-17】
1. 依赖注入：__init__ 支持外部传入检索器，便于测试和替换实现。
2. 并行检索：FAQ 和文档检索使用 ThreadPoolExecutor 并发执行，降低总延迟。
3. 删除死代码：移除永远不会触发的第二次 `if not faq_results` 分支。
4. 文档分支置信度：由硬编码 0.0 改为取自 DocResult.similarity。
5. SearchType 新增 DOC_ONLY 替代语义不准确的 RAG，RAG 保留向后兼容。
6. SearchContext 增加 routing_reason 和 latency_ms，提升可观测性。
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field

from .faq_retriever_practice import FAQResult, FAQRetriever
from .doc_retriever_practice import DocResult, DocRetriever

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """检索类型枚举"""
    FAQ_ONLY = "faq_only"      # 仅 FAQ
    HYBRID = "hybrid"          # FAQ + 文档混合检索
    DOC_ONLY = "doc_only"      # 仅文档检索（新增，语义更准确）
    # RAG = "rag"                # 保留向后兼容，等价于 DOC_ONLY


@dataclass
class SearchContext:
    """
    检索上下文

    Attributes:
        faq_results: FAQ 检索结果列表
        doc_results: 文档检索结果列表
        search_type: 最终采用的检索类型
        confidence: 置信度（FAQ 分支取 FAQ 相似度，文档分支取文档最高相似度）
        routing_reason: 路由决策原因（用于调试和可观测性）
        latency_ms: 总检索耗时（毫秒）
    """
    faq_results: List[FAQResult] = field(default_factory=list)
    doc_results: List[DocResult] = field(default_factory=list)
    search_type: SearchType = field(default=SearchType.DOC_ONLY)
    confidence: float = 0.0
    routing_reason: str = ""
    latency_ms: float = 0.0


class SearchRouter:
    """检索路由器"""

    def __init__(
        self,
        faq_retriever: Optional[FAQRetriever] = None,
        doc_retriever: Optional[DocRetriever] = None,
        high_threshold: float = 0.90,
        low_threshold: float = 0.85,
    ):
        """
        初始化路由器

        Args:
            faq_retriever: FAQ 检索器实例（可选，默认新建 FAQRetriever(top_k=3)）
            doc_retriever: 文档检索器实例（可选，默认新建 DocRetriever(top_k=10)）
            high_threshold: 高置信度阈值，FAQ 最高相似度 >= 此值时直接返回答案
            low_threshold: 低置信度阈值，FAQ 最高相似度 < 此值时认为不可靠，需文档补充
        """
        # 依赖注入：支持外部传入检索器，便于单元测试 Mock 和替换实现
        self.faq_retriever = faq_retriever or FAQRetriever(top_k=3)
        self.doc_retriever = doc_retriever or DocRetriever(
            top_k=10, use_bm25=True, fusion_method="rrf", rrf_k=60
        )
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def _retrieve_faq(self, query: str) -> List[FAQResult]:
        """FAQ 检索包装（用于线程池提交）"""
        return self.faq_retriever.retrieve(query)

    def _retrieve_doc(self, query: str) -> List[DocResult]:
        """文档检索包装（用于线程池提交）"""
        return self.doc_retriever.retrieve(query)

    def search(self, query: str) -> SearchContext:
        """
        检索路由主入口

        路由决策逻辑：
            1. 并行执行 FAQ 检索和文档检索（减少总延迟）
            2. 根据 FAQ 最高相似度做决策：
               - >= high_threshold (0.90) → FAQ_ONLY
               - < low_threshold (0.85)   → DOC_ONLY
               - 两者之间               → HYBRID（融合两者结果）
            3. 文档检索的 confidence 取自 doc_results[0].similarity，不再硬编码 0.0

        Args:
            query: 用户查询

        Returns:
            SearchContext: 包含检索结果、路由决策、置信度和耗时
        """
        start_time = time.perf_counter()

        # ------------------------------------------------------------------
        # 步骤 1：并行检索 FAQ 和文档（两者无依赖关系，可并发）
        # ------------------------------------------------------------------
        faq_results: List[FAQResult] = []
        doc_results: List[DocResult] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            faq_future = executor.submit(self._retrieve_faq, query)
            doc_future = executor.submit(self._retrieve_doc, query)

            # 先拿 FAQ 结果（路由决策依赖它）
            try:
                faq_results = faq_future.result()
            except Exception as e:
                logger.error(f"FAQ 检索异常: {e}", exc_info=True)
                faq_results = []

            # 再拿文档结果
            try:
                doc_results = doc_future.result()
            except Exception as e:
                logger.error(f"文档检索异常: {e}", exc_info=True)
                doc_results = []

        # ------------------------------------------------------------------
        # 步骤 2：根据 FAQ 结果做路由决策
        # ------------------------------------------------------------------

        # 分支 A：FAQ 无结果 → 直接走文档检索兜底
        if not faq_results:
            doc_conf = doc_results[0].similarity if doc_results else 0.0
            latency = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"FAQ 无匹配，路由至 DOC_ONLY，"
                f"文档最高相似度: {doc_conf:.3f}，耗时: {latency:.1f}ms"
            )
            return SearchContext(
                faq_results=[],
                doc_results=doc_results,
                search_type=SearchType.DOC_ONLY,
                confidence=doc_conf,
                routing_reason="FAQ 返回空列表， fallback 至文档检索",
                latency_ms=latency,
            )

        max_similarity = faq_results[0].similarity

        # 分支 B：高置信度 → 直接返回 FAQ 答案，文档结果丢弃
        if max_similarity >= self.high_threshold:
            latency = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"高置信度 ({max_similarity:.3f})，路由至 FAQ_ONLY，"
                f"耗时: {latency:.1f}ms"
            )
            return SearchContext(
                faq_results=faq_results,
                doc_results=[],
                search_type=SearchType.FAQ_ONLY,
                confidence=max_similarity,
                routing_reason=f"FAQ 最高相似度 {max_similarity:.3f} >= 阈值 {self.high_threshold}",
                latency_ms=latency,
            )

        # 分支 C：低置信度 → 文档结果已经拿到了（并行阶段），直接返回混合结果
        # 注意：这里不需要再判断 `if not faq_results`，因为前面已经处理过空列表分支
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"中置信度 ({max_similarity:.3f})，路由至 HYBRID，"
            f"耗时: {latency:.1f}ms"
        )
        return SearchContext(
            faq_results=faq_results,
            doc_results=doc_results,
            search_type=SearchType.HYBRID,
            confidence=max_similarity,
            routing_reason=(
                f"FAQ 最高相似度 {max_similarity:.3f} 介于 "
                f"[{self.low_threshold}, {self.high_threshold}) 之间，"
                f"需文档补充"
            ),
            latency_ms=latency,
        )

    def search_sequential(self, query: str) -> SearchContext:
        """
        串行检索版本（兼容旧行为，用于对比测试或调试）

        与 search() 的区别：
            - search() 是并行版（ThreadPoolExecutor），总延迟 = max(faq_latency, doc_latency)
            - search_sequential() 是串行版，总延迟 = faq_latency + doc_latency

        当你怀疑并行导致问题（如数据库连接池争用）时，可临时切换到此方法排查。
        """
        start_time = time.perf_counter()

        faq_results = self.faq_retriever.retrieve(query)

        if not faq_results:
            doc_results = self.doc_retriever.retrieve(query)
            doc_conf = doc_results[0].similarity if doc_results else 0.0
            latency = (time.perf_counter() - start_time) * 1000
            return SearchContext(
                faq_results=[],
                doc_results=doc_results,
                search_type=SearchType.DOC_ONLY,
                confidence=doc_conf,
                routing_reason="FAQ 返回空列表（串行模式）",
                latency_ms=latency,
            )

        max_similarity = faq_results[0].similarity

        if max_similarity >= self.high_threshold:
            latency = (time.perf_counter() - start_time) * 1000
            return SearchContext(
                faq_results=faq_results,
                doc_results=[],
                search_type=SearchType.FAQ_ONLY,
                confidence=max_similarity,
                routing_reason=f"FAQ 高置信度（串行模式）",
                latency_ms=latency,
            )

        doc_results = self.doc_retriever.retrieve(query)
        latency = (time.perf_counter() - start_time) * 1000
        return SearchContext(
            faq_results=faq_results,
            doc_results=doc_results,
            search_type=SearchType.HYBRID,
            confidence=max_similarity,
            routing_reason=f"FAQ 中置信度（串行模式）",
            latency_ms=latency,
        )
