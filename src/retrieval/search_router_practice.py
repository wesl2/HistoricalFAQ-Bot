#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索策略路由器（重构版）

目标：实现一个智能检索路由器，根据 FAQ 检索的置信度自动决策：
      - 高置信度 → 直接返回 FAQ 答案
      - 中置信度 → FAQ + 文档混合检索
      - 低置信度/无匹配 → 走 RAG 文档检索

【核心概念】
1. 路由策略（Routing Strategy）：根据检索质量动态选择检索路径
2. 置信度阈值（Confidence Threshold）：用数值边界划分决策区间
3. 多路召回（Multi-path Retrieval）：同时调用多个检索器，根据结果融合
"""

# =============================================================================
# 第一部分：导入模块
# =============================================================================
# TODO 1.1 ⭐ 导入 Python 标准库模块
# 提示：
#   - logging: 日志记录（替代 print，方便控制输出级别）
#   - enum.Enum: 枚举类型（定义检索类型常量）
#   - typing: 类型提示（List, Optional 等，提升代码可读性）
#   - dataclasses: 数据类（dataclass, field，简化数据容器定义）

import asyncio
import logging
from enum import Enum
from typing import List, NamedTuple, Optional
from dataclasses import dataclass, field
import time
from concurrent.futures import ThreadPoolExecutor
# TODO 1.2 ⭐ 从同级目录的 practice 模块导入检索器
# 提示：
#   from .faq_retriever_practice import FAQResult, FAQRetriever
#   from .doc_retriever_practice import DocResult, DocRetriever
# 思考：为什么用相对导入（.xxx）？因为 search_router 和检索器在同一包内

from src.retrieval.faq_retriever_practice import FAQResult, FAQRetriever
from src.retrieval.doc_retriever_practice import DocResult, DocRetriever


# =============================================================================
# 第二部分：日志配置
# =============================================================================
# TODO 2.1 ⭐ 获取以当前模块名命名的 logger
# 提示：logger = logging.getLogger(__name__)
# 注意：这里不需要 logging.basicConfig，因为项目入口已经配置过了
logger = logging.getLogger(__name__)

# =============================================================================
# 第三部分：检索类型枚举
# =============================================================================
# TODO 3.1 ⭐ 定义 SearchType 枚举类
# 提示：
#   class SearchType(Enum):
#       """检索类型枚举"""
#       FAQ_ONLY = "faq_only"
#       HYBRID = "hybrid"
#       RAG = "rag"
#
# 思考：为什么用 Enum 而不是普通字符串常量？
#   1. 类型安全：防止拼写错误在运行时才暴露
#   2. 可枚举性：可以遍历所有支持的检索类型
#   3. IDE 补全：编辑器能提示可用的选项

class SearchType(Enum):
    FAQ_ONLY = "faq_only"      # 仅 FAQ
    DOC_ONLY = "doc_only"      # 仅文档
    HYBRID = "hybrid"          # FAQ + 文档混合检索


# =============================================================================
# 第四部分：检索上下文数据类
# =============================================================================
# TODO 4.1 ⭐ 定义 SearchContext 数据类
# 提示：
#   @dataclass
#   class SearchContext:
#       """检索上下文"""
#       faq_results: List[FAQResult] = field(default_factory=list)
#       doc_results: List[DocResult] = field(default_factory=list)
#       search_type: SearchType = SearchType.RAG
#       confidence: float = 0.0
#
# 关键知识点：
#   - @dataclass: Python 3.7+ 语法糖，自动生成 __init__、__repr__ 等方法
#   - field(default_factory=list): 可变默认值必须用 factory，否则所有实例共享同一个 list
#   - 为什么需要这个类？它是检索结果的"统一出口"，让调用方不用关心内部路由逻辑
@dataclass
class SearchContext:
    """检索上下文"""
    faq_results: List[FAQResult] = field(default_factory=list)
    doc_results: List[DocResult] = field(default_factory=list)
    search_type: SearchType = SearchType.HYBRID
    confidence: float = 0.0
    routing_reason: str = ""
    latency_ms: float = 0.0
# =============================================================================
# 第五部分：检索路由器（核心类）
# =============================================================================
class SearchRouter:
    """
    检索路由器
    
    职责：
    1. 持有多个检索器实例（FAQ + 文档）
    2. 根据 FAQ 检索的置信度，动态选择检索策略
    3. 返回统一的 SearchContext 结果
    
    路由决策逻辑：
    ┌─────────────────────────────────────────┐
│  用户输入 query                            │
    │                    ↓                    │
│  1. 先查 FAQ                              │
    │                    ↓                    │
│  2. 判断 FAQ 结果？                        │
    │         ┌──────┴──────┐               │
│         无结果      有结果                │
    │           ↓           ↓               │
│       直接走 稠密检索   看最高相似度           │
    │                     ↓                 │
│              ┌─────┼─────┐              │
│           ≥0.90   0.85~0.90   <0.85     │
│             ↓        ↓         ↓        │
│        FAQ_ONLY   HYBRID      稠密检索   │
└─────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        faq_retriever: FAQRetriever,
        doc_retriever: DocRetriever,
        high_threshold: float = 0.82,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        初始化路由器
        
        Args:
            faq_retriever: FAQ 检索器实例
            doc_retriever: 文档检索器实例
            high_threshold: 高置信度阈值。FAQ 最高相似度 ≥ 此值时，
                           认为 FAQ 足以回答，直接返回 FAQ 结果。
            executor: 外部线程池（可选）。生产环境建议传入全局线程池，
                      避免每次检索都新建线程池。
        """
        self.faq_retriever = faq_retriever 
        self.doc_retriever = doc_retriever
        self.high_threshold = high_threshold
        self._executor = executor or ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="search_router"
        )
        self._own_executor = executor is None
    
    def _retrieve_faq(self, query: str) -> List[FAQResult]:
        """FAQ 检索包装（用于线程池提交）"""
        return self.faq_retriever.retrieve(query)

    def _retrieve_doc(self, query: str) -> List[DocResult]:
        """文档检索包装（用于线程池提交）"""
        return self.doc_retriever.retrieve(query)

    
    def _build_context(
        self,
        faq_results: List[FAQResult],
        doc_results: List[DocResult],
        latency_ms: float,
    ) -> SearchContext:
        """根据 FAQ + Doc 结果构建路由决策后的 SearchContext"""
        # 分支 A：FAQ 无结果 → 直接走文档检索兜底
        if not faq_results:
            doc_conf = doc_results[0].similarity if doc_results else 0.0
            logger.info(
                "FAQ 无匹配，路由至 DOC_ONLY, "
                "文档最高相似度: %.3f，耗时: %.1fms",
                doc_conf, latency_ms,
            )
            return SearchContext(
                faq_results=[],
                doc_results=doc_results,
                search_type=SearchType.DOC_ONLY,
                confidence=doc_conf,
                routing_reason="FAQ 返回空列表，fallback 至文档检索",
                latency_ms=latency_ms,
            )

        # 按相似度降序排列
        faq_results_sorted = sorted(faq_results, key=lambda x: x.similarity, reverse=True)
        max_similarity = faq_results_sorted[0].similarity

        # 分支 B：高置信度 → 直接返回 FAQ 答案
        if max_similarity >= self.high_threshold:
            return SearchContext(
                faq_results=faq_results_sorted,
                doc_results=[],
                search_type=SearchType.FAQ_ONLY,
                confidence=max_similarity,
                routing_reason=f"FAQ 最高相似度 {max_similarity:.3f} >= 阈值 {self.high_threshold}",
                latency_ms=latency_ms,
            )

        # 分支 C：中低置信度 → 返回混合结果
        return SearchContext(
            faq_results=faq_results_sorted,
            doc_results=doc_results,
            search_type=SearchType.HYBRID,
            confidence=max_similarity,
            routing_reason=(
                f"FAQ 最高相似度 {max_similarity:.3f} 低于阈值 {self.high_threshold}，"
                f"需文档补充"
            ),
            latency_ms=latency_ms,
        )

    def search(self, query: str) -> SearchContext:
        """同步检索（复用实例级线程池，避免每次新建）"""
        start_time = time.perf_counter()

        faq_future = self._executor.submit(self._retrieve_faq, query)
        doc_future = self._executor.submit(self._retrieve_doc, query)

        try:
            faq_results = faq_future.result()
        except Exception as e:
            logger.error("FAQ 检索失败: %s", e, exc_info=True)
            faq_results = []

        try:
            doc_results = doc_future.result()
        except Exception as e:
            logger.error("文档检索异常: %s", e, exc_info=True)
            doc_results = []

        latency = (time.perf_counter() - start_time) * 1000
        return self._build_context(faq_results, doc_results, latency)

    async def asearch(self, query: str) -> SearchContext:
        """异步检索（用 asyncio.to_thread 跑同步检索器，gather 并发）"""
        start_time = time.perf_counter()

        faq_task = asyncio.create_task(
            asyncio.to_thread(self._retrieve_faq, query)
        )
        doc_task = asyncio.create_task(
            asyncio.to_thread(self._retrieve_doc, query)
        )

        faq_results, doc_results = await asyncio.gather(
            faq_task, doc_task, return_exceptions=True
        )

        if isinstance(faq_results, Exception):
            logger.error("FAQ 检索失败: %s", faq_results, exc_info=True)
            faq_results = []
        if isinstance(doc_results, Exception):
            logger.error("文档检索异常: %s", doc_results, exc_info=True)
            doc_results = []

        latency = (time.perf_counter() - start_time) * 1000
        return self._build_context(faq_results, doc_results, latency)
    # -------------------------------------------------------------------------
    # 扩展练习（可选，⭐⭐⭐ 难度）
    # -------------------------------------------------------------------------
    def search_with_rerank(self, query: str) -> SearchContext:
        """
        适用Reranker模型进行精排
        
        Args:
            query: 用户查询字符串
        
        Returns:
            SearchContext: 包含融合排序后的结果
        """
        pass


# =============================================================================
# 第六部分：测试与验证
# =============================================================================
def test_search_router(query: str):
    """
    测试检索路由器
    
    TODO 6.1 ⭐⭐ 编写测试用例
    
    提示：在脚本底部加上测试代码：
    
    if __name__ == "__main__":
        router = SearchRouter(high_threshold=0.90, low_threshold=0.85)
        
        # 测试查询 1：高置信度 FAQ 问题
        result = router.search("王洪文是谁？")
        print(f"检索类型: {result.search_type.value}")
        print(f"置信度: {result.confidence:.3f}")
        print(f"FAQ 结果数: {len(result.faq_results)}")
        print(f"文档结果数: {len(result.doc_results)}")
        
        # 测试查询 2：可能需要 RAG 的问题
        result2 = router.search("请详细介绍一下玄武门之变的经过")
        print(f"\n检索类型: {result2.search_type.value}")
        print(f"置信度: {result2.confidence:.3f}")
    
    注意：测试前确保数据库里已经有 FAQ 和文档数据。
    """
    
    faq_retriever = FAQRetriever(top_k=3)
    doc_retriever = DocRetriever(top_k=10, use_bm25=True, fusion_method="rrf", rrf_k=60)
    router = SearchRouter(faq_retriever=faq_retriever, doc_retriever=doc_retriever)
    result = router.search(query)
    print(f"检索类型: {result.search_type.value}")
    print(f"置信度: {result.confidence:.3f}")
    print(f"FAQ 结果数: {len(result.faq_results)}")
    print(f"文档结果数: {len(result.doc_results)}")





if __name__ == "__main__":
    test_search_router("王洪文和张春桥好厉害哦")

# =============================================================================
# 第七部分：思考题（完成代码后回答）
# =============================================================================
"""
【思考题 1】为什么先查 FAQ，再决定要不要查文档？
    提示：从计算成本、响应速度、准确率三个角度分析。

【思考题 2】high_threshold=0.90 和 low_threshold=0.85 这两个值是怎么定的？
    提示：这和 embedding 模型的特性、数据分布有关。如果是你，会怎么调参？

【思考题 3】如果 FAQ 和 RAG 返回的结果有重叠（同一内容既在 FAQ 又在文档里），
    应该如何去重？

【思考题 4】原始脚本里的 `if not faq_results:` 出现了两次，
    第二次在"低置信度分支"里，理论上已经不可能为空了。
    思考：这是防御性编程吗？还是逻辑冗余？是否应该删掉第二次判断？

【思考题 5】在生产环境中，这个路由器的 top_k 和阈值是否应该做成可配置项？
    比如从环境变量或配置文件读取，而不是硬编码在代码里。
"""
