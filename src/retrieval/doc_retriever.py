# -*- coding: utf-8 -*-
"""
文档 RAG 检索器（支持 RRF 倒数排名融合）

支持多种检索模式：
1. 向量检索（默认）- 语义相似度匹配
2. BM25 检索 - 关键词精确匹配
3. 混合检索 - 向量 + BM25 融合
   - 线性加权（linear）：Min-Max 归一化 + 加权融合
   - RRF（倒数排名融合）：工业界标准方法

使用场景：
- 向量检索：开放性问题、语义查询（"李世民如何治国"）
- BM25 检索：关键词查询、专有名词（"玄武门之变发生在哪一年"）
- 混合检索：兼顾语义和关键词（推荐，用 RRF）
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.embedding.embedding_local import get_embedding
from src.vectorstore.pg_pool import get_connection
from config.pg_config import PG_DOC_TABLE

logger = logging.getLogger(__name__)


@dataclass
class DocResult:
    """
    文档检索结果

    Attributes:
        id: 文档片段 ID
        content: 文档片段内容
        doc_name: 来源文档名
        doc_page: 页码
        chunk_index: 片段序号
        similarity: 相似度分数或综合分数
        category: 结果类别（vector/bm25/hybrid）
        bm25_score: BM25 分数（仅混合检索时有值）
        vector_score: 向量分数（仅混合检索时有值）
        rrf_score: RRF 分数（仅 RRF 融合时有值）
        vector_rank: 向量检索排名（仅 RRF 时有值）
        bm25_rank: BM25 检索排名（仅 RRF 时有值）
    """
    id: int
    content: str
    doc_name: str
    doc_page: Optional[int]
    chunk_index: int
    similarity: float
    category: str = "document"
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    rrf_score: Optional[float] = None
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


class DocRetriever:
    """
    文档检索器（支持 RRF 倒数排名融合）

    参数说明：
    - use_bm25=False: 纯向量检索
    - use_bm25=True: 向量 + BM25 混合检索
    - fusion_method="rrf": 使用 RRF 融合（推荐）
    - fusion_method="linear": 使用线性加权
    """

    def __init__(
        self,
        top_k: int = 10,
        use_bm25: bool = False,
        bm25_weight: float = 0.3,
        fusion_method: str = "rrf",  # 【新增】融合方法：rrf 或 linear
        rrf_k: int = 60  # 【新增】RRF 参数 K
    ):
        """
        初始化文档检索器

        Args:
            top_k: 返回结果数量
            use_bm25: 是否启用 BM25 混合检索
            bm25_weight: BM25 权重（仅 linear 模式使用）
            fusion_method: 融合方法 - "rrf"（推荐）或 "linear"
            rrf_k: RRF 参数 K（通常 60，仅 rrf 模式使用）
        """
        self.top_k = top_k
        self.use_bm25 = use_bm25
        self.bm25_weight = bm25_weight
        self.fusion_method = fusion_method  # 【新增】
        self.rrf_k = rrf_k  # 【新增】
        self.doc_table = PG_DOC_TABLE

        # BM25 检索器（懒加载）
        self._bm25_retriever = None

        if self.use_bm22:
            logger.info(
                f"BM25 混合检索已启用："
                f"融合方法={fusion_method}, "
                f"BM25 权重={bm25_weight}, "
                f"RRF_K={rrf_k}"
            )

    @property
    def bm25_retriever(self):
        """懒加载 BM25 检索器"""
        if self._bm25_retriever is None:
            from .bm25_retriever import BM25Retriever
            self._bm25_retriever = BM25Retriever(top_k=self.top_k * 2)
        return self._bm25_retriever

    def retrieve(self, query: str) -> List[DocResult]:
        """
        检索文档片段（根据配置自动选择检索方式）

        Args:
            query: 用户问题

        Returns:
            文档检索结果列表
        """
        if self.use_bm25:
            return self.retrieve_hybrid(query)
        else:
            return self.retrieve_vector(query)

    def retrieve_vector(self, query: str) -> List[DocResult]:
        """
        纯向量检索

        优势：语义理解好（"李世民" = "唐太宗"）
        劣势：关键词匹配弱

        Args:
            query: 用户问题

        Returns:
            向量检索结果
        """
        # 1. 计算查询向量
        query_vector = get_embedding(query)

        # 2. SQL 检索
        with get_connection() as conn:
            cursor = conn.cursor()

            try:
                # 向量转字符串
                vector_str = "[" + ",".join([str(v) for v in query_vector]) + "]"

                # 向量相似度检索（余弦距离）
                cursor.execute(f"""
                    SELECT
                        id, chunk_text, doc_name, doc_page, chunk_index,
                        1 - (chunk_vector <=> %s::vector) AS similarity
                    FROM {self.doc_table}
                    ORDER BY similarity DESC
                    LIMIT %s;
                """, (vector_str, self.top_k))

                results = []
                for row in cursor.fetchall():
                    results.append(DocResult(
                        id=row[0],
                        content=row[1],
                        doc_name=row[2],
                        doc_page=row[3],
                        chunk_index=row[4] if row[4] is not None else 0,
                        similarity=row[5],
                        category="vector"
                    ))

                logger.info(f"向量检索完成：找到 {len(results)} 个片段")
                return results
            finally:
                cursor.close()

    def retrieve_bm25(self, query: str) -> List[DocResult]:
        """
        纯 BM25 检索

        优势：关键词精确匹配、可解释性强
        劣势：无法处理语义相似

        Args:
            query: 用户问题

        Returns:
            BM25 检索结果
        """
        bm25_results = self.bm25_retriever.retrieve(query)

        # 转换为 DocResult 格式
        results = []
        for r in bm25_results:
            results.append(DocResult(
                id=r.id,
                content=r.content,
                doc_name=r.doc_name,
                doc_page=r.doc_page,
                chunk_index=r.chunk_index,
                similarity=r.score,
                category="bm25",
                bm25_score=r.score
            ))

        logger.info(f"BM25 检索完成：找到 {len(results)} 个片段")
        return results

    def retrieve_hybrid(self, query: str) -> List[DocResult]:
        """
        混合检索：向量 + BM25

        根据 fusion_method 选择融合方法：
        - "rrf": RRF 倒数排名融合（推荐，工业界标准）
        - "linear": 线性加权（Min-Max 归一化）

        Args:
            query: 用户问题

        Returns:
            混合检索结果
        """
        if self.fusion_method == "rrf":
            return self._retrieve_rrf(query)
        else:
            return self._retrieve_linear(query)

    def _retrieve_linear(self, query: str) -> List[DocResult]:
        """
        线性加权融合（原有方法）

        流程：
        1. Min-Max 归一化（0-1 范围）
        2. 加权融合：final_score = 0.7×vector + 0.3×bm25
        3. 排序返回

        缺点：
        - 对异常值敏感
        - 新文档加入后分数分布会变
        """
        # 1. 同时检索
        vector_results = self.retrieve_vector(query)
        bm25_results = self.retrieve_bm25(query)

        # 2. Min-Max 归一化
        def normalize(results):
            if not results:
                return {}
            scores = [r.similarity for r in results]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1
            return {r.id: (r.similarity - min_s) / range_s for r in results}

        vector_scores = normalize(vector_results)
        bm25_scores = normalize(bm25_results)

        # 3. 合并所有文档
        all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())

        # 4. 创建 ID 到文档的映射（用于查找完整文档信息）
        vector_dict = {r.id: r for r in vector_results}
        bm25_dict = {r.id: r for r in bm25_results}

        # 5. 加权融合
        vector_weight = 1 - self.bm25_weight
        merged_results = []

        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, 0)
            b_score = bm25_scores.get(doc_id, 0)
            final_score = vector_weight * v_score + self.bm25_weight * b_score

            doc = vector_dict.get(doc_id) or bm25_dict.get(doc_id)
            merged_results.append(DocResult(
                id=doc.id,
                content=doc.content,
                doc_name=doc.doc_name,
                doc_page=doc.doc_page,
                chunk_index=doc.chunk_index,
                similarity=final_score,
                category="hybrid",
                vector_score=v_score,
                bm25_score=b_score
            ))

        # 5. 排序返回 Top-K
        merged_results.sort(key=lambda x: x.similarity, reverse=True)
        final_results = merged_results[:self.top_k]

        logger.info(
            f"混合检索完成（线性）: 找到 {len(final_results)} 个片段 "
            f"(向量权重={vector_weight:.1f}, BM25 权重={self.bm25_weight:.1f})"
        )

        return final_results

    def _retrieve_rrf(self, query: str) -> List[DocResult]:
        """
        RRF（Reciprocal Rank Fusion）倒数排名融合 ⭐ 新增

        工业界标准方法，Google/微软等公司都在用。

        公式：
            score = 1/(K+rank_vector) + 1/(K+rank_bm25)

        优势：
        1. 不需要归一化（天然免疫量纲差异）
        2. 对异常值不敏感
        3. 只关心相对排名，稳定可靠

        参数：
        - K: 平滑常数，通常设为 60（论文推荐值）
          - K 越大：排名差异影响越小
          - K 越小：排名差异影响越大

        流程：
        1. 分别进行向量检索和 BM25 检索
        2. 获取每个文档的排名（rank）
        3. 使用 RRF 公式计算融合分数
        4. 按 RRF 分数排序返回
        """
        # ========== 步骤 1：分别检索 ==========
        vector_results = self.retrieve_vector(query)
        bm25_results = self.retrieve_bm25(query)

        logger.info(
            f"RRF 检索：向量 {len(vector_results)} 条，"
            f"BM25 {len(bm25_results)} 条"
        )

        # ========== 步骤 2：构建排名映射 ==========
        # rank_map[doc_id] = rank (从 1 开始)
        def get_rank_map(results: List[DocResult]) -> Dict[int, int]:
            """
            构建文档 ID 到排名的映射

            Args:
                results: 检索结果列表（已按分数降序排序）

            Returns:
                {doc_id: rank} 字典，rank 从 1 开始
            """
            return {r.id: rank + 1 for rank, r in enumerate(results)}

        vector_ranks = get_rank_map(vector_results)
        bm25_ranks = get_rank_map(bm25_results)

        # ========== 步骤 3：合并所有文档 ID ==========
        all_doc_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        # ========== 步骤 4：RRF 融合 ==========
        K = self.rrf_k  # RRF 参数 K（通常 60）
        merged_scores = {}

        for doc_id in all_doc_ids:
            # 获取排名（如果某路检索没有该文档，给一个最大排名）
            v_rank = vector_ranks.get(doc_id, len(vector_results) + 1)
            b_rank = bm25_ranks.get(doc_id, len(bm25_results) + 1)

            # RRF 公式：1/(K+rank_v) + 1/(K+rank_b)
            rrf_score = 1.0 / (K + v_rank) + 1.0 / (K + b_rank)

            merged_scores[doc_id] = rrf_score

        # ========== 步骤 5：按 RRF 分数排序 ==========
        sorted_doc_ids = sorted(
            merged_scores.keys(),
            key=lambda x: merged_scores[x],
            reverse=True
        )

        # ========== 步骤 6：构建返回结果 ==========
        # 合并两个结果映射，优先使用向量检索的结果
        vector_dict = {r.id: r for r in vector_results}
        bm25_dict = {r.id: r for r in bm25_results}

        final_results = []
        for doc_id in sorted_doc_ids[:self.top_k]:
            # 优先使用向量检索的结果（内容更完整）
            doc = vector_dict.get(doc_id) or bm25_dict.get(doc_id)

            v_rank = vector_ranks.get(doc_id, -1)
            b_rank = bm25_ranks.get(doc_id, -1)
            rrf_score = merged_scores[doc_id]

            final_results.append(DocResult(
                id=doc.id,
                content=doc.content,
                doc_name=doc.doc_name,
                doc_page=doc.doc_page,
                chunk_index=doc.chunk_index,
                similarity=rrf_score,  # 使用 RRF 分数作为 similarity
                category="hybrid_rrf",
                rrf_score=rrf_score,
                vector_rank=v_rank,
                bm25_rank=b_rank
            ))

        logger.info(
            f"RRF 检索完成：返回 {len(final_results)} 个片段 "
            f"(K={K})"
        )

        return final_results

    def refresh_bm25_index(self):
        """
        刷新 BM25 索引

        当数据库中文档更新时，需要调用此方法刷新 BM25 索引。
        """
        if self._bm25_retriever:
            self._bm25_retriever.refresh_index()
            logger.info("BM25 索引已刷新")


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("文档检索器测试（支持 RRF 倒数排名融合）")
    print("=" * 60)

    # 测试查询
    test_queries = [
        ("玄武门之变", "关键词查询（BM25 优势）"),
        ("李世民如何治国", "语义查询（向量优势）"),
        ("贞观之治有哪些成就", "混合查询"),
    ]

    for query, description in test_queries:
        print(f"\n{'#'*60}")
        print(f"查询：{query}")
        print(f"类型：{description}")
        print(f"{'#'*60}")

        # 1. 纯向量检索
        retriever_v = DocRetriever(top_k=5, use_bm25=False)
        results_v = retriever_v.retrieve(query)

        print(f"\n【向量检索】Top 3:")
        for r in results_v[:3]:
            print(f"  [{r.category}] {r.doc_name} - 分数：{r.similarity:.3f}")

        # 2. RRF 混合检索
        retriever_rrf = DocRetriever(
            top_k=5,
            use_bm25=True,
            fusion_method="rrf",
            rrf_k=60
        )
        results_rrf = retriever_rrf.retrieve(query)

        print(f"\n【RRF 混合检索】Top 3:")
        for r in results_rrf[:3]:
            print(f"  [{r.category}] {r.doc_name}")
            print(f"       RRF 分数：{r.similarity:.4f}")
            print(f"       向量排名：{r.vector_rank}, BM25 排名：{r.bm25_rank}")

    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")
