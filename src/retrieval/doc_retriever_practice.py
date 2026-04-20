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

# TODO: 导入必要的模块
# 提示：
# 1. 导入日志模块 logging
# 2. 从 typing 导入类型提示：List, Optional, Dict, Any
# 3. 从 dataclasses 导入 dataclass 装饰器
# 4. 导入项目内的模块：
#    - src.embedding.embedding_local 中的 get_embedding
#    - src.vectorstore.pg_pool 中的 get_connection
#    - config.pg_config 中的 PG_DOC_TABLE
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from src.embedding.embedding_local_practice import get_embedding #TODO:看最后要不要统一把practice去掉
from src.vectorstore.pg_pool_practice import get_connection
from config.pg_config_practice import PG_DOC_TABLE
from rank_bm25 import BM25Okapi
from src.retrieval.bm25_retriever_practice import BM25Retriever



# TODO: 创建日志记录器
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
    # TODO: 定义数据类的字段
    id: int
    content: str
    doc_name: str
    doc_page: Optional[int] = None
    chunk_index: Optional[int] = None
    similarity: Optional[float] = None
    category: Optional[str] = None
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
        use_bm25: bool = True,
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
        # TODO: 初始化实例属性
        # 提示：
        # 1. 将 top_k, use_bm25, bm25_weight, fusion_method, rrf_k 保存为实例属性
        # 2. 从配置中获取 doc_table（PG_DOC_TABLE）
        # 3. 初始化 _bm25_retriever 为 None（懒加载）
        # 4. 如果 use_bm25 为 True，记录日志显示混合检索配置信息
        self.top_k = top_k
        self.use_bm25 = use_bm25
        self.bm25_weight = bm25_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.doc_table = PG_DOC_TABLE
        self._bm25_retriever = None
        if self.use_bm25:
            logger.info(f"Initialized DocRetriever with BM25 hybrid retrieval. "
                        f"BM25 weight: {self.bm25_weight}, Fusion method: {self.fusion_method}, RRF K: {self.rrf_k}")
        else:
            logger.info("Initialized DocRetriever with pure vector retrieval.")


    @property
    def bm25_retriever(self):
        """懒加载 BM25 检索器"""
        # TODO: 实现 BM25 检索器的懒加载
        # 提示：
        # 1. 检查 _bm25_retriever 是否为 None
        # 2. 如果是，从当前目录导入 BM25Retriever
        # 3. 创建实例，top_k 设置为 self.top_k * 2
        # 4. 返回 _bm25_retriever
        if self._bm25_retriever is None:
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
        # TODO: 根据配置选择检索方式
        # 提示：
        # - 如果 self.use_bm25 为 True，调用 retrieve_hybrid
        # - 否则调用 retrieve_vector
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
        # TODO: 实现纯向量检索
        # 提示：
        # 1. 调用 get_embedding(query) 计算查询向量
        # 2. 使用 get_connection() 获取数据库连接
        # 3. 将向量转换为 PostgreSQL 向量格式字符串 "[x,x,x]"
        # 4. 执行 SQL 向量相似度检索（使用余弦距离 <=>）
        # 5. 将查询结果转换为 DocResult 对象列表
        # 6. 记录日志并返回结果
        # 注意：记得关闭 cursor
        query_vector = get_embedding(query)
        query_str = '[' + ','.join(map(str,query_vector)) + ']'
        with get_connection() as conn:
            with conn.cursor() as cursor:
                sql = f"""
                SELECT id,chunk_text,doc_name,doc_page,chunk_index,
                1 - (chunk_vector <=> %s::vector) AS similarity
                FROM {self.doc_table}
                ORDER BY similarity DESC
                LIMIT %s      
                """
                cursor.execute(sql,(query_str,self.top_k))
                results = cursor.fetchall()
                doc_results = []
                for row in results:
                    doc_result = DocResult(
                        id=row[0],
                        content=row[1],
                        doc_name=row[2],
                        doc_page=row[3],
                        chunk_index=row[4],
                        similarity=row[5],
                        category="vector"
                    )
                    doc_results.append(doc_result)
                
                logger.info(f"向量检索完成：找到 {len(results)} 个片段")
                    
                return doc_results    


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
        # TODO: 实现纯 BM25 检索
        # 提示：
        # 1. 调用 self.bm25_retriever.retrieve(query) 获取结果
        # 2. 将 BM25 结果转换为 DocResult 列表
        # 3. 设置 category="bm25"，保存 bm25_score
        # 4. 记录日志并返回
        bm25_results = self.bm25_retriever.retrieve(query)
        doc_results = []
        for bm25_result in bm25_results:
            doc_result = DocResult(
                id=bm25_result.id,
                content=bm25_result.content,
                doc_name=bm25_result.doc_name,
                doc_page=bm25_result.doc_page,
                chunk_index=bm25_result.chunk_index,
                similarity=bm25_result.score,
                category="bm25",
                bm25_score=bm25_result.score
            )
            doc_results.append(doc_result)
        logger.info(f"BM25 检索完成：找到 {len(doc_results)} 个片段")
        return doc_results


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
        # TODO: 根据融合方法选择实现
        # 提示：
        # - 如果 self.fusion_method == "rrf"，调用 _retrieve_rrf
        # - 否则调用 _retrieve_linear
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
        # TODO: 实现线性加权融合
        # 提示：
        # 1. 调用 retrieve_vector 和 retrieve_bm25 获取结果
        # 2. 创建 Min-Max 归一化函数：
        #    - 如果没有结果返回空字典
        #    - 计算所有分数的最小值和最大值
        #    - 返回 {doc_id: normalized_score} 字典
        # 3. 分别对向量分数和 BM25 分数进行归一化
        # 4. 合并所有文档 ID（使用集合的并集）
        # 5. 创建 ID 到文档的映射（方便查找）
        # 6. 加权融合计算最终分数：
        #    - vector_weight = 1 - self.bm25_weight
        #    - final_score = vector_weight * v_score + bm25_weight * b_score
        # 7. 按分数降序排序，取前 top_k
        # 8. 记录日志并返回

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
        # TODO: 实现 RRF 倒数排名融合
        # 提示：
        # ========== 步骤 1：分别检索 ==========
        # 获取向量结果和 BM25 结果
        # 记录日志显示各路检索结果数量
        #
        # ========== 步骤 2：构建排名映射 ==========
        # 创建辅助函数 get_rank_map(results):
        #   - 返回 {doc_id: rank} 字典，rank 从 1 开始
        # 分别获取 vector_ranks 和 bm25_ranks
        #
        # ========== 步骤 3：合并所有文档 ID ==========
        # 使用集合的并集获取所有文档 ID
        #
        # ========== 步骤 4：RRF 融合 ==========
        # K = self.rrf_k
        # 遍历所有文档 ID：
        #   - 获取向量排名（如果没有则设为 len(vector_results)+1）
        #   - 获取 BM25 排名（如果没有则设为 len(bm25_results)+1）
        #   - 计算 rrf_score = 1/(K+v_rank) + 1/(K+b_rank)
        #
        # ========== 步骤 5：按 RRF 分数排序 ==========
        # 按 merged_scores 的分数降序排序
        #
        # ========== 步骤 6：构建返回结果 ==========
        # 创建 vector_dict 和 bm25_dict 映射
        # 遍历排序后的文档 ID（取前 top_k）：
        #   - 优先从 vector_dict 获取文档，否则从 bm25_dict
        #   - 创建 DocResult，设置：
        #     * category="hybrid_rrf"
        #     * rrf_score, vector_rank, bm25_rank
        # 记录日志并返回
        vector_results = self.retrieve_vector(query)
        bm25_results = self.retrieve_bm25(query)
        logger.info(f"混合检索：向量结果 {len(vector_results)} 个，BM25 结果 {len(bm25_results)} 个")

        def get_rank_map(results: List[DocResult]) -> Dict[int, int]:
            rank_map = {}
            for rank,result in enumerate(results, start=1):
                rank_map[result.id] = rank
            return rank_map
        
        vector_ranks = get_rank_map(vector_results)
        bm25_ranks = get_rank_map(bm25_results)

        all_doc_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        k = self.rrf_k
        merged_scores = {}

        for doc_id in all_doc_ids:
            vector_rank = vector_ranks.get(doc_id,len(vector_results) + 1)
            bm25_rank = bm25_ranks.get(doc_id,len(bm25_results) + 1)
            rrf_score = 1/(k + vector_rank) + 1/(k + bm25_rank)
            merged_scores[doc_id] = rrf_score
        
        score_items = list(merged_scores.items())
        score_items.sort(key=lambda item: item[1], reverse=True) #传入 item 返回item[1]
        #score_items是从大到小的
        topk_doc_ids = [doc_id for doc_id,rrf_score in score_items[:self.top_k]]

        vector_dict = {result.id: result for result in vector_results}
        bm25_dict = {result.id: result for result in bm25_results}
        final_results = []
        for doc_id in topk_doc_ids:
            if doc_id in vector_dict:
                base_result = vector_dict[doc_id]
            else:
                base_result = bm25_dict[doc_id]
            final_result = DocResult(
                id=base_result.id,
                content=base_result.content,
                doc_name=base_result.doc_name,
                doc_page=base_result.doc_page,
                chunk_index=base_result.chunk_index,
                similarity=merged_scores[doc_id],
                category="hybrid_rrf",
                rrf_score=merged_scores[doc_id],
                vector_rank=vector_ranks.get(doc_id, len(vector_results) + 1),
                bm25_rank=bm25_ranks.get(doc_id, len(bm25_results) + 1)
            )
            final_results.append(final_result)

        logger.info(f"RRF 检索完成：返回 {len(final_results)} 个片段")
        return final_results
    def refresh_bm25_index(self):
        """
        刷新 BM25 索引

        当数据库中文档更新时，需要调用此方法刷新 BM25 索引。
        """
        # TODO: 实现 BM25 索引刷新
        # 提示：
        # 1. 检查 _bm25_retriever 是否存在
        # 2. 如果存在，调用其 refresh_index() 方法
        # 3. 记录日志
        if self._bm25_retriever is not None:
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
        ("王洪文和李世民的历史地位", "关键词查询（BM25 优势）"),
        #("李世民如何治国", "语义查询（向量优势）"),
        #("贞观之治有哪些成就", "混合查询"),
    ]

    for query, description in test_queries:
        print(f"\n{'#'*60}")
        print(f"查询：{query}")
        print(f"类型：{description}")
        print(f"{'#'*60}")

        # TODO: 实现测试代码
        # 提示：
        # 1. 创建纯向量检索器（top_k=5, use_bm25=False）
        # 2. 调用 retrieve 获取结果，打印 Top 3
        # 3. 创建 RRF 混合检索器（top_k=5, use_bm25=True, fusion_method="rrf", rrf_k=60）
        # 4. 调用 retrieve 获取结果，打印 Top 3 及其 RRF 分数、排名信息
        # vector_retriever = DocRetriever(top_k=5, use_bm25=False)
        # vector_results = vector_retriever.retrieve(query)
        # print("\n纯向量检索结果：")
        # if not vector_results:
        #     print("未找到相关文档片段")
        # else:
        #     for i, result in enumerate(vector_results[:3], start=1):
        #         print(f"{i}. [相似度: {result.similarity:.4f}] 来源: {result.doc_name} (页 {result.doc_page}) 内容: {result.content[:100]}...")


        hybrid_retriever = DocRetriever(top_k=5, use_bm25=True, fusion_method="rrf", rrf_k=60)
        hybrid_results = hybrid_retriever.retrieve(query)
        print("\nRRF 混合检索结果：")
        if not hybrid_results:
            print("未找到相关文档片段")
        else:
            for i, result in enumerate(hybrid_results[:3], start=1):
                print(f"{i}. [RRF 分数: {result.rrf_score:.6f}] 来源: {result.doc_name} (页 {result.doc_page}) 内容: {result.content[:100]}...")
                print(f"   向量排名: {result.vector_rank}, BM25 排名: {result.bm25_rank}")


    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")
