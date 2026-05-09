# -*- coding: utf-8 -*-
"""
BM25 检索器（Python 实现）

使用 rank-bm25 库实现独立的 BM25 全文检索，不依赖数据库扩展。
适用于关键词匹配场景，如人名、地名、专有名词检索。

技术原理：
BM25（Best Matching 25）是一种基于概率的全文检索算法，
通过计算词频（TF）和逆文档频率（IDF）来评估文档相关性。

优势：
1. 关键词精确匹配（优于向量检索）
2. 可解释性强（匹配词可见）
3. 计算速度快（无需向量计算）
4. 独立于数据库（纯 Python 实现）

劣势：
1. 无法处理语义相似（"李世民" vs "唐太宗"）
2. 需要分词（中文需要额外处理）
3. 内存占用（需要加载全部文档）
"""

# TODO: 导入必要的模块
# 提示：
# 1. 导入日志模块 logging
# 2. 从 typing 导入类型提示：List, Dict, Any
# 3. 从 dataclasses 导入 dataclass 装饰器
# 4. 从 rank_bm25 导入 BM25Okapi
# 5. 导入 jieba（中文分词）
# 6. 导入项目内的模块：
#    - src.vectorstore.pg_pool 中的 get_connection
#    - config.pg_config 中的 PG_DOC_TABLE

# TODO: 创建日志记录器
import logging
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import jieba  # 中文分词
from src.vectorstore.pg_pool_practice import get_connection #TODO: practice 版本
from config.pg_config_practice import PG_DOC_TABLE
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# BM25 全局索引缓存（进程级单例）
# =============================================================================
# 设计：BM25 索引只加载一次，所有 DocRetriever 实例共享。
# 缓存 key 为 table_name（索引本身和 top_k 无关，top_k 只是截断结果），
# value 为 BM25Retriever 实例。当数据库文档更新时，调用 refresh_index() 会更新缓存。
_BM25_INDEX_CACHE: Dict[str, "BM25Retriever"] = {}
_BM25_CACHE_LOCK = threading.Lock()  # 真锁，防止并发初始化时重复建索引

@dataclass
class BM25Result:
    """BM25 检索结果"""
    id: int
    content: str 
    doc_name: str
    doc_page: int
    chunk_index: int
    score: float
    category: str = "bm25"
    

class BM25Retriever:
    """
    BM25 检索器
    
    特点：
    1. 适合关键词查询（人名、地名、专有名词）
    2. 可与向量检索互补（混合检索）
    3. 独立于数据库，纯 Python 实现
    4. 支持自动检测数据变化并热更新索引
    """
    
    def __init__(self, top_k: int = 10, _skip_cache: bool = False):
        """
        初始化 BM25 检索器
        
        Args:
            top_k: 返回结果数量
            _skip_cache: 内部标志，用于 refresh_index 时绕过缓存
        """
        self.top_k = top_k
        self.bm25: BM25Okapi = None
        self.documents = []
        self.tokenized_docs = []
        
        # 自动刷新相关状态
        self._last_doc_count = 0
        self._last_refresh_time = 0
        self._auto_refresh_interval = 60  # 每 60 秒检查一次数据变化
        
        # 缓存 key 只用表名，和 top_k 无关（索引本身和 top_k 无关，top_k 只是截断结果）
        cache_key = PG_DOC_TABLE
        
        if not _skip_cache and cache_key in _BM25_INDEX_CACHE:
            cached = _BM25_INDEX_CACHE[cache_key]
            self.bm25 = cached.bm25
            self.documents = cached.documents
            self.tokenized_docs = cached.tokenized_docs
            self._last_doc_count = getattr(cached, '_last_doc_count', len(self.documents))
            self._last_refresh_time = getattr(cached, '_last_refresh_time', 0)
            logger.info(f"BM25 索引从缓存加载 | 文档数: {len(self.documents)}")
        else:
            with _BM25_CACHE_LOCK:
                # 双重检查：拿到锁后再确认一次缓存是否已被其他线程建好
                if not _skip_cache and cache_key in _BM25_INDEX_CACHE:
                    cached = _BM25_INDEX_CACHE[cache_key]
                    self.bm25 = cached.bm25
                    self.documents = cached.documents
                    self.tokenized_docs = cached.tokenized_docs
                    self._last_doc_count = getattr(cached, '_last_doc_count', len(self.documents))
                    self._last_refresh_time = getattr(cached, '_last_refresh_time', 0)
                    logger.info(f"BM25 索引从缓存加载 | 文档数: {len(self.documents)}")
                else:
                    self._load_documents()
                    self._last_doc_count = len(self.documents)
                    self._last_refresh_time = __import__('time').time()
                    if not _skip_cache:
                        _BM25_INDEX_CACHE[cache_key] = self
                        logger.info(f"BM25 索引已缓存 | key={cache_key}")


    def _load_documents(self):
        """
        从数据库加载文档并构建 BM25 索引
        
        流程：
        1. 从 PostgreSQL 读取所有文档片段
        2. 使用 jieba 分词
        3. 构建 BM25 索引（内存中）
        """
        # 实现文档加载和索引构建
        # 提示：
        # 1. 记录日志：开始加载文档并构建 BM25 索引
        # 2. 使用 get_connection() 获取数据库连接
        # 3. 执行 SQL 查询：从 PG_DOC_TABLE 表中读取
        #    id, chunk_text, doc_name, doc_page, chunk_index 字段
        # 4. 将查询结果转换为 documents 列表（字典列表）
        #    注意处理 doc_page 和 chunk_index 的 None 值
        # 5. 关闭 cursor
        # 6. 使用 jieba.cut() 对每个文档内容进行分词
        #    将分词结果转换为列表存入 tokenized_docs
        # 7. 使用 BM25Okapi 构建索引
        # 8. 记录日志：显示文档数量和总词数

        logging.info("加载文档并构建 BM25 索引...")
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT ID, CHUNK_TEXT, PARENT_TEXT, DOC_NAME, DOC_PAGE, CHUNK_INDEX
                    FROM {PG_DOC_TABLE}
                    ORDER BY ID
                """)
                for row in cursor.fetchall():
                       self.documents.append(
                        {
                            "id": row[0],
                            "content": row[1],           # chunk_text: 用于 BM25 分词
                            "parent_text": row[2],       # parent_text: 用于返回给 LLM
                            "doc_name": row[3],
                            "doc_page": row[4] if row[4] is not None else 0,
                            "chunk_index": row[5] if row[5] is not None else 0
                        }
                       )
                cursor.close()
        
        
        #print(f"加载了 {len(self.documents)} 个文档")  # 看看是不是 0
        #print(f"第一个文档: {self.documents[0] if self.documents else '空'}")
        
        self.tokenized_docs = []
        # 中文分词（使用 jieba）    
        for doc in self.documents:
            tokens = list(jieba.cut(doc["content"])) #list[list[str]]
            self.tokenized_docs.append(tokens)
        
        # 空文档保护：文档表为空时不初始化 BM25，避免 ZeroDivisionError
        if not self.tokenized_docs:
            logger.warning("文档表为空，BM25 索引未初始化")
            self.bm25 = None
            return
        
        # corpus 必须是 List[List[str]]（列表的列表），即每个文档已经分词后的词列表。
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info(f"文档加载完成，数量: {len(self.documents)}, 总词数: {sum(len(tokens) for tokens in self.tokenized_docs)}")
    
    
    def _get_db_doc_count(self) -> int:
        """轻量查询数据库当前记录数"""
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {PG_DOC_TABLE}")
                    count = cursor.fetchone()[0]
                    return count
        except Exception as e:
            logger.warning("[BM25] 查询数据库记录数失败: %s", e)
            return self._last_doc_count  # 失败时返回上次记录数，避免误刷新
    
    def _check_and_refresh(self):
        """
        自动检测数据变化并刷新索引。
        
        策略：每隔 _auto_refresh_interval 秒检查一次数据库记录数，
        如果记录数变化了，自动调用 refresh_index() 重建索引。
        """
        import time
        now = time.time()
        if now - self._last_refresh_time < self._auto_refresh_interval:
            return  # 还没到检查间隔
        
        current_count = self._get_db_doc_count()
        if current_count != self._last_doc_count:
            logger.info(
                "[BM25] 检测到数据变化: %d → %d，自动刷新索引",
                self._last_doc_count, current_count
            )
            self.refresh_index()
        else:
            self._last_refresh_time = now  # 更新时间戳，避免频繁查询
    
    def retrieve(self, query: str) -> List[BM25Result]:
        """
        BM25 检索
        
        Args:
            query: 用户查询
        
        Returns:
            BM25 检索结果列表（按分数降序）
        """
        # 自动检测数据变化（热更新）
        self._check_and_refresh()
        
        if not self.bm25:
            logger.error("BM25 索引未加载")
            return []
        tokenized_query = list(jieba.cut(query))
        # BM25 打分
        scores = self.bm25.get_scores(tokenized_query)
        logger.debug("BM25 查询: %s | 最高分: %s", query, max(scores) if len(scores) > 0 else 'N/A')
        # 取 Top-K
        # 使用 argsort 获取排序索引
        # 通过 BM25 获取的是分数  ， np.argsort(scores) 返回的才是索引
        top_indices = np.argsort(scores)[-self.top_k * 2:][::-1]
        results = []
        seen_parents = set()
        for idx in top_indices:
            if scores[idx] < 0.1:
                continue
            doc = self.documents[idx]
            # PDR：优先返回 parent_text，fallback 到 chunk_text（兼容旧数据）
            parent_text = doc.get("parent_text") or doc["content"]
            if parent_text in seen_parents:
                continue
            seen_parents.add(parent_text)
            result = BM25Result(
                id=doc["id"],
                content=parent_text,
                doc_name=doc["doc_name"],
                doc_page=doc["doc_page"],
                chunk_index=doc["chunk_index"],
                score=scores[idx]
            )
            results.append(result)
            if len(results) >= self.top_k:
                break
        logger.info(f"BM25 检索完成：找到 {len(results)} 个片段（查询：{query}）")
        return results
    
    def retrieve_with_highlights(
        self, 
        query: str,
        highlight: bool = True
    ) -> List[BM25Result]:
        """
        BM25 检索（带关键词高亮）
        
        Args:
            query: 用户查询
            highlight: 是否高亮匹配词
        
        Returns:
            包含高亮信息的检索结果
        """
        # 实现带高亮的检索
        # 提示：
        # 1. 调用 retrieve() 获取检索结果
        # 2. 使用 jieba.cut() 对查询分词，转换为集合（去重）
        # 3. 遍历检索结果：
        #    - 复制内容用于高亮处理
        #    - 如果 highlight 为 True，遍历查询词：
        #      * 如果词长度大于 1 且在内容中，用 **词** 包裹实现高亮
        #    - 构建返回字典，包含：
        #      id, content(高亮后), doc_name, doc_page, score
        #      highlights: 查询词和内容的交集（实际匹配的词）
        # 4. 返回高亮结果列表
        
        results = self.retrieve(query)
        # 提取查询词（用于高亮）
        query_tokens = set(jieba.cut(query))
        highlighted_results = []
        for r in results:
            content = r.content
            if highlight:
                for token in query_tokens:
                    if len(token) > 1 and token in content:
                        content = content.replace(token, f"**{token}**")

            highlighted_results.append(BM25Result(
                id=r.id,
                content=content,
                doc_name=r.doc_name,
                doc_page=r.doc_page,
                chunk_index=r.chunk_index,
                score=r.score,
            ))
        return highlighted_results

    def refresh_index(self):
        """
        刷新 BM25 索引（当数据库文档更新时调用）
        """
        logger.info("刷新 BM25 索引...")
        cache_key = PG_DOC_TABLE
        with _BM25_CACHE_LOCK:
            if cache_key in _BM25_INDEX_CACHE:
                del _BM25_INDEX_CACHE[cache_key]
                logger.info(f"BM25 缓存已清除 | key={cache_key}")
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None
        self._load_documents()
        with _BM25_CACHE_LOCK:
            _BM25_INDEX_CACHE[cache_key] = self
        logger.info(f"BM25 索引已重新缓存 | key={cache_key}")

# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # TODO: 实现测试代码
    # 提示：
    # 1. 打印测试标题分隔线
    # 2. 创建 BM25Retriever 实例（top_k=5）
    # 3. 定义测试查询列表（如："李世民", "玄武门之变", "贞观之治"）
    # 4. 遍历测试查询：
    #    - 打印查询信息
    #    - 调用 retrieve() 获取结果
    #    - 如果没有结果，打印提示
    #    - 否则遍历结果，打印序号、分数、来源、内容前100字
    # 5. 打印测试完成分隔线
    import sys
    
    print("=" * 60)
    print("BM25 检索器测试")
    print("=" * 60)
    
    # 创建检索器
    retriever = BM25Retriever(top_k=5)
    
    # 测试查询
    test_queries = [
        "李世民",
        "玄武门之变",
        "贞观之治",
        "王洪文",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询：{query}")
        print(f"{'='*60}")
        
        results = retriever.retrieve_with_highlights(query)
        
        if not results:
            print("  未找到匹配结果")
            continue
        
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] 分数：{r.score:.3f}")
            print(f"      来源：{r.doc_name} 第{r.doc_page}页")
            print(f"      内容：{r.content[:100]}...")

        # for i, r in enumerate(results, 1):
        #     print(f"\n  [{i}] 分数：{r['score']:.3f}")
        #     print(f"      来源：{r['doc_name']} 第{r['doc_page']}页")
        #     print(f"      内容：{r['content'][:100]}...")
    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")


