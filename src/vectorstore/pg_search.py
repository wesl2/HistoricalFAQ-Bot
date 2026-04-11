# -*- coding: utf-8 -*-
"""
FAQ 检索模块

提供向量检索、全文检索、混合检索等功能
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .pg_pool import get_connection
from config.pg_config import PG_TABLE_NAME, PG_DOC_TABLE, VECTOR_DIM

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """检索结果数据类"""
    id: int
    question: str
    answer: str
    score: float  # 相似度分数
    source_doc: Optional[str] = None
    source_page: Optional[int] = None
    category: Optional[str] = None


class FAQSearcher:
    """FAQ 向量检索器"""
    
    def __init__(self, table_name: str = None):
        self.table_name = table_name or PG_TABLE_NAME
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.85
    ) -> List[SearchResult]:
        """
        向量相似度检索
        
        Args:
            query_vector: 查询向量（1024维）
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        # 向量转字符串
        vector_str = "[" + ",".join([str(v) for v in query_vector]) + "]"
        
        sql = f"""
            SELECT 
                id, question, answer,
                1 - (similar_question_vector <=> %s::vector) as similarity,
                source_doc, source_page, category
            FROM {self.table_name}
            WHERE 1 - (similar_question_vector <=> %s::vector) > %s
            ORDER BY similar_question_vector <=> %s::vector
            LIMIT %s;
        """
        
        with get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, (vector_str, vector_str, similarity_threshold, vector_str, top_k))
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append(SearchResult(
                        id=row[0],
                        question=row[1],
                        answer=row[2],
                        score=row[3],
                        source_doc=row[4],
                        source_page=row[5],
                        category=row[6]
                    ))
                return results
            finally:
                cursor.close()
    
    def fulltext_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """全文检索（使用 GIN 索引）"""
        
        sql = f"""
            SELECT 
                id, question, answer,
                ts_rank(search_vector, plainto_tsquery('simple', %s)) as rank,
                source_doc, source_page, category
            FROM {self.table_name}
            WHERE search_vector @@ plainto_tsquery('simple', %s)
            ORDER BY rank DESC
            LIMIT %s;
        """
        
        with get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, (query, query, top_k))
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append(SearchResult(
                        id=row[0],
                        question=row[1],
                        answer=row[2],
                        score=row[3],
                        source_doc=row[4],
                        source_page=row[5],
                        category=row[6]
                    ))
                return results
            finally:
                cursor.close()


class HybridSearcher:
    """混合检索器：FAQ + 文档"""
    
    def __init__(self):
        self.faq_searcher = FAQSearcher()
        self.doc_table = PG_DOC_TABLE
    
    def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5,
        faq_weight: float = 0.6,
        doc_weight: float = 0.4
    ) -> Dict[str, List[SearchResult]]:
        """
        混合检索：同时搜索 FAQ 和文档
        
        Returns:
            {
                "faq": [...],
                "doc": [...]
            }
        """
        # FAQ 向量检索
        faq_results = self.faq_searcher.search(query_vector, top_k=top_k)
        
        # 文档向量检索
        doc_results = self._search_docs(query_vector, top_k=top_k)
        
        return {
            "faq": faq_results,
            "doc": doc_results
        }
    
    def _search_docs(
        self,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[SearchResult]:
        """检索文档片段"""
        vector_str = "[" + ",".join([str(v) for v in query_vector]) + "]"
        
        sql = f"""
            SELECT 
                id, chunk_text, doc_name,
                1 - (chunk_vector <=> %s::vector) as similarity,
                doc_name as source_doc, doc_page, NULL as category
            FROM {self.doc_table}
            ORDER BY chunk_vector <=> %s::vector
            LIMIT %s;
        """
        
        with get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, (vector_str, vector_str, top_k))
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append(SearchResult(
                        id=row[0],
                        question="",  # 文档片段没有 question
                        answer=row[1],  # chunk_text 作为 answer
                        score=row[3],
                        source_doc=row[4],
                        source_page=row[5],
                        category="document"
                    ))
                return results
            finally:
                cursor.close()
