# -*- coding: utf-8 -*-
"""FAQ 检索器"""
import logging
from typing import List, NamedTuple
from src.embedding.embedding_local import get_embedding
from src.vectorstore.pg_pool import get_connection
from config.pg_config import PG_TABLE_NAME

logger = logging.getLogger(__name__)

class FAQResult(NamedTuple):
    """FAQ 检索结果"""
    id: int
    question: str
    similar_question: str
    answer: str
    similarity: float
    category: str
    source_doc: str


class FAQRetriever:
    """FAQ 检索器"""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[FAQResult]:
        """
        检索 FAQ
        
        Args:
            query: 用户问题
        
        Returns:
            FAQ 结果列表
        """
        # 1. 计算查询向量
        query_vector = get_embedding(query)
        
        # 2. SQL 检索
        with get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # 向量相似度检索 (余弦距离)
                cursor.execute(f"""
                    SELECT 
                        id, question, similar_question, answer,
                        1 - (similar_question_vector <=> %s::vector) AS similarity,
                        category, source_doc
                    FROM {PG_TABLE_NAME}
                    ORDER BY similarity DESC
                    LIMIT %s;
                """, (f"[{','.join(map(str, query_vector))}]", self.top_k))
                
                results = []
                for row in cursor.fetchall():
                    results.append(FAQResult(
                        id=row[0],
                        question=row[1],
                        similar_question=row[2],
                        answer=row[3],
                        similarity=row[4],
                        category=row[5] or "general",
                        source_doc=row[6] or "unknown"
                    ))
                #fetchall :默认返回 list 包 tuple，类似 [(1, 'Alice'), (2, 'Bob')]。

                logger.info(f"FAQ 检索完成：找到 {len(results)} 条记录")
                return results
            finally:
                cursor.close()


# 测试
if __name__ == "__main__":
    retriever = FAQRetriever(top_k=3)
    results = retriever.retrieve("王洪文是谁？")
    
    for r in results:
        print(f"相似度：{r.similarity:.3f}")
        print(f"问题：{r.question}")
        print(f"答案：{r.answer[:50]}...")
        print()
