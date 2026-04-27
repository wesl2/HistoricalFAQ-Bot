# -*- coding: utf-8 -*-
"""
标准 LangChain Retriever 封装（PGVector 版 - 含 RRF 融合排序）

使用 PostgreSQL + pgvector 作为向量存储（公司级实践）
不使用 Chroma/FAISS，完全基于 PGVector
"""

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any, Optional
import numpy as np
import hashlib
import re
from collections import defaultdict

from src.vectorstore.pg_pool import get_connection
from src.vectorstore.pg_search import FAQSearcher, HybridSearcher
from config.pg_config import PG_TABLE_NAME, PG_DOC_TABLE
from config.model_config_practice import EMBEDDING_CONFIG

logger = None


def get_logger():
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    return logger


def _norm_text(t: str) -> str:
    """规范化文本：去空白、转小写，用于计算指纹"""
    t = (t or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _hash_text(t: str) -> str:
    """计算文本 SHA1 指纹，用于 RRF 融合时的去重与对齐"""
    return hashlib.sha1(_norm_text(t).encode("utf-8")).hexdigest()


class PGVectorRetriever(BaseRetriever):
    """
    PGVector 检索器
    直接使用 PostgreSQL + pgvector，不使用 Chroma/FAISS
    """

    search_kwargs: Dict[str, Any] = {"top_k": 5}
    search_type: str = "vector"  # "vector", "faq", "hybrid"
    _embeddings: Optional[Any] = None  # 缓存 embedding 实例

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        标准检索接口（BaseRetriever 要求实现）
        直接使用项目自研的 pg_search 模块
        """
        embeddings = self._get_embeddings()
        query_vector = embeddings.embed_query(query)

        if self.search_type == "faq":
            return self._search_faq(query_vector)
        elif self.search_type == "hybrid":
            return self._search_hybrid(query_vector, query)
        else:
            return self._search_docs(query_vector)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步检索接口"""
        return self._get_relevant_documents(query)

    def _get_embeddings(self):
        """获取嵌入模型（懒加载 + 缓存，避免重复初始化）"""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_CONFIG["model_path"],
                model_kwargs={"device": EMBEDDING_CONFIG["device"]},
                encode_kwargs={"normalize_embeddings": EMBEDDING_CONFIG["normalize"]}
            )
        return self._embeddings

    def _search_faq(self, query_vector: List[float]) -> List[Document]:
        """FAQ 检索（使用 pg_search）"""
        searcher = FAQSearcher()
        results = searcher.search(
            query_vector,
            top_k=self.search_kwargs.get("top_k", 5)
        )

        docs = []
        for r in results:
            docs.append(Document(
                page_content=r.answer,
                metadata={
                    "type": "faq",
                    "id": r.id,
                    "question": r.question,
                    "source": r.source_doc or "unknown",
                    "score": r.score
                }
            ))

        get_logger().info(f"FAQ 检索完成：找到 {len(docs)} 条记录")
        return docs

    def _search_docs(self, query_vector: List[float]) -> List[Document]:
        """文档检索（使用 pg_search）"""
        hybrid_searcher = HybridSearcher()
        doc_results = hybrid_searcher._search_docs(
            query_vector,
            top_k=self.search_kwargs.get("top_k", 5)
        )

        docs = []
        for r in doc_results:
            docs.append(Document(
                page_content=r.answer,
                metadata={
                    "type": "doc",
                    "id": r.id,
                    "source": r.source_doc or "unknown",
                    "page": r.source_page,
                    "score": r.score
                }
            ))

        get_logger().info(f"文档检索完成：找到 {len(docs)} 条记录")
        return docs

    def _search_hybrid(self, query_vector: List[float], query_text: str) -> List[Document]:
        """
        混合检索（双路召回 + RRF 融合排序 + 跨源去重）
        
        工业界实践（落地版）：
        1. 扩大召回：每路召回 Top-K * 2，给融合留空间
        2. RRF 融合：基于内容 Hash 对齐 FAQ 和 Doc，计算 1/(K+rank)
        3. 智能去重：如果 FAQ 答案和 Doc 片段高度相似，优先保留结构化更好的 FAQ
        """
        top_k = self.search_kwargs.get("top_k", 5)
        recall_k = top_k * 2  # 扩大召回因子

        hybrid_searcher = HybridSearcher()
        results = hybrid_searcher.hybrid_search(
            query_vector,
            query_text,
            top_k=recall_k,
        )
        
        faq_items = results.get("faq", [])
        doc_items = results.get("doc", [])

        # ========== RRF 融合计算 ==========
        K = 60  # 经典推荐值
        rrf_scores = defaultdict(float)
        best_item = {}  # key -> 代表 Item (优先存 FAQ)
        ranks = {}      # key -> {faq_rank, doc_rank}

        # 1. FAQ 排名贡献
        for idx, r in enumerate(faq_items, start=1):
            # 使用 Content Hash 作为统一 Key，实现跨源去重
            key = _hash_text(r.answer)
            rrf_scores[key] += 1.0 / (K + idx)
            ranks.setdefault(key, {})["faq_rank"] = idx
            
            # FAQ 信息更全，优先作为 Best Item
            if key not in best_item:
                best_item[key] = ("faq", r)

        # 2. Doc 排名贡献
        for idx, r in enumerate(doc_items, start=1):
            key = _hash_text(r.answer) # Doc 的 chunk_text 存在 answer 字段
            rrf_scores[key] += 1.0 / (K + idx)
            ranks.setdefault(key, {})["doc_rank"] = idx
            
            # 如果 FAQ 没出现过，才存入
            if key not in best_item:
                best_item[key] = ("doc", r)

        # 3. 统一排序 & 截断
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        docs = []
        for k in sorted_keys[:top_k]:
            source_type, r = best_item[k]
            
            meta = {
                "type": source_type,
                "id": r.id,
                "source": getattr(r, "source_doc", None) or "unknown",
                "rrf_score": rrf_scores[k],
                "faq_rank": ranks.get(k, {}).get("faq_rank"),
                "doc_rank": ranks.get(k, {}).get("doc_rank"),
                "content_hash": k,
            }
            
            # 补充特定字段
            if source_type == "faq":
                meta["question"] = getattr(r, "question", "")
            if source_type == "doc":
                meta["page"] = getattr(r, "source_page", None)
            
            docs.append(Document(page_content=r.answer, metadata=meta))

        get_logger().info(
            f"RRF 混合检索完成：FAQ={len(faq_items)}, DOC={len(doc_items)} -> 输出={len(docs)}"
        )
        return docs

    def add_documents(self, documents: List[Document]):
        """添加文档到 PGVector（使用 pg_indexer）"""
        from src.vectorstore.pg_indexer import FAQIndexer
        import json
        import tempfile
        import os

        # 修复：embeddings 放到循环外，避免重复初始化
        embeddings = self._get_embeddings()

        # 转换为 FAQIndexer 需要的格式
        faq_data = []
        for doc in documents:
            vector = embeddings.embed_query(doc.page_content)

            faq_data.append({
                "question": doc.metadata.get("question", doc.page_content[:50]),
                "similar_question": doc.metadata.get("question", doc.page_content[:50]),
                "answer": doc.page_content,
                "vector": vector,
                "metadata": {
                    "source_doc": doc.metadata.get("source", "unknown"),
                    "category": doc.metadata.get("category", "general")
                }
            })

        # 写入临时 JSONL 文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            for item in faq_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_file = f.name

        try:
            indexer = FAQIndexer()
            count = indexer.index_from_file(temp_file)
            get_logger().info(f"添加 {count} 个文档到 PGVector")
        finally:
            os.unlink(temp_file)


# 全局缓存
_retriever_cache = {}


def get_pgvector_retriever(**kwargs) -> PGVectorRetriever:
    """获取 PGVector Retriever 实例（单例）"""
    key = str(sorted(kwargs.items()))
    if key not in _retriever_cache:
        _retriever_cache[key] = PGVectorRetriever(**kwargs)
    return _retriever_cache[key]
