# -*- coding: utf-8 -*-
"""
向量存储模块

基于 PostgreSQL + pgvector 的向量数据库操作模块。
提供连接池管理、数据索引、混合检索等功能。
"""

from .pg_pool import get_pool, close_pool, get_connection
from .pg_schema import init_database, create_tables
from .pg_indexer import FAQIndexer
from .pg_search import FAQSearcher, HybridSearcher

__all__ = [
    'get_pool',
    'close_pool', 
    'get_connection',
    'init_database',
    'create_tables',
    'FAQIndexer',
    'FAQSearcher',
    'HybridSearcher'
]