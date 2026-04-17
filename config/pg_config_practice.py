# -*- coding: utf-8 -*-
"""
PostgreSQL 数据库配置

本文件包含 PostgreSQL 连接参数和 pgvector 相关配置。
支持从环境变量读取配置，便于生产环境部署。
"""

import os

# =============================================================================
# 数据库连接配置
# =============================================================================
PG_HISTORICAL_TABLE = "HistoricalRAG"

# PostgreSQL 连接 URL
# 格式: postgresql://用户名:密码@主机:端口/数据库名?参数
# 可从环境变量 PG_URL 读取，默认使用本地开发配置
PG_URL = os.getenv(
    "PG_URL", 
    "postgresql://wesl:151083@localhost:5432/faq_db?sslmode=prefer&client_encoding=utf8"
)

# 数据库表名配置
# FAQ 主表名，存储标准问答对
PG_TABLE_NAME = os.getenv("PG_TABLE_NAME", "faq_knowledge")

# 文档片段表名，用于 RAG 检索
PG_DOC_TABLE = os.getenv("PG_DOC_TABLE", "doc_chunks")

# 对话历史表名
PG_CHAT_TABLE = os.getenv("PG_CHAT_TABLE", "chat_history")

# =============================================================================
# 向量配置
# =============================================================================

# 向量维度
# BGE-M3 模型输出 1024 维向量，建表时必须指定
VECTOR_DIM = 1024

# 批量插入批次大小
# 控制每次插入的数据量，避免内存溢出
# 对于 1024 维向量，100 条大约占用 400KB 内存
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

# =============================================================================
# 连接池配置
# =============================================================================

# 连接池最小连接数
# 保持空闲连接，避免频繁创建销毁
POOL_MIN_CONN = int(os.getenv("POOL_MIN_CONN", "1"))

# 连接池最大连接数
# 限制并发连接数，防止数据库过载
POOL_MAX_CONN = int(os.getenv("POOL_MAX_CONN", "10"))

# =============================================================================
# 索引配置
# =============================================================================

# HNSW 向量索引参数
# m: 图中每个节点的最大连接数，越大召回率越高，但索引越大
# ef_construction: 构建时的搜索范围，越大构建越慢但质量越高
HNSW_M = int(os.getenv("HNSW_M", "16"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "64"))

# HNSW 查询时的 ef 参数
# 越大搜索越慢但召回率越高，应大于 top_k
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))

# =============================================================================
# 路径配置
# =============================================================================

# 数据文件根目录
DATA_ROOT = os.getenv("DATA_ROOT", "./data")

# FAQ 数据路径
FAQ_DATA_PATH = os.path.join(DATA_ROOT, "qa_pairs", "wang_qa_fusion.jsonl")

# 原始文档路径
RAW_DOC_PATH = os.path.join(DATA_ROOT, "raw")

# 处理后文档路径
PROCESSED_PATH = os.path.join(DATA_ROOT, "processed")