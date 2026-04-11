# -*- coding: utf-8 -*-
"""
配置模块

本模块包含项目的所有配置信息，包括数据库连接、模型路径、LLM配置等。
所有配置都可以通过环境变量覆盖，便于不同环境部署。
"""

from .pg_config import PG_URL, PG_TABLE_NAME, VECTOR_DIM, BATCH_SIZE
from .model_config import LLM_CONFIG, EMBEDDING_CONFIG
from .retrieval_config import RETRIEVAL_CONFIG

__all__ = [
    'PG_URL',
    'PG_TABLE_NAME', 
    'VECTOR_DIM',
    'BATCH_SIZE',
    'LLM_CONFIG',
    'EMBEDDING_CONFIG',
    'RETRIEVAL_CONFIG'
]