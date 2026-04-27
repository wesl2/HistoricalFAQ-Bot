# -*- coding: utf-8 -*-
"""
Embedding 模块

提供文本向量化功能，支持 BGE-M3 模型
"""

#from .embedding_local import get_embedding, compute_embedding
from .embedding_local_practice import get_embedding, compute_embedding

__all__ = ['get_embedding', 'compute_embedding']
