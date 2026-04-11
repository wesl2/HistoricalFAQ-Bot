# -*- coding: utf-8 -*-
"""
数据处理模块

包含文档处理和 QA 转换功能
"""

from .qa_transformer import transform_rag_to_faq
from .document_processor import DocumentProcessor

__all__ = [
    "transform_rag_to_faq",
    "DocumentProcessor"
]
