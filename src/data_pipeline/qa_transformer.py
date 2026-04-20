#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA 格式转换器

将 RAG_Test 格式的 QA 数据转换为 FAQ 格式

RAG 格式: {"query": "...", "pos": ["..."], "neg": [...]}
FAQ 格式: {"question": "...", "answer": "...", "metadata": {...}}
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transform_rag_to_faq(
    input_file: str,
    output_file: str,
    source_doc: str = "王洪文传",
    category: str = "general"
) -> int:
    """
    将 RAG 格式 QA 转换为 FAQ 格式
    
    Args:
        input_file: 输入 JSONL 文件（RAG 格式）
        output_file: 输出 JSONL 文件（FAQ 格式）
        source_doc: 来源文档名称
        category: 默认类别
        
    Returns:
        int: 转换的记录数
    """
    count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                rag_record = json.loads(line)
                
                # 转换格式
                faq_record = {
                    "question": rag_record["query"],
                    "answer": rag_record["pos"][0] if rag_record.get("pos") else "",
                    "metadata": {
                        "source_doc": source_doc,
                        "category": category,
                        "confidence": 0.9,
                        "created_by": "auto",
                        "has_negative": len(rag_record.get("neg", [])) > 0
                    }
                }
                
                fout.write(json.dumps(faq_record, ensure_ascii=False) + "\n")
                count += 1
                
            except Exception as e:
                logger.warning(f"第 {line_num} 行转换失败: {e}")
    
    logger.info(f"转换完成: {count} 条记录")
    return count


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python qa_transformer.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    
    transform_rag_to_faq(sys.argv[1], sys.argv[2])
