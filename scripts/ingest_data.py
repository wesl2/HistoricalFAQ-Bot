#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据导入脚本

将 FAQ 数据导入数据库
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectorstore import FAQIndexer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导入 FAQ 数据")
    parser.add_argument("file", help="JSONL 文件路径")
    parser.add_argument("--clear", action="store_true", help="清空现有数据")
    args = parser.parse_args()
    
    print(f"正在导入数据: {args.file}")
    indexer = FAQIndexer()
    count = indexer.index_from_file(args.file, clear_existing=args.clear)
    print(f"成功导入 {count} 条记录")
