#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库初始化脚本

创建所有必要的表和索引
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectorstore import init_database

if __name__ == "__main__":
    print("正在初始化数据库...")
    init_database()
    print("数据库初始化完成！")
