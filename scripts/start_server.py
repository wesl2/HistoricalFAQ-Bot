#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速启动脚本

用法:
    python scripts/start_server.py           # 默认模式（启用 LangChain）
    python scripts/start_server.py --native  # 原生模式
    python scripts/start_server.py --advanced # 启用高级检索器
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="启动 HistoricalFAQ-Bot 服务")
    parser.add_argument(
        "--native", 
        action="store_true",
        help="使用原生模式（禁用 LangChain）"
    )
    parser.add_argument(
        "--no-advanced",
        action="store_true",
        help="禁用高级检索器（Multi-Query + Rerank）"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="服务主机地址（默认: 0.0.0.0）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务端口（默认: 8000）"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="启用热重载（开发模式）"
    )
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["USE_LANGCHAIN"] = "false" if args.native else "true"
    os.environ["USE_ADVANCED_RETRIEVER"] = "false" if args.no_advanced else "true"
    os.environ["API_HOST"] = args.host
    os.environ["API_PORT"] = str(args.port)
    os.environ["API_RELOAD"] = "true" if args.reload else "false"
    
    print("=" * 60)
    print("HistoricalFAQ-Bot 启动配置")
    print("=" * 60)
    print(f"LangChain 集成: {'禁用' if args.native else '启用'}")
    print(f"高级检索器: {'禁用' if args.no_advanced else '启用'}")
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"热重载: {'启用' if args.reload else '禁用'}")
    print("=" * 60)
    print()
    
    # 启动服务
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
