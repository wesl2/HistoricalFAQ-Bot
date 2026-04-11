#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话演示脚本

简单的命令行交互界面
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.chat.chat_engine import ChatEngine


def main():
    print("=" * 60)
    print("历史人物 FAQ 问答 Bot")
    print("输入 'quit' 退出，输入 'mode local/api' 切换模式")
    print("=" * 60)
    
    engine = ChatEngine()
    
    while True:
        print()
        query = input("你: ").strip()
        
        if not query:
            continue
        
        if query.lower() == 'quit':
            print("再见！")
            break
        
        if query.startswith('mode '):
            mode = query.split()[1]
            engine = ChatEngine(llm_mode=mode)
            print(f"已切换到 {mode} 模式")
            continue
        
        # 处理查询
        try:
            result = engine.chat(query)
            print(f"\nBot: {result['answer']}")
            print(f"\n[检索类型: {result['search_type']}, 置信度: {result['confidence']:.3f}]")
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
