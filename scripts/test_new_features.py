#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试新功能脚本

测试内容：
1. 高级检索器（Multi-Query + Rerank）—— 已废弃
2. 流式输出
3. 监控指标

【修改留痕 - 2024-04-17】
- 高级检索器（advanced_retriever）模块在重构中已移除，
  test_advanced_retriever() 改为直接提示不可用，不再尝试导入。
- 其余 test_streaming、test_callbacks 逻辑保持不变。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_advanced_retriever():
    """测试高级检索器"""
    print("\n" + "=" * 60)
    print("测试 1: 高级检索器")
    print("=" * 60)
    
    # 2024-04-17 修复：advanced_retriever 模块已移除，该测试直接跳过
    print("⚠ 高级检索器（advanced_retriever）在重构中已移除，本测试跳过")
    print("  提示：如需测试检索功能，请使用 scripts/test_bm25.py 或新的标准检索器")
    return False


def test_streaming():
    """测试流式输出"""
    print("\n" + "=" * 60)
    print("测试 2: 流式输出")
    print("=" * 60)
    
    try:
        from src.chat.chat_engine import ChatEngine
        
        engine = ChatEngine(use_langchain=True)
        
        query = "王洪文"
        print(f"查询: {query}")
        print("流式输出: ", end="", flush=True)
        
        # 模拟流式输出（FAQ 模式会按句子分割）
        for chunk in engine.stream(query):
            print(chunk, end="", flush=True)
            time.sleep(0.1)  # 模拟延迟
        
        print("\n✓ 流式输出测试完成")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_callbacks():
    """测试监控指标"""
    print("\n" + "=" * 60)
    print("测试 3: 监控指标")
    print("=" * 60)
    
    try:
        from src.rag.callbacks import get_callback_manager
        
        manager = get_callback_manager()
        
        # 模拟一些事件
        logging_callback = manager.logging_callback
        logging_callback.on_chain_start(
            {"name": "test_chain"},
            {"input": "测试输入"}
        )
        logging_callback.on_llm_start(
            {"name": "test_llm"},
            ["测试提示词"]
        )
        logging_callback.on_llm_end(
            type('obj', (object,), {
                'llm_output': {'token_usage': {'total_tokens': 100}},
                'generations': [[type('obj', (object,), {'text': '测试输出'})()]]
            })()
        )
        
        # 获取指标
        metrics = manager.get_metrics()
        print(f"性能指标: {metrics['performance']}")
        print(f"Token 使用: {metrics['token_usage']}")
        print("✓ 监控指标测试完成")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoints():
    """测试 API 端点"""
    print("\n" + "=" * 60)
    print("测试 4: API 端点（需要服务已启动）")
    print("=" * 60)
    
    try:
        import requests
        
        base_url = "http://localhost:8000"
        
        # 测试健康检查
        response = requests.get(f"{base_url}/api/health")
        print(f"健康检查: {response.status_code}")
        print(f"  响应: {response.json()}")
        
        # 测试查询
        response = requests.post(
            f"{base_url}/api/query",
            json={"question": "王洪文是谁？"}
        )
        print(f"\n查询接口: {response.status_code}")
        result = response.json()
        print(f"  回答: {result.get('answer', 'N/A')[:100]}...")
        print(f"  检索类型: {result.get('search_type', 'N/A')}")
        print(f"  置信度: {result.get('confidence', 'N/A')}")
        
        # 测试监控指标
        response = requests.get(f"{base_url}/api/metrics")
        print(f"\n监控指标: {response.status_code}")
        print(f"  指标: {response.json()}")
        
        print("\n✓ API 端点测试完成")
        return True
        
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到服务，请先启动服务: python scripts/start_server.py")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("HistoricalFAQ-Bot 新功能测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("高级检索器", test_advanced_retriever()))
    results.append(("流式输出", test_streaming()))
    results.append(("监控指标", test_callbacks()))
    
    # 询问是否测试 API
    print("\n" + "-" * 60)
    response = input("是否测试 API 端点？(需要先启动服务) [y/N]: ")
    if response.lower() == 'y':
        results.append(("API 端点", test_api_endpoints()))
    
    # 打印结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    print(f"\n总计: {passed_count}/{total_count} 通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
