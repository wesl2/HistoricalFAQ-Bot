#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BM25 检索测试脚本

测试内容：
1. 纯向量检索
2. 纯 BM25 检索
3. 向量 + BM25 混合检索
4. 性能对比
"""

import sys
import time
sys.path.insert(0, '.')

from src.retrieval.doc_retriever_practice import DocRetriever
from config.model_config_practice import BM25_CONFIG


def print_results(title, results, show_details=True):
    """打印检索结果"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    if not results:
        print("  未找到结果")
        return
    
    print(f"  找到 {len(results)} 个结果\n")
    
    for i, r in enumerate(results, 1):
        print(f"  [{i}] 分数：{r.similarity:.3f} 类型：{r.category}")
        print(f"      来源：{r.doc_name}")
        if show_details:
            if r.vector_score is not None:
                print(f"      向量：{r.vector_score:.3f}, BM25: {r.bm25_score:.3f}")
            print(f"      内容：{r.content[:80]}...")
        print()


def test_retrieval_methods():
    """测试不同检索方法"""
    
    # 测试查询（覆盖不同场景）
    test_queries = [
        ("玄武门之变", "关键词查询（BM25 优势场景）"),
        ("李世民如何治国", "语义查询（向量优势场景）"),
        ("贞观之治有哪些成就", "混合查询"),
        ("王洪文生平", "关键词 + 语义混合"),
    ]
    
    print("="*60)
    print("BM25 检索系统测试")
    print("="*60)
    print(f"配置：BM25 启用={BM25_CONFIG['enabled']}, 权重={BM25_CONFIG['weight']}")
    
    for query, description in test_queries:
        print(f"\n{'#'*60}")
        print(f"# 查询：{query}")
        print(f"# 类型：{description}")
        print(f"{'#'*60}")
        
        # 1. 纯向量检索
        start = time.time()
        retriever_vector = DocRetriever(top_k=5, use_bm25=False)
        results_v = retriever_vector.retrieve(query)
        time_v = time.time() - start
        
        print_results(f"【向量检索】(耗时：{time_v*1000:.1f}ms)", results_v, show_details=False)
        
        # 2. 纯 BM25 检索
        start = time.time()
        retriever_bm25 = DocRetriever(top_k=5, use_bm25=True)
        retriever_bm25.bm25_weight = 1.0  # 纯 BM25
        results_b = retriever_bm25.retrieve_hybrid(query)
        # 过滤出纯 BM25 结果
        results_b = [r for r in results_b if r.category == "bm25"][:5]
        time_b = time.time() - start
        
        print_results(f"【BM25 检索】(耗时：{time_b*1000:.1f}ms)", results_b, show_details=False)
        
        # 3. 混合检索
        start = time.time()
        retriever_hybrid = DocRetriever(top_k=5, use_bm25=True, bm25_weight=BM25_CONFIG['weight'])
        results_h = retriever_hybrid.retrieve(query)
        time_h = time.time() - start
        
        print_results(f"【混合检索】(耗时：{time_h*1000:.1f}ms)", results_h, show_details=True)
    
    # 性能对比总结
    print(f"\n{'='*60}")
    print("性能对比总结")
    print(f"{'='*60}")
    print(f"查询：'玄武门之变'")
    print(f"  向量检索：{time_v*1000:.1f}ms")
    print(f"  BM25 检索：{time_b*1000:.1f}ms")
    print(f"  混合检索：{time_h*1000:.1f}ms")
    print(f"\n建议：")
    print(f"  - 关键词查询：使用 BM25 或混合检索")
    print(f"  - 语义查询：使用向量检索")
    print(f"  - 通用场景：使用混合检索（向量 0.7 + BM25 0.3）")


def test_bm25_standalone():
    """测试独立的 BM25 检索器"""
    from src.retrieval.bm25_retriever_practice import BM25Retriever
    
    print(f"\n{'='*60}")
    print("独立 BM25 检索器测试")
    print(f"{'='*60}")
    
    retriever = BM25Retriever(top_k=5)
    
    queries = ["李世民", "玄武门之变", "贞观之治"]
    
    for query in queries:
        print(f"\n查询：{query}")
        results = retriever.retrieve(query)
        
        for r in results:
            print(f"  [{r.score:.3f}] {r.doc_name} 第{r.doc_page}页")
            print(f"       {r.content[:60]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BM25 检索测试")
    parser.add_argument("--standalone", action="store_true", help="只测试独立 BM25 检索器")
    parser.add_argument("--query", type=str, help="自定义测试查询")
    args = parser.parse_args()
    
    if args.standalone:
        test_bm25_standalone()
    elif args.query:
        # 自定义查询测试
        retriever = DocRetriever(top_k=5, use_bm25=True)
        results = retriever.retrieve(args.query)
        print_results(f"自定义查询：{args.query}", results)
    else:
        # 完整测试
        test_retrieval_methods()
    
    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")
