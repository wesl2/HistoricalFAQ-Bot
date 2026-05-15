#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 端到端调试脚本
==================
功能：独立测试《唐太宗传》导入后的完整 RAG 链路

用法：
    python debug_rag.py
    
环境要求：
    - PostgreSQL 运行中（faq_db）
    - BGE-M3 模型已下载
    - DeepSeek API Key 已配置（下方可直接修改）
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Any

# =============================================================================
# 第一部分：配置（请按需修改）
# =============================================================================

# DeepSeek API Key（必填，否则 LLM 生成会失败）
# 也可以从环境变量读取：os.environ.get("API_KEY", "")
API_KEY = "sk-d722d9cb7edf49b5b9fe88bb37908162"
API_PROVIDER = "deepseek"  # 可选: deepseek, openai, anthropic, zhipu, moonshot

# 测试问题列表
TEST_QUESTIONS = [
    #"唐太宗李世民的玄武门之变发生在哪一年？",
    #"唐太宗和魏征的关系如何？",
    #"贞观之治的主要政策措施有哪些？",
    #"唐太宗在位期间有哪些重要的军事征服？",
    #"唐太宗的用人之道有什么特点？",
    #"义仓的税率是多少",            # 事实型
    #"三省六部制与均田制的关系",     # 推理型
    #"长孙无忌和李世民的关系是什么样子的？",
    #"李世民作为天可汗，如何统治周围的异族和国家？",
    #"李世民一生最大的成就是什么？",
    #"李世民对当时全球除了中国外其他国家和地区的影响是什么？"
    #"李世民在隋唐的地位？",
    #"李世民的文治成就主要体现在什么部分，着重从制度建设方面讲讲。"
    #"电视剧《贞观之治》里，长孙无忌有一句经典台词：“八百人就八百人，八百人先下手为强！”，这最有可能出自什么历史事件？",
    #"李世民在‘渭水之盟’与突厥颉利可汗对峙时，采取了哪几项关键手段来迫使突厥退兵？请逐项列出",
    "李世民在玄武门之变前经历了什么样的政治斗争？为什么他选择了杀兄逼父亲？这种行为导致什么后果？"
]

# Multi-Query 开关
ENABLE_MULTI_QUERY = True   # 开启多维度召回
MULTI_QUERY_COUNT = 3       # 扩展 query 数量

# 检索参数
TOP_K = 5  # 显示前 K 个检索结果
VECTOR_TOP_K = 10  # 向量检索内部 top_k
BM25_TOP_K = 15  # BM25 检索内部 top_k

# LLM 参数
LLM_TIMEOUT = 60  # 秒

# =============================================================================
# 第二部分：初始化环境
# =============================================================================

# 设置环境变量
os.environ["API_KEY"] = API_KEY
os.environ["API_PROVIDER"] = API_PROVIDER
os.environ["LLM_MODE"] = "api"  # 强制使用 API 模式，跳过 local
os.environ["MULTI_QUERY_ENABLED"] = "true" if ENABLE_MULTI_QUERY else "false"
os.environ["MULTI_QUERY_COUNT"] = str(MULTI_QUERY_COUNT)
os.environ["LOCAL_LLM_TIMEOUT"] = str(LLM_TIMEOUT)

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 70)
print("RAG 端到端调试脚本")
print("=" * 70)
print(f"API Provider: {API_PROVIDER}")
print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
print(f"测试问题数: {len(TEST_QUESTIONS)}")
print("=" * 70)
print()

# =============================================================================
# 第三部分：组件测试
# =============================================================================

def test_embedding():
    """测试 Embedding 模型加载"""
    print("【测试 1/4】Embedding 模型加载")
    print("-" * 50)
    try:
        from src.embedding.embedding_local_practice import get_embedding
        vec = get_embedding("唐太宗")
        print(f"✅ Embedding 模型加载成功")
        print(f"   向量维度: {len(vec)}")
        print(f"   前5维: {vec[:5]}")
        print()
        return True
    except Exception as e:
        print(f"❌ Embedding 模型加载失败: {e}")
        print()
        return False


def test_database():
    """测试数据库连接和文档数量"""
    print("【测试 2/4】数据库连接")
    print("-" * 50)
    try:
        from src.vectorstore.pg_pool_practice import get_connection
        with get_connection() as conn:
            cur = conn.cursor()
            
            # 查总记录数
            cur.execute("SELECT COUNT(*) FROM doc_chunks")
            total = cur.fetchone()[0]
            print(f"✅ 数据库连接成功")
            print(f"   doc_chunks 总记录数: {total}")
            
            # 查文档名称
            cur.execute("SELECT DISTINCT doc_name FROM doc_chunks LIMIT 5")
            docs = [row[0] for row in cur.fetchall()]
            print(f"   文档列表: {docs}")
            
            # 查 parent_text 非空率
            cur.execute("SELECT COUNT(*) FROM doc_chunks WHERE parent_text IS NOT NULL")
            with_parent = cur.fetchone()[0]
            print(f"   含 parent_text: {with_parent}/{total}")
            
            cur.close()
        print()
        return True
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        print()
        return False


def test_retrieval(query: str):
    """测试检索链路"""
    print(f"【检索测试】查询: {query}")
    print("-" * 50)
    
    try:
        from src.retrieval.doc_retriever_practice import DocRetriever
        from src.retrieval.search_router_practice import SearchRouter
        from src.embedding.embedding_local_practice import get_embedding
        from src.vectorstore.pg_pool_practice import get_connection
        
        # 1. 向量检索
        print("1. 向量检索...")
        vec = get_embedding(query)
        vec_str = '[' + ','.join(str(v) for v in vec) + ']'
        
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT chunk_text, parent_text, doc_name, doc_page,
                       1 - (chunk_vector <=> %s::vector) as sim
                FROM doc_chunks 
                ORDER BY chunk_vector <=> %s::vector 
                LIMIT %s
            ''', (vec_str, vec_str, VECTOR_TOP_K))
            
            vector_results = []
            for row in cur.fetchall():
                vector_results.append({
                    "chunk_text": row[0][:150] + "..." if len(row[0]) > 150 else row[0],
                    "parent_text": row[1][:200] + "..." if row[1] and len(row[1]) > 200 else row[1],
                    "doc_name": row[2],
                    "doc_page": row[3],
                    "similarity": round(float(row[4]), 4)
                })
            cur.close()
        
        print(f"   找到 {len(vector_results)} 个向量结果")
        for i, r in enumerate(vector_results[:TOP_K], 1):
            print(f"   [{i}] 相似度:{r['similarity']} | {r['chunk_text'][:80]}...")
        
        # 2. BM25 检索
        print("\n2. BM25 检索...")
        retriever = DocRetriever(top_k=BM25_TOP_K)
        bm25_results = retriever.bm25_retriever.retrieve(query)
        print(f"   找到 {len(bm25_results)} 个 BM25 结果")
        for i, r in enumerate(bm25_results[:TOP_K], 1):
            content = r.content if hasattr(r, 'content') else str(r)
            print(f"   [{i}] {content[:80]}...")
        
        # 3. RRF 融合
        print("\n3. RRF 融合...")
        hybrid_results = retriever.retrieve(query)
        print(f"   融合后 {len(hybrid_results)} 个结果")
        for i, r in enumerate(hybrid_results[:TOP_K], 1):
            content = r.content if hasattr(r, 'content') else str(r)
            print(f"   [{i}] {content[:80]}...")
        
        print()
        return hybrid_results
        
    except Exception as e:
        print(f"❌ 检索失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return []


async def test_chat_engine(question: str):
    """测试 ChatEngine 完整链路"""
    print(f"【ChatEngine 测试】问题: {question}")
    print("-" * 50)
    
    try:
        from src.chat.chat_engine import ChatEngine
        
        engine = ChatEngine(session_id="debug_session_001")
        result = await engine.achat(question)
        
        print(f"✅ ChatEngine 调用完成")
        print(f"   答案: {result.get('answer', 'N/A')}")
        print(f"   引用数: {len(result.get('citations', []))}")
        print(f"   Session ID: {result.get('session_id', 'N/A')}")
        
        if result.get('error_code'):
            print(f"   错误码: {result['error_code']}")
        
        # 打印引用详情 + 校验结果
        citations = result.get('citations', [])
        if citations:
            print(f"\n   引用详情（共 {len(citations)} 条）:")
            for i, c in enumerate(citations, 1):
                print(f"\n   ─── 引用 [{i}] ───")
                print(f"      来源ID: {c.get('id', 'N/A')}")
                print(f"      类型: {c.get('type', 'N/A')}")
                if c.get('type') == 'doc':
                    print(f"      文档: {c.get('doc_name', 'N/A')}")
                    content = c.get('content', '')
                    preview = content[:300].replace('\n', ' ')
                    print(f"      内容: {preview}{'...' if len(content) > 300 else ''}")
                elif c.get('type') == 'faq':
                    print(f"      问题: {c.get('question', 'N/A')}")
                    answer = c.get('answer', '')
                    preview = answer[:300].replace('\n', ' ')
                    print(f"      答案: {preview}{'...' if len(answer) > 300 else ''}")
            
            # 引用一致性校验（生产级关键检查）
            from src.chat.citation_verifier import verify_citations, format_issues
            # 从 sources 重建 source_map 做校验
            source_map = {str(c.get('id', i)): c for i, c in enumerate(citations, 1)}
            issues = verify_citations(result['answer'], source_map)
            if issues:
                print(f"\n   ⚠️  引用校验发现问题:")
                for issue in issues:
                    print(f"      [{issue.severity}] 引用 [{issue.citation_id}] 不匹配 | {issue.reason[:80]}")
            else:
                print(f"\n   ✅ 引用一致性校验通过")
        
        print()
        return result
        
    except Exception as e:
        print(f"❌ ChatEngine 调用失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return None


# =============================================================================
# 第四部分：主流程
# =============================================================================

def main():
    """主流程：逐步测试各组件"""
    
    # 1. Embedding 测试
    if not test_embedding():
        print("Embedding 测试失败，终止调试")
        return
    
    # 2. 数据库测试
    if not test_database():
        print("数据库测试失败，终止调试")
        return
    
    # 3. 检索测试（用第一个问题）
    print("【测试 3/4】检索链路")
    print("=" * 70)
    retrieval_results = test_retrieval(TEST_QUESTIONS[0])
    
    if not retrieval_results:
        print("检索测试失败，跳过 ChatEngine 测试")
        return
    
    # 4. ChatEngine 测试
    print("【测试 4/4】ChatEngine 端到端")
    print("=" * 70)
    
    async def run_chat_tests():
        for i, question in enumerate(TEST_QUESTIONS, 1):
            print(f"\n{'='*70}")
            print(f"问题 {i}/{len(TEST_QUESTIONS)}")
            print(f"{'='*70}")
            await test_chat_engine(question)
    
    asyncio.run(run_chat_tests())
    
    print("\n" + "=" * 70)
    print("调试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
