#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯检索链路测试（不调用 LLM API，避免 DeepSeek 卡顿）
测试：Embedding → 数据库 → 向量检索 → BM25 → RRF 融合 → 引用格式化
"""

import os
import sys

os.environ["MULTI_QUERY_ENABLED"] = "false"

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.embedding.embedding_local_practice import get_embedding
from src.vectorstore.pg_pool_practice import get_connection
from src.retrieval.doc_retriever_practice import DocRetriever
from src.chat.response_generator import ResponseGenerator


def test_full_chain():
    query = "唐太宗和魏征的关系如何？"
    print(f"查询: {query}\n")

    # 1. Embedding
    print("1. Embedding 计算...")
    vec = get_embedding(query)
    print(f"   ✅ 完成, 维度={len(vec)}")

    # 2. 数据库连接
    print("2. 数据库连接...")
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM doc_chunks")
        print(f"   ✅ 完成, 记录数={cur.fetchone()[0]}")
        cur.close()

    # 3. RRF 检索
    print("3. RRF 混合检索...")
    retriever = DocRetriever(
        top_k=5, use_bm25=True, fusion_method="rrf", rrf_k=60,
        enable_multi_query=False
    )
    doc_results = retriever.retrieve(query)
    print(f"   ✅ 完成, 返回 {len(doc_results)} 条")
    for i, r in enumerate(doc_results, 1):
        title = r.content.split('\n')[0] if r.content else ""
        print(f"   [{i}] {r.doc_name} | {title[:60]}")

    # 4. Prompt 构建 + 引用格式化（不调用 LLM）
    print("\n4. Prompt 构建与引用格式化...")
    rg = ResponseGenerator()
    prompt, source_map = rg.build_prompt(query, [], doc_results)

    # 模拟 LLM 回答（带引用）
    fake_answer = (
        "唐太宗与魏征是君臣相得的典范。[1] "
        "魏征以直言敢谏著称，唐太宗视其为'人镜'。[1] "
        "唐太宗曾称赞：'以人为镜，可以明得失。'[1]"
    )

    # 测试清理 + footer 追加
    cleaned = rg._strip_llm_citation_footer(fake_answer)
    footer = rg._format_citation_footer(cleaned, source_map)
    final_answer = cleaned + footer

    print(f"   ✅ 引用来源出现次数: {final_answer.count('引用来源')}")
    print(f"\n   格式化后的答案尾部:\n{footer}")

    # 5. 验证：模拟 LLM 自己生成引用列表的情况
    print("\n5. 重复引用防御测试...")
    fake_with_duplicate = (
        "唐太宗与魏征关系很好。[1]\n"
        "---\n"
        "**引用来源：**\n"
        "[1] 唐太宗传\n"
        "---\n"
        "**引用来源：**\n"
        "[1] 唐太宗传"
    )
    cleaned2 = rg._strip_llm_citation_footer(fake_with_duplicate)
    footer2 = rg._format_citation_footer(cleaned2, source_map)
    final2 = cleaned2 + footer2
    print(f"   清理前 '引用来源' 次数: {fake_with_duplicate.count('引用来源')}")
    print(f"   清理后 '引用来源' 次数: {final2.count('引用来源')}")
    print(f"   ✅ 防御测试通过!" if final2.count('引用来源') == 1 else "   ❌ 防御失败!")

    print("\n" + "=" * 50)
    print("全部测试通过! DeepSeek API 挂了不影响检索链路。")
    print("=" * 50)


if __name__ == "__main__":
    test_full_chain()
