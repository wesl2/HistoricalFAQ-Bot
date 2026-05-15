import os, sys, logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
os.chdir('/root/autodl-tmp/HistoricalFAQ-Bot')
sys.path.insert(0, '.')

from src.retrieval.doc_retriever_practice import DocRetriever

retriever = DocRetriever(
    top_k=10,
    use_bm25=True,
    fusion_method="rrf",
    rrf_k=60,
    enable_multi_query=True,
    multi_query_count=3,
)

query = "唐太宗在位期间有哪些重要的军事征服？"
results = retriever.retrieve(query)

print(f"\n{'='*60}")
print(f"MultiQuery 返回 {len(results)} 个结果")
print(f"{'='*60}")

for i, r in enumerate(results, 1):
    has_tuguhun = "吐谷浑" in r.content
    has_gaocang = "高昌" in r.content
    has_tujue = "突厥" in r.content
    has_yanqi = "焉耆" in r.content
    has_qiuci = "龟兹" in r.content
    has_xueyantuo = "薛延陀" in r.content
    
    print(f"\n[{i}] ID={r.id} | doc={r.doc_name} | page={r.doc_page}")
    print(f"    RRF={r.rrf_score:.4f} | v_rank={r.vector_rank} | b_rank={r.bm25_rank}")
    print(f"    战役标记: 突厥={has_tujue} 吐谷浑={has_tuguhun} 高昌={has_gaocang} 焉耆={has_yanqi} 龟兹={has_qiuci} 薛延陀={has_xueyantuo}")
    # 显示 content 前200字和后200字（因为parent_text很长）
    content_preview = r.content[:200].replace('\n', ' ')
    print(f"    内容开头: {content_preview}...")
