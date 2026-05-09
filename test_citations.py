import os, sys, logging
logging.basicConfig(level=logging.WARNING)
os.chdir('/root/autodl-tmp/HistoricalFAQ-Bot')
sys.path.insert(0, '.')

from src.chat.chat_engine import ChatEngine

engine = ChatEngine(session_id="test_session")

questions = [
    "义仓的税率是多少",
    "唐太宗与魏征的关系如何",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"问题: {q}")
    print(f"{'='*60}")
    result = engine.achat(q)
    import asyncio
    r = asyncio.get_event_loop().run_until_complete(result)
    print(f"\n答案: {r.get('answer', '')[:500]}...")
    citations = r.get('citations', [])
    print(f"\n引用详情 ({len(citations)} 个):")
    for c in citations:
        title = c.get('chapter_title', '')
        doc = c.get('doc_name', '')
        cid = c.get('id', '')
        print(f"  [{cid}] 《{doc}》· {title}")
