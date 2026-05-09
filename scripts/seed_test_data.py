#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插入李世民测试数据（PDR 模式）

- parent_text: 完整段落（给 LLM）
- chunk_text: 小块（用于向量检索）
- chunk_vector: 小块的 embedding
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding.embedding_local_practice import get_embedding
from src.vectorstore.pg_pool_practice import get_connection
from config.pg_config_practice import PG_DOC_TABLE


def split_into_children(text: str, chunk_size: int = 150, overlap: int = 30):
    """按字符数切分 child（简单实现）"""
    children = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        children.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return children


# ========== 测试数据：7 个 parent，约 10 个 child ==========

parents = [
    {
        "doc_name": "唐太宗传_测试",
        "doc_page": 1,
        "text": (
            "李世民（598年—649年），祖籍陇西成纪，是唐高祖李渊和窦皇后的次子。"
            "隋朝末年，天下大乱，李世民跟随父亲李渊在太原起兵反隋。"
            "他多次指挥重大战役，表现出卓越的军事才能，为李唐家族的崛起立下汗马功劳。"
            "特别是在太原起兵后，李世民在攻占长安、平定关中等战役中发挥了关键作用。"
            "唐朝建立后，李世民官居尚书令、右武候大将军，受封为秦国公，后晋封为秦王。"
        ),
    },
    {
        "doc_name": "唐太宗传_测试",
        "doc_page": 2,
        "text": (
            "唐朝建立之初，李渊封长子李建成为太子，次子李世民为秦王。"
            "随着李世民在军事和政治上的功绩日益显著，太子李建成开始排挤李世民。"
            "李建成向李渊建议由亲信李元吉做统帅出征突厥，目的是掌握兵权并趁机除掉李世民。"
            "李世民在危急时先发制人，决定发动政变夺取政权。"
            "他精心策划，选择了玄武门这个战略要地作为伏击地点。"
        ),
    },
    {
        "doc_name": "唐太宗传_测试",
        "doc_page": 3,
        "text": (
            "公元626年7月2日，武德九年六月初四，李世民在玄武门发动政变。"
            "他亲自率领亲信将领，伏击并杀死了皇太子李建成和齐王李元吉。"
            "事后，李世民杀李建成、李元吉诸子，并将他们从宗籍中除名。"
            "三天后，唐高祖李渊宣布立秦王李世民为太子，军国事务悉数委任太子处决。"
            "同年9月4日，李渊退位称太上皇，李世民登基为帝，是为唐太宗，次年改元贞观。"
        ),
    },
    {
        "doc_name": "贞观政要_测试",
        "doc_page": 1,
        "text": (
            "唐太宗即位后，任用贤良，兼听纳谏，对内实行轻徭薄赋、疏缓刑罚的政策。"
            "他重用魏徵、房玄龄、杜如晦等贤臣，形成房谋杜断的高效决策机制。"
            "魏徵曾谏二百余事，唐太宗大多采纳，形成开明政治风气。"
            "他完善三省六部制，加强中央集权，提高行政效率，为后世制度奠定基础。"
        ),
    },
    {
        "doc_name": "贞观政要_测试",
        "doc_page": 2,
        "text": (
            "贞观年间，推行均田制、租庸调制，鼓励农业生产，恢复经济。"
            "轻徭薄赋，节俭治国，减轻百姓负担，使百姓能够休养生息。"
            "社会稳定，物价低廉，米斗不过三四钱，人口显著增长。"
            "国泰民安的局面开创了中国历史上著名的贞观之治，为后来的开元盛世奠定基础。"
        ),
    },
    {
        "doc_name": "唐书_测试",
        "doc_page": 1,
        "text": (
            "唐太宗实行开明的民族政策，被少数民族尊为天可汗。"
            "他派李靖等击败东突厥、吐谷浑，拓展疆域，巩固边疆稳定。"
            "公元630年灭东突厥，俘颉利可汗，重新控制西域，设立安西都护府。"
            "他任用少数民族将领如阿史那社尔，促进民族融合与文化交流。"
        ),
    },
    {
        "doc_name": "唐书_测试",
        "doc_page": 2,
        "text": (
            "唐太宗重视文化建设，设史馆修史，兴办学校，推动唐诗和书法发展。"
            "他尊崇儒学，修撰五经正义，统一经典解释，兼容佛道思想。"
            "玄奘西行取经归国后得到官方支持，长安成为国际都市。"
            "晚年渐生骄奢，修建翠微宫、玉华宫，两次亲征高句丽未能成功，消耗国力。"
        ),
    },
]


def main():
    print("开始插入李世民测试数据...")

    inserted = 0
    with get_connection() as conn:
        with conn.cursor() as cursor:
            for parent_idx, parent in enumerate(parents, 1):
                parent_text = parent["text"].strip()
                children = split_into_children(parent_text, chunk_size=150, overlap=30)

                for child_idx, child_text in enumerate(children):
                    # 计算 embedding
                    vec = get_embedding(child_text)
                    vec_str = "[" + ",".join(map(str, vec)) + "]"

                    cursor.execute(f"""
                        INSERT INTO {PG_DOC_TABLE}
                        (chunk_text, parent_text, chunk_vector, doc_name, doc_page, chunk_index)
                        VALUES (%s, %s, %s::vector, %s, %s, %s)
                    """, (
                        child_text,
                        parent_text,
                        vec_str,
                        parent["doc_name"],
                        parent["doc_page"],
                        child_idx,
                    ))
                    inserted += 1
                    print(f"  插入 [{inserted}] {parent['doc_name']} p{parent['doc_page']} child-{child_idx}")

            conn.commit()

    print(f"\n完成！共插入 {inserted} 条记录。")


if __name__ == "__main__":
    main()
