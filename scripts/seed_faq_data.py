#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插入李世民 FAQ 测试数据
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding.embedding_local_practice import get_embedding
from src.vectorstore.pg_pool_practice import get_connection
from config.pg_config import PG_TABLE_NAME

faqs = [
    {
        "question": "李世民是谁？",
        "answer": "李世民（598年—649年），即唐太宗，是唐朝第二位皇帝。他通过玄武门之变夺取皇位后，开创了中国历史上著名的贞观之治。",
        "category": "人物介绍",
        "source_doc": "唐太宗传_测试",
    },
    {
        "question": "玄武门之变是怎么回事？",
        "answer": "玄武门之变发生于公元626年7月2日，是李世民在长安城玄武门附近发动的政变。他杀死了皇太子李建成和齐王李元吉，随后唐高祖李渊退位，李世民登基为帝。",
        "category": "历史事件",
        "source_doc": "唐太宗传_测试",
    },
    {
        "question": "什么是贞观之治？",
        "answer": "贞观之治是唐太宗李世民在位期间（627年—649年）出现的政治清明、经济复苏、文化繁荣的治世局面。唐太宗任用贤良、虚心纳谏、轻徭薄赋，使社会安定、百姓安居乐业。",
        "category": "历史事件",
        "source_doc": "贞观政要_测试",
    },
    {
        "question": "李世民为什么被称为天可汗？",
        "answer": "唐太宗实行开明的民族政策，被西域诸国和少数民族尊为'天可汗'。他击败东突厥、平定吐谷浑，拓展疆域并设立安西都护府，促进了民族融合。",
        "category": "民族政策",
        "source_doc": "唐书_测试",
    },
    {
        "question": "魏徵和李世民是什么关系？",
        "answer": "魏徵是唐太宗时期著名的谏臣，以敢于直言进谏著称。李世民重用魏徵，曾说他可以当一面镜子，'以铜为镜，可以正衣冠；以人为镜，可以明得失'。",
        "category": "人物关系",
        "source_doc": "贞观政要_测试",
    },
]


def main():
    print("开始插入 FAQ 测试数据...")

    inserted = 0
    with get_connection() as conn:
        with conn.cursor() as cursor:
            for item in faqs:
                vec = get_embedding(item["question"])
                vec_str = "[" + ",".join(map(str, vec)) + "]"

                cursor.execute(f"""
                    INSERT INTO {PG_TABLE_NAME}
                    (question, question_vector, answer, category, source_doc)
                    VALUES (%s, %s::vector, %s, %s, %s)
                """, (
                    item["question"],
                    vec_str,
                    item["answer"],
                    item["category"],
                    item["source_doc"],
                ))
                inserted += 1
                print(f"  插入 [{inserted}] {item['question']}")

            conn.commit()

    print(f"\n完成！共插入 {inserted} 条 FAQ 记录。")


if __name__ == "__main__":
    main()
