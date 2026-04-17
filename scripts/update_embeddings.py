#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给 doc_chunks 表中缺失 chunk_vector 的记录补全 embedding
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from src.vectorstore.pg_pool_practice import get_connection
from src.embedding.embedding_local_practice import get_embedding
from config.pg_config_practice import PG_DOC_TABLE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def update_missing_embeddings(batch_size: int = 5):
    """
    扫描 doc_chunks 表中 chunk_vector 为 NULL 的记录，
    逐批生成 embedding 并更新回数据库。
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # 1. 查询所有缺失向量的记录
            cursor.execute(f"""
                SELECT id, chunk_text 
                FROM {PG_DOC_TABLE}
                WHERE chunk_vector IS NULL
                ORDER BY id
            """)
            rows = cursor.fetchall()

            if not rows:
                logger.info("没有需要更新的记录，所有 chunk_vector 已存在")
                return 0

            logger.info(f"发现 {len(rows)} 条记录需要生成 embedding")

            total_updated = 0
            for i in range(0, len(rows), batch_size):
                batch = rows[i: i + batch_size]
                updates = []

                for doc_id, chunk_text in batch:
                    logger.info(f"正在生成 embedding: ID={doc_id}, 文本={chunk_text[:30]}...")
                    vector = get_embedding(chunk_text)
                    vector_str = "[" + ",".join(map(str, vector)) + "]"
                    updates.append((vector_str, doc_id))

                # 2. 批量更新
                cursor.executemany(f"""
                    UPDATE {PG_DOC_TABLE}
                    SET chunk_vector = %s::vector
                    WHERE id = %s
                """, updates)
                conn.commit()

                total_updated += len(updates)
                logger.info(f"已更新 {total_updated}/{len(rows)} 条记录")

            logger.info(f"全部完成，共更新 {total_updated} 条记录")
            return total_updated


if __name__ == "__main__":
    update_missing_embeddings(batch_size=5)