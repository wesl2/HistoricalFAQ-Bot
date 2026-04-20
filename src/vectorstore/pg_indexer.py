# -*- coding: utf-8 -*-
"""
FAQ 数据索引模块

负责将清洗后的 FAQ 数据导入 PostgreSQL
"""

import json
import logging
from typing import List, Dict, Any
from psycopg2.extras import execute_values
from .pg_pool import get_connection
from config.pg_config import PG_TABLE_NAME, BATCH_SIZE

logger = logging.getLogger(__name__)


class FAQIndexer:
    """FAQ 数据索引器"""
    
    def __init__(self, table_name: str = None):
        self.table_name = table_name or PG_TABLE_NAME
        self.batch_size = BATCH_SIZE
        
    def index_from_file(self, file_path: str, clear_existing: bool = False) -> int:
        """从 JSONL 文件导入 FAQ 数据"""
        if clear_existing:
            self._clear_table()
        
        total = 0
        batch = []
        logger.info(f"开始从文件导入: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    row = self._transform_record(record)
                    batch.append(row)
                    
                    if len(batch) >= self.batch_size:
                        inserted = self._insert_batch(batch)
                        total += inserted
                        logger.info(f"已导入 {total} 条记录...")
                        batch = []
                except Exception as e:
                    logger.warning(f"第 {line_num} 行处理失败: {e}")
        
        if batch:
            total += self._insert_batch(batch)
        
        logger.info(f"导入完成，共 {total} 条记录")
        return total
    
    def _transform_record(self, record: Dict[str, Any]) -> tuple:
        """将记录转换为数据库行元组"""
        # 处理 RAG 格式转换
        if "query" in record and "pos" in record:
            question = record["query"]
            answer = record["pos"][0] if record["pos"] else ""
            vector = record.get("vector", [0.0] * 1024)
            metadata = record.get("metadata", {})
        else:
            question = record["question"]
            answer = record["answer"]
            vector = record.get("vector", [0.0] * 1024)
            metadata = record.get("metadata", {})
        
        vector_str = "[" + ",".join([str(v) for v in vector]) + "]"
        
        category = metadata.get("category")
        source_doc = metadata.get("source_doc")
        source_page = metadata.get("source_page")
        confidence = metadata.get("confidence", 0.9)
        created_by = metadata.get("created_by", "auto")
        
        return (question, vector_str, answer,
                category, source_doc, source_page, confidence, created_by)
    
    def _insert_batch(self, rows: List[tuple]) -> int:
        """批量插入数据"""
        if not rows:
            return 0
        
        insert_sql = f"""
            INSERT INTO {self.table_name} 
            (question, question_vector, answer,
             category, source_doc, source_page, confidence, created_by)
            VALUES %s
        """

        
        with get_connection() as conn:
            cursor = conn.cursor()
            try:
                execute_values(cursor, insert_sql, rows)
                conn.commit()
                return len(rows)
            except Exception as e:
                conn.rollback()
                logger.error(f"批量插入失败: {e}")
                raise
            finally:
                cursor.close()
    
    def _clear_table(self):
        """清空表数据"""
        with get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"TRUNCATE TABLE {self.table_name} CASCADE")
                conn.commit()
                logger.warning(f"已清空表: {self.table_name}")
            finally:
                cursor.close()
