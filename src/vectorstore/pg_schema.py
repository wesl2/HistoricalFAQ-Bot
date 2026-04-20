# -*- coding: utf-8 -*-
"""
PostgreSQL 数据库表结构定义
"""

import logging
from .pg_pool import get_connection
from config.pg_config import PG_TABLE_NAME, PG_DOC_TABLE, PG_CHAT_TABLE, VECTOR_DIM

logger = logging.getLogger(__name__)

# SQL 模板
ENABLE_VECTOR_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"


def create_tables(drop_existing: bool = False):
    """创建所有数据库表和索引"""
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        try:
            # 启用 pgvector 扩展
            logger.info("启用 pgvector 扩展...")
            cursor.execute(ENABLE_VECTOR_EXTENSION)
            
            # 删除旧表（如果指定）
            if drop_existing:
                logger.warning("删除已存在的表...")
                cursor.execute(f"DROP TABLE IF EXISTS {PG_TABLE_NAME} CASCADE")
                cursor.execute(f"DROP TABLE IF EXISTS {PG_DOC_TABLE} CASCADE")
                cursor.execute(f"DROP TABLE IF EXISTS {PG_CHAT_TABLE} CASCADE")
            
            # 创建 FAQ 表
            logger.info(f"创建 FAQ 表: {PG_TABLE_NAME}")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {PG_TABLE_NAME} (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    question_vector vector({VECTOR_DIM}),
                    answer TEXT NOT NULL,
                    search_vector tsvector 
                        GENERATED ALWAYS AS (to_tsvector('simple', question)) STORED,
                    category VARCHAR(50),
                    source_doc VARCHAR(200),
                    source_page INTEGER,
                    confidence FLOAT DEFAULT 0.9,
                    created_by VARCHAR(50) DEFAULT 'auto',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 创建索引
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_hnsw_faq 
                ON {PG_TABLE_NAME} 
                USING hnsw (question_vector vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_gin_faq
                ON {PG_TABLE_NAME}
                USING gin (search_vector);
            """)
            
            # 创建文档片段表
            logger.info(f"创建文档片段表: {PG_DOC_TABLE}")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {PG_DOC_TABLE} (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    chunk_vector vector({VECTOR_DIM}),
                    doc_name VARCHAR(200) NOT NULL,
                    doc_page INTEGER,
                    chunk_index INTEGER,
                    search_vector tsvector 
                        GENERATED ALWAYS AS (to_tsvector('simple', chunk_text)) STORED,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_hnsw_doc 
                ON {PG_DOC_TABLE} 
                USING hnsw (chunk_vector vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            
            # 创建对话历史表（LangChain 标准格式）
            logger.info(f"创建对话历史表: {PG_CHAT_TABLE}")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {PG_CHAT_TABLE} (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) NOT NULL,
                    role VARCHAR(20) NOT NULL,        -- 'human' 或 'ai'
                    content TEXT NOT NULL,            -- 消息内容
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 创建会话索引，提高查询性能
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_chat_history_session 
                ON {PG_CHAT_TABLE} (session_id);
            """)
            
            conn.commit()
            logger.info("所有表和索引创建成功")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"创建表失败: {e}")
            raise
        finally:
            cursor.close()


def init_database():
    """初始化数据库（创建所有表）"""
    create_tables(drop_existing=False)
