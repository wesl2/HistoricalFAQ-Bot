# -*- coding: utf-8 -*-
"""
PostgreSQL 数据库表结构定义 - 练习版本

目标：自己手敲实现数据库表的创建和管理
参考：pg_schema.py

核心概念：
- pgvector 扩展：PostgreSQL 的向量支持插件
- HNSW 索引：高维向量的近似最近邻搜索索引（ANN）
- GIN 索引：倒排索引，用于全文搜索
- tsvector：PostgreSQL 的全文搜索向量类型
- SQL 模板：使用 Python f-string 构建动态 SQL

表结构：
1. faq_knowledge: FAQ 知识库表（含向量字段 + 全文搜索）
2. doc_chunks: 文档片段表（RAG 用）
3. chat_history: 对话历史表
"""

# =============================================================================
# TODO 1: 导入依赖
# =============================================================================
# 需要导入：
# - logging: 日志记录
# - get_connection from .pg_pool: 使用连接池获取连接
# - PG_TABLE_NAME, PG_DOC_TABLE, PG_CHAT_TABLE, VECTOR_DIM from config.pg_config: 表名配置
from pathlib import Path
import logging,sys,os
project_root = Path(__file__).parent.parent.parent  # 根据文件层级调整
sys.path.insert(0,str(project_root))
# 你的代码：
import logging
from pg_pool import get_connection
from config.pg_config import PG_TABLE_NAME,PG_DOC_TABLE, PG_CHAT_TABLE, VECTOR_DIM
from jinja2 import Template,Environment,FileSystemLoader

logger = logging.getLogger(__name__)
# =============================================================================



# =============================================================================
# TODO 3: 定义 SQL 常量模板
# =============================================================================
# 定义常量 ENABLE_VECTOR_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"
# 这是启用 pgvector 扩展的 SQL，只需要执行一次

# 你的代码：
ENABLE_VECTOR_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"


# =============================================================================
# TODO 4: 实现 create_tables() 函数（核心）
# =============================================================================
# 功能：创建所有数据库表和索引
#
# 参数：
# - drop_existing: bool = False，是否删除已存在的表（危险操作，默认 False）
#
# 执行流程：
# 1. 使用 get_connection() 获取连接
# 2. 创建 cursor
# 3. try-except-finally 结构：
#    - try: 执行所有建表操作
#    - except: 回滚事务并 raise
#    - finally: 关闭 cursor
#
# 详细步骤：
#
# 步骤 1：启用 pgvector 扩展
# - 执行 ENABLE_VECTOR_EXTENSION
# - 使用 logger.info 记录
#
# 步骤 2：可选删除旧表
# - 如果 drop_existing 为 True：
#   - 使用 logger.warning 记录（这是危险操作）
#   - 执行 DROP TABLE IF EXISTS ... CASCADE 删除三个表
#
# 步骤 3：创建 FAQ 知识库表（最复杂）
# 表名：PG_TABLE_NAME
# 字段：
#   - id: SERIAL PRIMARY KEY（自增主键）
#   - question: TEXT NOT NULL（标准问题）
#   - similar_question: TEXT NOT NULL（相似问题表述）
#   - similar_question_vector: vector(VECTOR_DIM)（向量，用于语义搜索）
#   - answer: TEXT NOT NULL（答案内容）
#   - search_vector: tsvector（全文搜索向量，GENERATED ALWAYS）
#     - 使用 to_tsvector('simple', similar_question) 自动生成
#     - STORED 表示存储计算结果（不是虚拟列）
#   - category: VARCHAR(50)（分类）
#   - source_doc: VARCHAR(200)（来源文档名）
#   - source_page: INTEGER（来源页码）
#   - confidence: FLOAT DEFAULT 0.9（置信度）
#   - created_by: VARCHAR(50) DEFAULT 'auto'（创建者）
#   - created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#   - updated_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#
# 步骤 4：创建 FAQ 表的索引
# 索引 1：HNSW 向量索引（用于语义相似度搜索）
#   - 名称：idx_hnsw_faq
#   - 类型：USING hnsw
#   - 字段：similar_question_vector vector_cosine_ops（余弦相似度）
#   - 参数：m = 16, ef_construction = 64
#
# 索引 2：GIN 全文搜索索引（用于关键词搜索）
#   - 名称：idx_gin_faq
#   - 类型：USING gin
#   - 字段：search_vector
#
# 步骤 5：创建文档片段表（RAG 用）
# 表名：PG_DOC_TABLE
# 字段：
#   - id: SERIAL PRIMARY KEY
#   - chunk_text: TEXT NOT NULL（片段文本）
#   - chunk_vector: vector(VECTOR_DIM)（向量）
#   - doc_name: VARCHAR(200) NOT NULL（文档名）
#   - doc_page: INTEGER（页码）
#   - chunk_index: INTEGER（片段序号）
#   - search_vector: tsvector（全文搜索）
#     - GENERATED ALWAYS AS (to_tsvector('simple', chunk_text)) STORED
#   - created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#
# 索引：HNSW 向量索引
#   - 名称：idx_hnsw_doc
#   - 字段：chunk_vector vector_cosine_ops
#   - 参数：m = 16, ef_construction = 64
#
# 步骤 6：创建对话历史表
# 表名：PG_CHAT_TABLE
# 字段：
#   - id: SERIAL PRIMARY KEY
#   - session_id: VARCHAR(100) NOT NULL（会话 ID）
#   - user_query: TEXT NOT NULL（用户问题）
#   - retrieved_faq_ids: INTEGER[]（检索到的 FAQ ID 数组）
#   - retrieved_doc_ids: INTEGER[]（检索到的文档 ID 数组）
#   - llm_response: TEXT（LLM 回答）
#   - llm_mode: VARCHAR(20)（模式：faq_only/hybrid/rag）
#   - created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#
# 步骤 7：提交事务
# - conn.commit()
# - logger.info("所有表和索引创建成功")
#
# 异常处理：
# - 捕获 Exception，conn.rollback() 回滚，logger.error 记录，raise

def create_tables(drop_existing: bool = False):
    """创建所有数据库表和索引"""
    with get_connection() as conn:
        cursor = conn.cursor()
        # 使用基于当前文件的绝对路径
        template_dir = Path(__file__).parent / 'sql_templates'
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        try:
            logger.info("启用 pgvector 扩展...")
            cursor.execute(ENABLE_VECTOR_EXTENSION)
            if drop_existing:
                logger.warning("删除已存在的表...")
                cursor.execute(f"DROP TABLE IF EXISTS {PG_TABLE_NAME} CASCADE")
                cursor.execute(f"DROP TABLE IF EXISTS {PG_DOC_TABLE} CASCADE")
                cursor.execute(f"DROP TABLE IF EXISTS {PG_CHAT_TABLE} CASCADE")

            logger.info(f"创建 FAQ 表: {PG_TABLE_NAME}")
            faq_template = env.get_template('create_FAQ.sql')
            faq_template = faq_template.render(
                table_name=PG_TABLE_NAME,
                vector_dim=VECTOR_DIM)
            cursor.execute(faq_template)

            logger.info(f"对表:{PG_TABLE_NAME} GIN和hnsw索引")
            index_template = env.get_template('create_index_onFAQ.sql')
            index_template = index_template.render(table_name=PG_TABLE_NAME)
            cursor.execute(index_template)

            logger.info(f"创建文档片段表: {PG_DOC_TABLE}")
            doc_template = env.get_template('create_pg_docTable.sql')
            doc_template = doc_template.render(
                table_name=PG_DOC_TABLE,
                vector_dim=VECTOR_DIM)
            cursor.execute(doc_template)
            
            logger.info(f"对表:{PG_DOC_TABLE} GIN和hnsw索引")
            doc_index_template = env.get_template('create_index_on_docTable.sql')
            doc_index_template = doc_index_template.render(table_name=PG_DOC_TABLE)
            cursor.execute(doc_index_template)

            # 创建对话历史表（LangChain兼容 非标准JSONB）
            logger.info(f"创建对话历史表: {PG_CHAT_TABLE}")
            chat_template = env.get_template('create_chatTable_and_index.sql')
            chat_template = chat_template.render(table_name=PG_CHAT_TABLE)
            # 创建会话索引，B树原生索引，适合对列排序和整批次查找
            cursor.execute(chat_template)

        except Exception as e:
            logger.error(f"创建表失败: {e}")
            raise
        finally:
            cursor.close()

# =============================================================================
# TODO 5: 实现 init_database() 便捷函数
# =============================================================================
# 功能：初始化数据库（安全模式，不删除已有表）
#
# 实现：
# - 调用 create_tables(drop_existing=False)

def init_database():
    """初始化数据库（创建所有表）"""
    # 你的代码：
    create_tables(drop_existing=True)


# =============================================================================
# TODO 6: 实现删除表函数（可选扩展）
# =============================================================================
# 功能：删除所有表（危险操作！）
#
# 实现思路：
# - 函数名：drop_all_tables()
# - 使用 get_connection()
# - 执行 DROP TABLE IF EXISTS ... CASCADE
# - 注意：CASCADE 会级联删除外键关联的数据
# - 记录 warning 日志

# 你的代码：
def drop_all_tables():
    """删除所有表（危险操作！）"""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            logger.warning("正在删除所有表... 这是一个危险操作！")
            cursor.execute(f"DROP TABLE IF EXISTS {PG_TABLE_NAME} CASCADE")
            cursor.execute(f"DROP TABLE IF EXISTS {PG_DOC_TABLE} CASCADE")
            cursor.execute(f"DROP TABLE IF EXISTS {PG_CHAT_TABLE} CASCADE")
            logger.info("所有表已删除")
        except Exception as e:
            logger.error(f"删除表失败: {e}")
            raise
        finally:
            cursor.close()


# =============================================================================
# TODO 7: 实现表结构检查函数（可选扩展）
# =============================================================================
# 功能：检查表是否存在
#
# 实现思路：
# - 函数名：check_tables_exist() -> bool
# - 查询 information_schema.tables
# - 检查 PG_TABLE_NAME 是否存在
# - 返回 True/False

# 你的代码：
def check_tables_exist() -> bool:
    """检查表是不是存在"""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (PG_TABLE_NAME,))
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"检查表存在性失败: {e}")
            raise
        finally:
            cursor.close()

# =============================================================================
# TODO 8: 测试代码
# =============================================================================
# 功能：验证实现是否正确
#
# 测试项：
# 1. 调用 init_database() 创建表（安全模式）
# 2. 再次调用 init_database()，测试 IF NOT EXISTS（应该不报错）
# 3. （可选）调用 drop_all_tables() 删除表
# 4. （可选）重新创建表，验证删除成功
#
# 注意：
# - 确保 PostgreSQL 服务已启动
# - 确保 pgvector 扩展已安装（CREATE EXTENSION）
# - 观察日志输出，确认表和索引创建成功

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # # TODO 8.1: 测试创建表
    # print("=" * 50)
    # print("测试1：创建数据库表")
    # print("=" * 50)
    # init_database()
    
    # # TODO 8.2: 测试重复创建（验证 IF NOT EXISTS）
    # print("\n" + "=" * 50)
    # print("测试2：重复创建（应不报错）")
    # print("=" * 50)
    # init_database()
    
    # TODO 8.3: （可选）测试删除表
    # print("\n" + "=" * 50)
    # print("测试3：删除表")
    # print("=" * 50)
    # drop_all_tables()

    check_tables_exist()
# =============================================================================
# 练习检查清单
# =============================================================================
# □ 能正确导入所有依赖（包括上一级目录的 pg_pool）
# □ 正确定义 ENABLE_VECTOR_EXTENSION 常量
# □ create_tables() 使用 get_connection() 上下文管理器
# □ create_tables() 有正确的 try-except-finally 结构
# □ 能创建 FAQ 表（含所有字段，特别是 vector 和 tsvector）
# □ 能创建 HNSW 向量索引（m 和 ef_construction 参数正确）
# □ 能创建 GIN 全文搜索索引
# □ 能创建文档片段表（doc_chunks）
# □ 能创建对话历史表（chat_history）
# □ 正确处理事务（commit/rollback）
# □ init_database() 是便捷封装
# □ 测试代码能成功创建表并输出日志
# =============================================================================
