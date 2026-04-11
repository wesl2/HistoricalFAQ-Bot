# -*- coding: utf-8 -*-
"""
FAQ 检索器 - 练习版本

目标：自己手敲实现 FAQ 向量检索功能
参考：faq_retriever.py

核心概念：
- NamedTuple: 具名元组，用于定义结构化数据类型
- 向量相似度检索: 使用 PostgreSQL 的 <=> 操作符（余弦距离）
- Top-K 检索: 返回最相似的 K 条记录

检索流程：
用户Query → Embedding向量化 → PostgreSQL向量检索 → 返回FAQ结果
"""

# =============================================================================
# TODO 1: 导入依赖
# =============================================================================
# 需要导入：
# - logging: 日志记录
# - List, NamedTuple from typing: 类型提示
# - get_embedding from src.embedding.embedding_local: 向量化函数
# - get_connection from src.vectorstore.pg_pool: 数据库连接
# - PG_TABLE_NAME from config.pg_config: FAQ表名配置
from pathlib import Path
import logging,sys,os
project_root = Path(__file__).parent.parent.parent  # 替换为你的项目根目录
sys.path.insert(0, str(project_root))  # 将项目根目录添加到 sys.path
import logging
from typing import List, NamedTuple
from src.embedding.embedding_local import get_embedding
from src.vectorstore.pg_pool import get_connection,get_cursor
from config.pg_config import PG_TABLE_NAME
from pydantic import BaseModel,Field,field_validator,ConfigDict


# =============================================================================
# TODO 2: 创建 logger
# =============================================================================
# 使用 logging.getLogger(__name__) 创建模块级日志器

# 你的代码：
logger = logging.getLogger(__name__)

# =============================================================================
# TODO 3: 定义 FAQResult 具名元组
# =============================================================================
# 功能：定义 FAQ 检索结果的数据结构
# 字段（按顺序）：
# - id: int - FAQ记录ID
# - question: str - 标准问题
# - similar_question: str - 相似问法（用于匹配）
# - answer: str - 答案内容
# - similarity: float - 相似度分数（0-1，越接近1越相似）
# - category: str - 分类标签
# - source_doc: str - 来源文档名
#
# 提示：使用 NamedTuple 创建，比 dataclass 更轻量，适合不可变数据

# 你的代码：
class FAQResult(BaseModel):
    """FAQ 检索结果（带运行时校验）"""
    model_config = ConfigDict(frozen=True)  #  冻结：创建后不可修改字段值
    id: int
    question: str
    similar_question: str
    answer: str
    similarity: float = Field(ge=0.0, le=1.0)  # 0-1范围
    category: str
    source_doc: str
    @field_validator("similar_question","answer","question",mode="before") #修饰下面的方法，表示在校验前处理输入值
    @classmethod
    def strip_whitespace(cls, v):
        if v is None:
            return ""
        return str(v).strip()  # 去除字符串首尾的空白字符
    
    @field_validator("category","source_doc",mode="before")
    @classmethod
    def default_category_source_doc(cls, v, info):
        if v is None:
            if info.field_name == "category":
                return "general"  # 默认分类
            elif info.field_name == "source_doc":
                return "unknown"  # 默认来源文档
        return v
    @field_validator("similarity", mode="before")
    @classmethod
    def default_similarity(cls, v):
        return 0.0 if v is None else float(v)
# =============================================================================
# TODO 4: 实现 FAQRetriever 类
# =============================================================================

class FAQRetriever:
    """
    FAQ 检索器
    
    核心功能：将用户问题向量化，在数据库中检索最相似的FAQ记录
    """
    
    def __init__(self, top_k: int = 5):

        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[FAQResult]:
        """
        检索 FAQ（核心方法）
        
        执行流程：
        1. 将用户 query 转换为向量（调用 get_embedding）
        2. 连接数据库，执行向量相似度检索
        3. 将结果封装为 FAQResult 列表返回
        
        Args:
            query: 用户输入的问题
        
        Returns:
            List[FAQResult]: 检索结果列表，按相似度降序排列
        
        TODO 4.2: 实现检索逻辑
        """
        # 步骤1：计算查询向量
        # 调用 get_embedding(query) 获取向量（1024维列表）
        # 你的代码：
        query_vector = get_embedding(query)
        query_str = '[' + ','.join(map(str,query_vector)) + ']'  # 转换为 PostgreSQL 格式字符串  
        # 步骤2：SQL 向量检索
        # 使用 with get_connection() as conn: 获取连接
        # 创建 cursor
        # 
        # SQL 查询要点：
        # - 使用 <=> 操作符计算余弦距离（pgvector语法）
        # - 相似度 = 1 - 余弦距离（转换为0-1范围，越大越相似）
        # - ORDER BY similarity DESC 按相似度降序
        # - LIMIT %s 限制返回数量（用 top_k）
        # 
        # 向量参数处理：
        # PostgreSQL 向量格式: [0.1,0.2,...]（方括号+逗号分隔）
        # 需要将 Python list 转换为字符串格式
        # 
        # 你的代码：
        with get_cursor() as cursor:
                
            cursor.execute(f"""
            SELECT id,question,similar_question,answer,                       
            1 - (similar_question_vector <=> %s::vector) AS similarity,
            category,source_doc
            FROM {PG_TABLE_NAME}
            ORDER BY similarity DESC
            LIMIT %s;
    """, (query_str, self.top_k))            
    # 步骤3：处理结果
    # 使用 cursor.fetchall() 获取所有行
    # 遍历每一行，创建 FAQResult 对象
    # 注意：row[4] 是 similarity，可能为 None，需要处理
    # category 和 source_doc 如果为 None，给默认值 "general"/"unknown"
    # 你的代码：
            results = []
            for row in cursor.fetchall():
                results.append(FAQResult(
                    id=row[0],
                    question=row[1],
                    similar_question=row[2],
                    answer=row[3],
                    similarity=row[4],
                    category=row[5],
                    source_doc=row[6]
                ))
        
        logger.info(f"FAQ 检索完成：找到 {len(results)} 条记录")
        return results

        

# =============================================================================
# TODO 5: 测试代码
# =============================================================================
# 功能：验证实现是否正确
#
# 测试步骤：
# 1. 创建 FAQRetriever 实例（top_k=3）
# 2. 调用 retrieve 检索 "王洪文是谁？"
# 3. 遍历结果，打印相似度、问题、答案（截断前50字符）
#
# 预期输出：
# - 找到 3 条记录（如果数据库有数据）
# - 相似度分数在 0-1 之间
# - 问题和答案不为空

if __name__ == "__main__":
    # 配置日志（可选，方便观察）
    logging.basicConfig(level=logging.INFO)
    
    
    # TODO 5.1: 创建检索器实例
    retriever = FAQRetriever(top_k=3)
    
    # TODO 5.2: 执行检索
    results = retriever.retrieve("王洪文是谁？")
    
    # TODO 5.3: 打印结果
    for r in results:
        print(f"相似度：{r.similarity:.3f}")
        print(f"问题：{r.question}")
        print(f"答案：{r.answer[:50]}...")
        print()
    



# =============================================================================
# 练习检查清单
# =============================================================================
# □ 能正确导入所有依赖（特别是 get_embedding 和 get_connection）
# □ 正确定义 FAQResult NamedTuple（7个字段）
# □ FAQRetriever.__init__ 正确保存 top_k
# □ retrieve 方法正确调用 get_embedding 向量化查询
# □ 正确使用 with get_connection() 获取数据库连接
# □ SQL 查询使用 <=> 操作符计算余弦距离
# □ 正确将向量列表转换为 PostgreSQL 字符串格式 [x,y,z]
# □ 正确使用 cursor.execute 执行参数化查询
# □ 正确处理查询结果，创建 FAQResult 列表
# □ 处理 similarity 为 None 的情况
# □ category 和 source_doc 为 None 时给默认值
# □ 使用 logger.info 记录检索结果数量
# □ 测试代码能正确运行并打印结果
# □ 理解 NamedTuple vs dataclass 的区别
# □ 理解 <=> 余弦距离操作符的含义（0=相同，2=相反）
# =============================================================================

# =============================================================================
# 关键知识点备忘
# =============================================================================
# 
# 1. NamedTuple 用法
#    from typing import NamedTuple
#    class Person(NamedTuple):
#        name: str
#        age: int
#    p = Person("张三", 20)
#    print(p.name)  # 张三
# 
# 2. PostgreSQL 向量操作符
#    - <=> : 余弦距离（0=完全相同，2=完全相反）
#    - <-> : 欧氏距离
#    - <#> : 内积
#    相似度 = 1 - 余弦距离（转换为0-1，越大越相似）
# 
# 3. 向量参数格式转换
#    query_vector = [0.1, 0.2, ...]  # Python list
#    vector_str = "[0.1,0.2,...]"     # PostgreSQL 格式
#    # 转换方法：f"[{','.join(map(str, query_vector))}]"
# 
# 4. Top-K 检索
#    ORDER BY similarity DESC LIMIT {top_k}
#    先排序（降序），再限制数量
# 
# 5. 空值处理
#    similarity = row[4] if row[4] is not None else 0.0
#    category = row[5] or "general"  # 简洁写法
# =============================================================================
