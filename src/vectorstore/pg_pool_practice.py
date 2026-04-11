# -*- coding: utf-8 -*-
"""
PostgreSQL 连接池管理模块 - 练习版本

目标：自己手敲实现连接池单例管理
参考：pg_pool.py

核心概念：
- 连接池（Connection Pool）：预先创建多个数据库连接，避免频繁创建/销毁
- 单例模式（Singleton）：全局只有一个连接池实例
- 上下文管理器（Context Manager）：用 with 语句自动管理资源
- 延迟初始化（Lazy Initialization）：第一次使用时才创建，而非导入时
"""

# =============================================================================
# TODO 1: 导入依赖
# =============================================================================
# 需要导入：
# - logging: 日志记录
# - psycopg2: PostgreSQL 数据库驱动
# - psycopg2.pool: 连接池实现（ThreadedConnectionPool）
# - contextmanager: 用于创建上下文管理器的装饰器
# - PG_URL, POOL_MIN_CONN, POOL_MAX_CONN from config.pg_config: 配置

# 你的代码：
from pathlib import Path
import logging,sys,os
project_root = Path(__file__).parent.parent.parent  # 根据文件层级调整
sys.path.insert(0,str(project_root))
# ======================================
import psycopg2

from psycopg2 import pool
from contextlib import contextmanager
from config.pg_config import PG_URL, POOL_MIN_CONN, POOL_MAX_CONN
from urllib.parse import urlparse, urlunparse
# TODO 2: 创建 logger
# =============================================================================
# 使用 logging.getLogger(__name__) 创建模块级日志器

# 你的代码：
logger = logging.getLogger(__name__)

# =============================================================================
# TODO 3: 单例模式变量
# =============================================================================
# 定义一个模块级变量 _pool，初始值为 None
# 使用下划线前缀表示"私有变量，外部不应直接访问"
# 这是实现单例模式的关键：通过全局变量存储唯一实例

# 你的代码：
_pool = None

# =============================================================================
# TODO 4: 实现 get_pool() 函数（核心）
# =============================================================================
# 功能：延迟初始化连接池（第一次调用时才创建）
#
# 要点：
# 1. 使用 global _pool 声明修改全局变量
# 2. 如果 _pool is None，说明还没创建，执行初始化
# 3. 创建 psycopg2.pool.ThreadedConnectionPool：
#    - minconn: 最小连接数（从配置读取）
#    - maxconn: 最大连接数（从配置读取）
#    - dsn: 数据库连接字符串（PG_URL）
# 4. 使用 try-except 捕获 psycopg2.Error
# 5. 日志脱敏技巧：隐藏密码后再打印 URL
#    - PG_URL 格式: postgresql://user:password@host/db
#    - 用字符串操作把 password 部分替换成 ***
# 6. 如果 _pool 已存在，直接返回
#
# 异常处理：
# - 创建失败时记录 error 日志并 raise

def get_pool():
    """
    获取连接池实例（单例模式，延迟初始化）
    
    Returns:
        ThreadedConnectionPool: 连接池实例
    """
    global _pool
    if _pool is None:
        try:
            logger.info(f"正在创建连接池 (min={POOL_MIN_CONN}, max={POOL_MAX_CONN})...")
            _pool = psycopg2.pool.ThreadedConnectionPool(
                minconn = POOL_MIN_CONN,
                maxconn = POOL_MAX_CONN,
                dsn = PG_URL #DSN = Data Source Name（数据源名称）
            )
            parsed = urlparse(PG_URL)
            masked_url = PG_URL.replace(f":{parsed.password}@", ":***@") if parsed.password else PG_URL
            logger.info(f"连接池创建成功: {masked_url}")
       
        except psycopg2.Error as e:
            logger.error(f"连接池创建失败: {e}")    
            raise
            
    return _pool

# =============================================================================
# TODO 5: 实现 close_pool() 函数
# =============================================================================
# 功能：关闭连接池，释放所有连接
#
# 要点：
# 1. 使用 global _pool
# 2. 检查 _pool is not None 才关闭
# 3. 调用 _pool.closeall() 关闭所有连接
# 4. 使用 try-except-finally：
#    - try: 执行 closeall()
#    - except: 记录警告日志
#    - finally: 把 _pool = None（重置状态，下次 get_pool() 会重新创建）
# 5. 记录 info 日志表示关闭成功

def close_pool():
    """
    关闭连接池，释放所有连接
    
    应在程序退出时调用，确保资源正确释放。
    """
    # 你的代码：
    global _pool
    if _pool is not None:
        try:
            _pool.closeall()
            logger.info("连接池已关闭，所有连接已释放")
        except Exception as e:
            logger.warning(f"关闭连接池时发生错误: {e}")
        finally:
            _pool = None

# =============================================================================
# TODO 6: 实现 get_connection() 上下文管理器（核心）
# =============================================================================
# 功能：用 with 语句自动获取和归还连接
#
# 装饰器：
# - 使用 @contextmanager（从 contextlib 导入）
#
# 执行流程：
# 1. 从连接池获取连接：get_pool().getconn()
# 2. yield 连接给调用者使用（这就是 with 语句拿到的对象）
# 3. with 块结束后：
#    - 如果没有异常：自动 commit() 提交事务
#    - 如果有异常：rollback() 回滚事务，然后 raise
#    - finally：无论成败，putconn(conn) 归还连接
#
# 要点：
# - yield 前后的代码分别在 with 进入和退出时执行
# - 使用 try-except-finally 确保连接一定归还
# - 使用 conn 变量在 finally 中判断是否已获取连接
# - 用 id(conn) 在日志中标识不同连接

# 你的代码：
@contextmanager
def get_connection(cursor_factory=None):
    conn = None
    try:
        conn = get_pool().getconn()
        logger.debug(f"获取连接成功 (id={id(conn)})")
        #with前的内容
        yield conn #with的内容
        #with后的内容
        if conn:
            conn.commit()
    except Exception as e:
        if conn:
            conn .rollback()
            logger.debug(f"事务回滚 (id={id(conn)}) due to {e}")
        raise
    finally:
        if conn:
            get_pool().putconn(conn)
            logger.debug(f"连接已归还 (id={id(conn)})")
# =============================================================================
# TODO 7: 实现 get_cursor() 上下文管理器
# =============================================================================
# 功能：更上层封装，直接获取 cursor，连 connection 都不用管
#
# 实现方式：
# - 使用 with get_connection() as conn: 复用上面的函数
# - 在内部创建 cursor = conn.cursor()
# - yield cursor 给调用者
# - finally 中 cursor.close()
#
# 关键点：
# - 这是"组合优于继承"的思想，基于 get_connection() 构建
# - 调用者只需要处理 cursor，connection 自动管理

@contextmanager
def get_cursor():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            yield cursor
# =============================================================================
# TODO 8: 实现 check_connection() 健康检查
# =============================================================================
# 功能：检查数据库连接是否正常
#
# 要点：
# - 使用 get_cursor() 上下文管理器
# - 执行简单 SQL："SELECT 1"
# - 成功返回 True，异常返回 False
# - 使用 try-except 捕获所有异常，不要抛出去

def check_connection() -> bool:
    """
    检查数据库连接是否正常
    
    Returns:
        bool: 连接正常返回 True，否则 False
    """
    with get_cursor() as cursor:
        try:
            cursor.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"连接检查失败: {e}")
            return False
# =============================================================================
# TODO 9: 实现 get_pool_status() 获取连接池状态
# =============================================================================
# 功能：返回连接池的配置信息
#
# 返回格式（字典）：
# - 如果 _pool is None:
#   {"initialized": False, "minconn": POOL_MIN_CONN, "maxconn": POOL_MAX_CONN}
# - 如果 _pool 已创建:
#   {"initialized": True, "minconn": _pool.minconn, "maxconn": _pool.maxconn, "dsn": ...}
#
# 脱敏处理：
# - dsn 只返回 @ 后面的部分（host:port/db），隐藏用户名密码

def get_pool_status() -> dict:
    """
    获取连接池状态信息
    
    Returns:
        dict: 包含最小连接数、最大连接数、当前状态
    """
    global  _pool

    if _pool is None:
        return{
            "initialized": False,
            "minconn": POOL_MIN_CONN,
            "maxconn": POOL_MAX_CONN
        }
    else:
        return{
            "initialized": True,
            "minconn": _pool.minconn,
            "maxconn": _pool.maxconn,
            "dsn": urlparse(PG_URL).netloc + urlparse(PG_URL).path # 只返回 host:port/db
        }

# =============================================================================
# TODO 10: 测试代码
# =============================================================================
# 功能：验证实现是否正确
#
# 测试项：
# 1. 调用 check_connection() 检查是否能连上数据库
# 2. 使用 get_connection() 执行简单查询（如 SELECT version()）
# 3. 使用 get_cursor() 执行查询并打印结果
# 4. 调用 get_pool_status() 打印连接池状态
# 5. 测试单例：多次调用 get_pool()，确认返回的是同一个对象（id 相同）
# 6. 测试 close_pool() 后能否重新创建
#
# 注意：
# - 配置好日志输出，方便观察连接获取/归还的过程
# - 故意制造异常，验证事务回滚是否正确

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,  # 使用 DEBUG 级别观察详细过程
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # TODO 10.1: 测试连接
    print("=" * 50)
    print("测试1：连接检查")
    print("=" * 50)
    result = check_connection()
    print(f"连接状态: {result}")
    
    # TODO 10.2: 使用 get_connection() 执行查询
    print("\n" + "=" * 50)
    print("测试2：使用 get_connection()")
    print("=" * 50)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        print(cursor.fetchone())
        cursor.close()
    
    # TODO 10.3: 使用 get_cursor() 执行查询
    print("\n" + "=" * 50)
    print("测试3：使用 get_cursor()")
    print("=" * 50)
    with get_cursor() as cursor:
        cursor.execute("SELECT 1+1 as result")
        print(cursor.fetchone())
    
    # TODO 10.4: 测试单例模式
    print("\n" + "=" * 50)
    print("测试4：单例模式验证")
    print("=" * 50)
    pool1 = get_pool()
    pool2 = get_pool()
    print(f"同一对象? {pool1 is pool2}")
    print(f"id(pool1)={id(pool1)}, id(pool2)={id(pool2)}")
    
    # TODO 10.5: 打印连接池状态
    print("\n" + "=" * 50)
    print("测试5：连接池状态")
    print("=" * 50)
    status = get_pool_status()
    print(status)
    
    # TODO 10.6: 测试异常回滚
    print("\n" + "=" * 50)
    print("测试6：事务回滚")
    print("=" * 50)
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1/0")  # 故意制造除零异常
    except Exception as e:
        print(f"捕获到异常: {e}")

    # TODO 10.7: 关闭连接池
    print("\n" + "=" * 50)
    print("测试7：关闭连接池")
    print("=" * 50)
    close_pool()


# =============================================================================
# 练习检查清单
# =============================================================================
# □ 能正确导入所有依赖
# □ _pool 全局变量正确使用 global 声明
# □ get_pool() 实现延迟初始化（第一次调用才创建）
# □ get_pool() 正确创建 ThreadedConnectionPool
# □ get_pool() 实现日志脱敏（隐藏密码）
# □ close_pool() 正确释放资源并置空 _pool
# □ get_connection() 使用 @contextmanager 装饰器
# □ get_connection() 正确获取和归还连接
# □ get_connection() 正确处理事务提交/回滚
# □ get_cursor() 基于 get_connection() 组合实现
# □ check_connection() 能正确检测连接状态
# □ get_pool_status() 返回正确的状态字典
# □ 测试代码能验证单例模式（多次获取同一对象）
# □ 测试代码能验证异常时正确回滚
# =============================================================================
