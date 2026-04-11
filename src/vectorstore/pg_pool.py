# -*- coding: utf-8 -*-
"""
PostgreSQL 连接池管理模块

本模块实现单例模式的连接池管理，确保整个应用生命周期内
只有一个连接池实例，避免资源浪费。

使用方式:
    # 获取连接
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM faq_knowledge")
        
    # 程序结束时关闭连接池
    close_pool()
"""

import logging
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from config.pg_config import PG_URL, POOL_MIN_CONN, POOL_MAX_CONN

# 模块级日志器
logger = logging.getLogger(__name__)

# =============================================================================
# 单例连接池（延迟初始化）
# =============================================================================

# 私有变量，存储连接池实例
# 使用下划线前缀表示"内部使用，请勿直接访问"
_pool = None


def get_pool():
    """
    获取连接池实例（单例模式，延迟初始化）
    
    第一次调用时创建连接池，后续调用返回同一个实例。
    这种设计避免在导入模块时就创建连接，提高启动速度。
    
    Returns:
        ThreadedConnectionPool: 连接池实例
        
    Raises:
        psycopg2.Error: 连接数据库失败时抛出
    """
    global _pool
    
    # 延迟初始化：第一次调用时才创建
    if _pool is None:
        try:
            logger.info(f"正在创建连接池 (min={POOL_MIN_CONN}, max={POOL_MAX_CONN})...")
            
            _pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=POOL_MIN_CONN,
                maxconn=POOL_MAX_CONN,
                dsn=PG_URL
            )
            
            # 脱敏后的日志（隐藏密码）
            masked_url = PG_URL.replace(PG_URL.split('@')[0].split(':')[-1], '***')
            logger.info(f"连接池创建成功: {masked_url}")
            
        except psycopg2.Error as e:
            logger.error(f"连接池创建失败: {e}")
            raise
    
    return _pool


def close_pool():
    """
    关闭连接池，释放所有连接
    
    应在程序退出时调用，确保资源正确释放。
    调用后如果再次 get_pool()，会重新创建连接池。
    
    Example:
        try:
            # 主程序运行
            main()
        finally:
            # 确保退出时关闭连接池
            close_pool()
    """
    global _pool
    
    if _pool is not None:
        try:
            _pool.closeall()
            logger.info("连接池已关闭，所有连接已释放")
        except Exception as e:
            logger.warning(f"关闭连接池时出错: {e}")
        finally:
            _pool = None


@contextmanager
def get_connection():
    """
    上下文管理器：自动获取和归还连接
    
    使用 with 语句可以确保连接一定会归还到池中，
    即使发生异常也会正确释放。
    
    Yields:
        connection: 数据库连接对象
        
    Example:
        # 基础用法
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
            rows = cursor.fetchall()
            cursor.close()
        # 这里连接自动归还，无需手动 putconn
        
        # 事务处理
        with get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("INSERT ...")
                conn.commit()  # 成功则提交
            except:
                conn.rollback()  # 失败则回滚
                raise
            finally:
                cursor.close()
    """
    conn = None
    try:
        # 从池中获取连接
        conn = get_pool().getconn()
        logger.debug(f"获取连接成功 (id={id(conn)})")
        
        # 生成连接给调用者使用
        yield conn
        
        # 如果没有异常，自动提交事务
        # 注意：如果调用者已经手动 commit/rollback，这里不会报错
        if conn:
            conn.commit()
            logger.debug(f"事务已提交 (id={id(conn)})")
            
    except Exception as e:
        # 发生异常时回滚事务
        if conn:
            conn.rollback()
            logger.warning(f"事务已回滚 (id={id(conn)}): {e}")
        raise
        
    finally:
        # 无论成功还是失败，都要归还连接
        if conn:
            get_pool().putconn(conn)
            logger.debug(f"连接已归还 (id={id(conn)})")


@contextmanager
def get_cursor():
    """
    更便捷的上下文管理器：直接获取 cursor
    
    如果连 cursor 也不想手动管理，用这个。
    
    Yields:
        cursor: 数据库游标对象
        
    Example:
        with get_cursor() as cursor:
            cursor.execute("SELECT * FROM table")
            rows = cursor.fetchall()
        # cursor 和 connection 都自动关闭
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()


# =============================================================================
# 连接健康检查
# =============================================================================

def check_connection() -> bool:
    """
    检查数据库连接是否正常
    
    Returns:
        bool: 连接正常返回 True，否则 False
    """
    try:
        with get_cursor() as cursor:
            cursor.execute("SELECT 1")
            return True
    except Exception as e:
        logger.error(f"连接检查失败: {e}")
        return False


def get_pool_status() -> dict:
    """
    获取连接池状态信息
    
    Returns:
        dict: 包含最小连接数、最大连接数、当前状态
    """
    global _pool
    
    if _pool is None:
        return {
            "initialized": False,
            "minconn": POOL_MIN_CONN,
            "maxconn": POOL_MAX_CONN
        }
    
    # psycopg2 的连接池没有直接暴露当前连接数
    # 这里返回配置信息
    return {
        "initialized": True,
        "minconn": _pool.minconn,
        "maxconn": _pool.maxconn,
        "dsn": PG_URL.split('@')[1] if '@' in PG_URL else "unknown"  # 脱敏
    }