# -*- coding: utf-8 -*-
"""
标准 LangChain Memory 管理（公司级实践）

外部持久化 + 清晰的历史管理
不混在 Chain 里，单独管理
"""

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import PostgresChatMessageHistory, ChatMessageHistory
from typing import List, Optional, Dict

from config.pg_config import PG_URL, PG_CHAT_TABLE

logger = None


def get_logger():
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    return logger


class StandardMemory:
    """
    标准 Memory 管理

    公司实践：
    - 外部持久化（PostgreSQL 或内存）
    - 清晰的历史管理
    - 不混在 Chain 里
    """

    def __init__(self, session_id: str, use_postgres: bool = True):
        """
        初始化 Memory

        Args:
            session_id: 会话 ID
            use_postgres: 是否使用 PostgreSQL 持久化
        """
        self.session_id = session_id
        self.use_postgres = use_postgres
        self.history = self._load_history()

    def _load_history(self) -> BaseChatMessageHistory:
        """加载对话历史"""
        if self.use_postgres:
            try:
                get_logger().info(f"使用 PostgreSQL 持久化对话历史: session_id={self.session_id}")
                return PostgresChatMessageHistory(
                    connection_string=PG_URL,
                    session_id=self.session_id,
                    table_name=PG_CHAT_TABLE
                )
            except Exception as e:
                get_logger().warning(f"PostgreSQL 连接失败，降级为内存存储: {e}")
                self.use_postgres = False

        get_logger().warning("使用内存存储对话历史（非持久化）")
        return ChatMessageHistory()

    def add_user_message(self, message: str):
        """添加用户消息"""
        self.history.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        """添加 AI 消息"""
        self.history.add_message(AIMessage(content=message))

    def add_message(self, message: BaseMessage):
        """添加任意消息"""
        self.history.add_message(message)

    def get_messages(self) -> List[BaseMessage]:
        """获取所有消息"""
        return self.history.messages

    def get_last_n_messages(self, n: int = 10) -> List[BaseMessage]:
        """获取最近 N 条消息（修复：不再假设每轮严格两条）"""
        return self.history.messages[-n:]

    def get_history_string(self, n: int = 10) -> str:
        """获取历史字符串（用于 Prompt 拼接）"""
        messages = self.get_last_n_messages(n)

        return "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in messages
        )

    def clear(self):
        """清空历史"""
        self.history.clear()
        get_logger().info(f"清空会话历史: session_id={self.session_id}")

    def __len__(self) -> int:
        """获取消息数量"""
        return len(self.history.messages)


# 全局缓存
_memory_cache: Dict[str, StandardMemory] = {}


def get_standard_memory(session_id: str, use_postgres: bool = True) -> StandardMemory:
    """获取标准 Memory 实例（单例）"""
    if session_id not in _memory_cache:
        _memory_cache[session_id] = StandardMemory(session_id, use_postgres)
    return _memory_cache[session_id]


def clear_memory(session_id: str):
    """清空指定会话的 Memory"""
    if session_id in _memory_cache:
        _memory_cache[session_id].clear()
        del _memory_cache[session_id]
