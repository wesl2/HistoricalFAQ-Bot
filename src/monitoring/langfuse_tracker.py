# -*- coding: utf-8 -*-
"""
Langfuse 可观测性追踪模块（预留接口）

目标：集成 Langfuse 实现 LLM 调用全链路追踪与可视化，
      覆盖检索、生成、延迟、Token 消耗等关键指标。

待实现：
- 自动追踪 ChatEngine.achat() / astream() 调用
- 记录检索召回的文档片段、相似度分数
- 记录 LLM 输入输出、延迟、Token 用量
- 评分反馈（用户点赞/点踩）回传
- Dashboard 配置与告警规则

依赖：langfuse>=2.0.0
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LangfuseTracker:
    """Langfuse 追踪器占位类"""

    def __init__(self, public_key: str, secret_key: str, host: str = "https://cloud.langfuse.com"):
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        # TODO: 初始化 Langfuse 客户端
        logger.warning("Langfuse 追踪器尚未初始化，仅为预留接口。")

    def trace_query(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: Optional[list] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """记录一次问答追踪（占位）"""
        raise NotImplementedError

    def score_response(
        self,
        trace_id: str,
        score: float,
        comment: Optional[str] = None,
    ) -> None:
        """用户反馈评分（占位）"""
        raise NotImplementedError
