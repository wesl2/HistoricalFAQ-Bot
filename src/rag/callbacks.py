# -*- coding: utf-8 -*-
"""
Callback 可观测性系统

提供日志记录、性能监控、Token 统计等功能
"""

import json
import time
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class LoggingCallbackHandler(BaseCallbackHandler):
    """
    日志回调处理器
    
    记录 Chain 执行的完整流程
    """
    
    def __init__(self, log_file: str = None):
        super().__init__()
        self.log_file = log_file or f"logs/callback_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.logs = []
        self.start_times = {}
        
        # 确保日志目录存在
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def _log(self, event_type: str, data: Dict[str, Any]):
        """记录日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.logs.append(log_entry)
        
        # 实时写入文件
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Chain 开始"""
        chain_id = id(serialized)
        self.start_times[chain_id] = time.time()
        
        self._log("chain_start", {
            "chain_type": serialized.get("name", "unknown"),
            "inputs": {k: str(v)[:200] for k, v in inputs.items()}  # 截断长文本
        })
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Chain 结束"""
        self._log("chain_end", {
            "outputs": {k: str(v)[:200] for k, v in outputs.items()}
        })
    
    def on_chain_error(self, error: Exception, **kwargs):
        """Chain 错误"""
        self._log("chain_error", {
            "error_type": type(error).__name__,
            "error_message": str(error)
        })
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """LLM 开始"""
        llm_id = id(serialized)
        self.start_times[llm_id] = time.time()
        
        self._log("llm_start", {
            "llm_type": serialized.get("name", "unknown"),
            "prompt_count": len(prompts),
            "prompts": [p[:200] for p in prompts]  # 截断
        })
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        """LLM 结束"""
        # 统计 token 使用情况
        token_usage = {}
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
        
        generations = response.generations
        output_text = generations[0][0].text if generations else ""
        
        self._log("llm_end", {
            "token_usage": token_usage,
            "output_length": len(output_text),
            "output_preview": output_text[:200]
        })
    
    def on_llm_error(self, error: Exception, **kwargs):
        """LLM 错误"""
        self._log("llm_error", {
            "error_type": type(error).__name__,
            "error_message": str(error)
        })
    
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs):
        """检索器开始"""
        retriever_id = id(serialized)
        self.start_times[retriever_id] = time.time()
        
        self._log("retriever_start", {
            "retriever_type": serialized.get("name", "unknown"),
            "query": query[:200]
        })
    
    def on_retriever_end(self, documents: List[Document], **kwargs):
        """检索器结束"""
        self._log("retriever_end", {
            "document_count": len(documents),
            "documents": [
                {
                    "content": doc.page_content[:100],
                    "metadata": doc.metadata
                }
                for doc in documents[:5]  # 只记录前5个
            ]
        })
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """工具开始"""
        self._log("tool_start", {
            "tool_name": serialized.get("name", "unknown"),
            "input": input_str[:200]
        })
    
    def on_tool_end(self, output: str, **kwargs):
        """工具结束"""
        self._log("tool_end", {
            "output": output[:200]
        })


class PerformanceCallbackHandler(BaseCallbackHandler):
    """
    性能监控回调处理器
    
    统计耗时和吞吐量
    """
    
    def __init__(self):
        super().__init__()
        self.metrics = {
            "chain_count": 0,
            "llm_count": 0,
            "retriever_count": 0,
            "tool_count": 0,
            "total_latency": 0,
            "llm_latency": 0,
            "retriever_latency": 0
        }
        self.start_times = {}
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        self.start_times["chain"] = time.time()
        self.metrics["chain_count"] += 1
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        if "chain" in self.start_times:
            latency = time.time() - self.start_times["chain"]
            self.metrics["total_latency"] += latency
            del self.start_times["chain"]
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        self.start_times["llm"] = time.time()
        self.metrics["llm_count"] += 1
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        if "llm" in self.start_times:
            latency = time.time() - self.start_times["llm"]
            self.metrics["llm_latency"] += latency
            del self.start_times["llm"]
    
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs):
        self.start_times["retriever"] = time.time()
        self.metrics["retriever_count"] += 1
    
    def on_retriever_end(self, documents: List[Document], **kwargs):
        if "retriever" in self.start_times:
            latency = time.time() - self.start_times["retriever"]
            self.metrics["retriever_latency"] += latency
            del self.start_times["retriever"]
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        self.metrics["tool_count"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = self.metrics.copy()
        
        # 计算平均值
        if metrics["chain_count"] > 0:
            metrics["avg_total_latency"] = metrics["total_latency"] / metrics["chain_count"]
        if metrics["llm_count"] > 0:
            metrics["avg_llm_latency"] = metrics["llm_latency"] / metrics["llm_count"]
        if metrics["retriever_count"] > 0:
            metrics["avg_retriever_latency"] = metrics["retriever_latency"] / metrics["retriever_count"]
        
        return metrics
    
    def reset_metrics(self):
        """重置指标"""
        for key in self.metrics:
            self.metrics[key] = 0
        self.start_times.clear()


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """
    Token 使用统计回调处理器
    
    记录 LLM Token 消耗和成本估算
    """
    
    # Token 价格（每 1K tokens，仅供参考）
    PRICING = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "deepseek-chat": {"input": 0.001, "output": 0.002},
        "default": {"input": 0.001, "output": 0.002}
    }
    
    def __init__(self, model_name: str = "default"):
        super().__init__()
        self.model_name = model_name
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.session_count = 0
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        """记录 Token 使用"""
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.session_count += 1
    
    def get_usage(self) -> Dict[str, Any]:
        """获取使用统计"""
        pricing = self.PRICING.get(self.model_name, self.PRICING["default"])
        
        input_cost = (self.total_prompt_tokens / 1000) * pricing["input"]
        output_cost = (self.total_completion_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "model": self.model_name,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "session_count": self.session_count,
            "estimated_cost_usd": round(total_cost, 4),
            "avg_tokens_per_session": self.total_tokens // max(self.session_count, 1)
        }
    
    def reset(self):
        """重置统计"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.session_count = 0


class CallbackManager:
    """
    回调管理器
    
    统一管理多个回调处理器
    """
    
    def __init__(self):
        self.logging_callback = LoggingCallbackHandler()
        self.performance_callback = PerformanceCallbackHandler()
        self.token_callback = TokenUsageCallbackHandler()
        
        self.all_callbacks = [
            self.logging_callback,
            self.performance_callback,
            self.token_callback
        ]
    
    def get_callbacks(self) -> List[BaseCallbackHandler]:
        """获取所有回调处理器"""
        return self.all_callbacks
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        return {
            "performance": self.performance_callback.get_metrics(),
            "token_usage": self.token_callback.get_usage()
        }
    
    def reset(self):
        """重置所有统计"""
        self.performance_callback.reset_metrics()
        self.token_callback.reset()


# 全局回调管理器实例
_callback_manager = None


def get_callback_manager() -> CallbackManager:
    """获取回调管理器（单例）"""
    global _callback_manager
    if _callback_manager is None:
        _callback_manager = CallbackManager()
    return _callback_manager
