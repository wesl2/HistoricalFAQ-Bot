# -*- coding: utf-8 -*-
"""
工具模块

提供各种工具以支持 Agent 功能
"""

import logging
import math
from typing import List, Dict, Any

from src.chat.chat_engine import ChatEngine
from src.data_pipeline.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class Tools:
    """
    工具类
    """
    
    def __init__(self, chat_engine: ChatEngine):
        """
        初始化工具
        
        Args:
            chat_engine: 对话引擎实例
        """
        self.chat_engine = chat_engine
        self.document_processor = DocumentProcessor()
    
    def search_knowledge_base(self, query: str) -> str:
        """
        搜索知识库
        
        Args:
            query: 查询语句
            
        Returns:
            搜索结果
        """
        try:
            result = self.chat_engine.chat(query)
            answer = result.get("answer", "未找到相关信息")
            sources = result.get("sources", [])
            
            # 格式化结果
            formatted_result = f"搜索结果：\n{answer}\n\n参考来源：\n"
            for i, source in enumerate(sources[:3], 1):
                if source.get("type") == "faq":
                    formatted_result += f"{i}. FAQ: {source.get('question')} (置信度: {source.get('confidence', 0):.2f})\n"
                else:
                    formatted_result += f"{i}. 文档: {source.get('source', '未知')} (页码: {source.get('page', '未知')})\n"
            
            return formatted_result
        except Exception as e:
            logger.error(f"搜索知识库失败: {e}")
            return f"搜索失败: {str(e)}"
    
    def calculate(self, expression: str) -> str:
        """
        计算器工具
        
        Args:
            expression: 数学表达式
            
        Returns:
            计算结果
        """
        try:
            # 安全计算
            # 只允许基本的数学运算
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "不支持的表达式"
            
            # 计算结果
            result = eval(expression, {"__builtins__": None}, {"math": math})
            return f"计算结果: {result}"
        except Exception as e:
            logger.error(f"计算失败: {e}")
            return f"计算失败: {str(e)}"
    
    def process_document(self, file_path: str) -> str:
        """
        处理文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理结果
        """
        try:
            chunks = self.document_processor.process_file(file_path)
            if not chunks:
                return "文档处理失败"
            
            return f"文档处理成功，生成 {len(chunks)} 个文档块"
        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            return f"文档处理失败: {str(e)}"
    
    def get_weather(self, city: str) -> str:
        """
        获取天气信息
        
        Args:
            city: 城市名称
            
        Returns:
            天气信息
        """
        # 这里只是一个示例，实际应用中需要调用天气 API
        return f"{city} 的天气信息：晴朗，温度 25°C"
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有工具
        
        Returns:
            工具列表
        """
        return [
            {
                "name": "SearchKnowledgeBase",
                "func": self.search_knowledge_base,
                "description": "搜索知识库，用于回答关于历史人物的问题"
            },
            {
                "name": "Calculate",
                "func": self.calculate,
                "description": "计算器，用于进行数学计算"
            },
            {
                "name": "ProcessDocument",
                "func": self.process_document,
                "description": "处理文档，用于加载和分块文档"
            },
            {
                "name": "GetWeather",
                "func": self.get_weather,
                "description": "获取天气信息，用于查询城市天气"
            }
        ]


if __name__ == "__main__":
    # 测试工具
    from src.chat.chat_engine import ChatEngine
    
    chat_engine = ChatEngine(use_langchain=True)
    tools = Tools(chat_engine)
    
    # 测试搜索工具
    result = tools.search_knowledge_base("王洪文是谁")
    print("搜索结果:")
    print(result)
    
    # 测试计算器工具
    result = tools.calculate("1 + 2 * 3")
    print("\n计算结果:")
    print(result)
