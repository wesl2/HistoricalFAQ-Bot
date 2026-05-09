#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt 管理器

职责：
1. 从外部文件加载 prompt 模板（热重载友好）
2. 支持变量渲染（{question}, {context} 等）
3. 区分 System Prompt 和 User Prompt

文件结构：
    prompts/
    ├── system_prompt.txt          # SystemMessage 内容（角色设定 + 任务要求）
    ├── user_context_template.txt  # HumanMessage 模板（参考资料包裹格式）
    └── ...（可扩展其他场景模板）

用法：
    from src.chat.prompt_manager import PromptManager
    pm = PromptManager()
    system_msg = pm.get_system_prompt()           # str
    user_msg = pm.render_user_prompt(docs, question)  # str
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# 默认模板目录（相对于项目根目录）
DEFAULT_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class PromptManager:
    """
    Prompt 模板管理器
    
    支持：
    - 从文件加载模板
    - 运行时热重载（不缓存，每次读取最新文件）
    - 变量替换渲染
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        初始化
        
        Args:
            prompts_dir: 模板文件目录，None 则使用默认 prompts/
        """
        self.prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR
        logger.info(f"[PromptManager] 模板目录: {self.prompts_dir}")

    def _load_template(self, filename: str) -> str:
        """
        从文件加载模板（每次读取，支持热重载）
        
        Args:
            filename: 模板文件名（如 "system_prompt.txt"）
        
        Returns:
            模板内容字符串
        """
        filepath = self.prompts_dir / filename
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            return content
        except FileNotFoundError:
            logger.error(f"[PromptManager] 模板文件不存在: {filepath}")
            raise
        except Exception as e:
            logger.error(f"[PromptManager] 读取模板失败: {filepath} | {e}")
            raise

    def get_system_prompt(self) -> str:
        """
        获取 System Prompt（角色设定 + 任务要求）
        
        对应文件：prompts/system_prompt.txt
        
        Returns:
            SystemMessage 内容
        """
        return self._load_template("system_prompt.txt")

    def render_user_prompt(
        self,
        docs_xml: str,
        question: str,
        history_text: str = "",
    ) -> str:
        """
        渲染 User Prompt（参考资料 + 用户问题）
        
        对应文件：prompts/user_context_template.txt
        
        Args:
            docs_xml: 参考资料的 XML 字符串（已格式化）
            question: 用户问题
            history_text: 对话历史文本（可选）
        
        Returns:
            HumanMessage 内容
        """
        template = self._load_template("user_context_template.txt")
        
        # 如果模板包含 {docs} 和 {question} 占位符，直接替换
        if "{docs}" in template and "{question}" in template:
            return template.format(docs=docs_xml, question=question)
        
        # 兼容旧格式：直接拼接
        parts = [docs_xml]
        if history_text:
            parts.append(f"<对话历史>\n{history_text}\n</对话历史>")
        parts.append(f"<用户问题>{question}</用户问题>")
        parts.append("请回答：")
        return "\n\n".join(parts)

    def list_templates(self) -> List[str]:
        """
        列出所有可用模板文件
        
        Returns:
            模板文件名列表
        """
        if not self.prompts_dir.exists():
            return []
        return [f.name for f in self.prompts_dir.iterdir() if f.suffix == ".txt"]


# 模块级单例（进程内共享）
_default_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """获取默认 PromptManager 实例（单例）"""
    global _default_manager
    if _default_manager is None:
        _default_manager = PromptManager()
    return _default_manager
