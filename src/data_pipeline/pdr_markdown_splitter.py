# -*- coding: utf-8 -*-
"""
PDR (Parent Document Retrieval) Markdown 切分器

业界最佳实践整合：
1. LangChain MarkdownHeaderTextSplitter 的标题层级切分
2. Anthropic Contextual Retrieval (2024) 的上下文增强
3. TokenTextSplitter 的精确 token 级 child 切分

设计原则：
- Parent 必须保留完整语义（章节级，按 Markdown 标题切分）
- Child 用于向量检索（小粒度，512 token，有 overlap）
- Parent 用于 LLM 生成（大粒度，保留完整上下文）
- 每个 child 携带 parent 的完整内容和上下文头

参考：
- https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/
- https://www.anthropic.com/news/contextual-retrieval
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class PdrChunk:
    """PDR 切分后的统一 chunk 格式"""
    chunk_text: str              # child 文本（用于向量检索）
    parent_text: str             # parent 完整文本（用于 LLM 生成）
    heading: str                 # 所属章节标题
    heading_level: int           # 标题层级（1=H1, 2=H2）
    chunk_index: int             # 在 parent 内的序号
    doc_name: str = ""           # 文档名
    doc_page: int = 0            # 页码/章节号
    contextual_header: str = ""  # Anthropic 式上下文头
    metadata: Dict = field(default_factory=dict)


class PdrMarkdownSplitter:
    """
    PDR Markdown 切分器

    Args:
        parent_headers: 哪些标题层级作为 parent 边界。
                       默认 [("#", "H1"), ("##", "H2")]，
                       即 # 和 ## 都作为 parent 切分点。
        child_chunk_size: child 的 token 大小（默认 512）
        child_chunk_overlap: child 的重叠 token 数（默认 128）
        min_parent_length: parent 最小字符数，小于此值不单独成 parent（默认 200）
        add_contextual_header: 是否添加 Anthropic 式上下文头（默认 True）
    """

    def __init__(
        self,
        parent_headers: Optional[List[Tuple[str, str]]] = None,
        child_chunk_size: int = 512,
        child_chunk_overlap: int = 128,
        min_parent_length: int = 200,
        add_contextual_header: bool = True,
    ):
        self.parent_headers = parent_headers or [("#", "H1"), ("##", "H2")]
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.min_parent_length = min_parent_length
        self.add_contextual_header = add_contextual_header

        # Child 切分器：精确到 token，适合中文
        self._child_splitter = TokenTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
        )

        # 编译 parent 切分正则
        header_levels = [re.escape(h[0]) for h in self.parent_headers]
        self._header_pattern = re.compile(
            r'^(' + '|'.join(header_levels) + r')\s+(.+)$',
            re.MULTILINE
        )

    def split(
        self,
        markdown_text: str,
        doc_name: str = "",
        doc_page: int = 0,
    ) -> List[PdrChunk]:
        """
        执行 PDR 切分。

        Args:
            markdown_text: Markdown 格式文本
            doc_name: 文档名（用于 metadata）
            doc_page: 页码/章节号

        Returns:
            PdrChunk 列表
        """
        # 1. 按标题切 parent
        parents = self._split_parents(markdown_text)
        logger.info(
            "PDR Parent 切分完成 | doc=%s | parents=%d",
            doc_name, len(parents)
        )

        # 2. 收集标题层级关系（用于生成上下文头）
        heading_context = self._build_heading_context(parents)

        # 3. 每个 parent 切 child
        all_chunks = []
        for heading, level, parent_text in parents:
            chunks = self._split_child(
                heading=heading,
                level=level,
                parent_text=parent_text,
                heading_context=heading_context,
                doc_name=doc_name,
                doc_page=doc_page,
            )
            all_chunks.extend(chunks)

        logger.info(
            "PDR Child 切分完成 | doc=%s | total_chunks=%d",
            doc_name, len(all_chunks)
        )
        return all_chunks

    def _split_parents(self, text: str) -> List[Tuple[str, int, str]]:
        """
        按 Markdown 标题切分 parent。

        Returns:
            [(heading, level, content), ...]
            heading: 标题文本（不含 #）
            level: 1=H1, 2=H2
            content: 包含标题行的完整内容
        """
        lines = text.split('\n')
        parents = []
        current_level = 0
        current_heading = "开篇"
        current_content = []

        for line in lines:
            stripped = line.strip()
            match = self._header_pattern.match(stripped)

            if match:
                # 保存上一个 parent
                if current_content:
                    content = '\n'.join(current_content).strip()
                    # 修复：保留 H1 级别的 parent（即使内容短），
                    # 因为 H1 用于构建 heading_context，为 H2 提供上级章节信息
                    if len(content) >= self.min_parent_length or current_level == 1:
                        parents.append((current_heading, current_level, content))
                    else:
                        logger.debug(
                            "跳过短 parent | heading=%s | len=%d",
                            current_heading, len(content)
                        )

                # 开始新 parent
                hashes = match.group(1)
                title = match.group(2).strip()
                current_level = len(hashes)
                current_heading = title
                current_content = [line]
            else:
                current_content.append(line)

        # 最后一个 parent
        if current_content:
            content = '\n'.join(current_content).strip()
            if len(content) >= self.min_parent_length or current_level == 1:
                parents.append((current_heading, current_level, content))

        return parents

    @staticmethod
    def _build_heading_context(
        parents: List[Tuple[str, int, str]]
    ) -> Dict[str, str]:
        """
        构建标题上下文映射。

        用于生成 Anthropic 式上下文头：
        "这个 chunk 来自《唐太宗传》第一章 青少年生活 > 第一节 从隋末到唐初"
        """
        context = {}
        current_h1 = ""

        for heading, level, _ in parents:
            if level == 1:
                current_h1 = heading
                context[heading] = heading
            elif level == 2:
                context[heading] = f"{current_h1} > {heading}" if current_h1 else heading
            else:
                context[heading] = heading

        return context

    def _split_child(
        self,
        heading: str,
        level: int,
        parent_text: str,
        heading_context: Dict[str, str],
        doc_name: str,
        doc_page: int,
    ) -> List[PdrChunk]:
        """
        在单个 parent 内部切 child。
        """
        # 获取完整章节路径（如"第六章 统一边疆 > 第一节 抗击东突厥"）
        heading_path = heading_context.get(heading, heading)
        
        # 在 parent_text 开头加入完整路径，方便后续提取章节标题
        # 格式：# 第六章 统一边疆 > 第一节 抗击东突厥\n\n原始内容
        parent_text_with_path = f"# {heading_path}\n\n{parent_text}"

        # 如果 parent 本身就很短，不需要切分
        if len(parent_text) < self.child_chunk_size * 2:
            return [PdrChunk(
                chunk_text=parent_text_with_path,
                parent_text=parent_text_with_path,
                heading=heading,
                heading_level=level,
                chunk_index=0,
                doc_name=doc_name,
                doc_page=doc_page,
                contextual_header=self._build_contextual_header(
                    doc_name, heading_path, parent_text
                ),
            )]

        # 用 TokenTextSplitter 切分（基于原始 parent_text，不带路径前缀，避免影响语义）
        child_docs = self._child_splitter.create_documents([parent_text])

        chunks = []
        for i, child_doc in enumerate(child_docs):
            contextual_header = ""
            if self.add_contextual_header:
                contextual_header = self._build_contextual_header(
                    doc_name, heading_path, child_doc.page_content
                )

            chunks.append(PdrChunk(
                chunk_text=child_doc.page_content,
                parent_text=parent_text_with_path,
                heading=heading,
                heading_level=level,
                chunk_index=i,
                doc_name=doc_name,
                doc_page=doc_page,
                contextual_header=contextual_header,
            ))

        return chunks

    @staticmethod
    def _build_contextual_header(
        doc_name: str, heading_context: str, chunk_text: str
    ) -> str:
        """
        生成 Anthropic 式上下文头。

        原理：原始 chunk 可能缺少上下文（如"他于当天去世"，不知道"他"是谁）。
        给每个 chunk 前面加一句说明，让 LLM 理解这个 chunk 属于哪个文档、哪个章节。

        格式：
        "以下内容出自《唐太宗传》第一章 青少年生活 > 第一节 从隋末到唐初："
        """
        if not doc_name and not heading_context:
            return ""

        parts = []
        if doc_name:
            parts.append(f"《{doc_name}》")
        if heading_context:
            parts.append(heading_context)

        return f"以下内容出自{' '.join(parts)}："

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[PdrChunk]:
        """
        批量处理 Document 对象（和 LangChain 接口兼容）。
        """
        all_chunks = []
        for doc in documents:
            chunks = self.split(
                markdown_text=doc.page_content,
                doc_name=doc.metadata.get("doc_name", ""),
                doc_page=doc.metadata.get("doc_page", 0),
            )
            all_chunks.extend(chunks)
        return all_chunks


# =============================================================================
# 便捷函数
# =============================================================================

def split_markdown_pdr(
    markdown_text: str,
    doc_name: str = "",
    doc_page: int = 0,
    **kwargs
) -> List[PdrChunk]:
    """
    便捷函数：Markdown 文本 → PDR Chunk 列表。
    """
    splitter = PdrMarkdownSplitter(**kwargs)
    return splitter.split(markdown_text, doc_name=doc_name, doc_page=doc_page)
