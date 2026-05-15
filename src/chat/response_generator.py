# -*- coding: utf-8 -*-
"""
回答生成器（适配 StandardLLM）

基于检索结果和 LLM 生成最终回答。
支持注入对话历史（history messages）。

v2.0 改动：
1. 线程安全：build_prompt / extract_citations 改为纯函数，source_map 显式传递
2. prompt 改用 XML 标签包裹参考资料，结构化更强
"""

import html
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Generator, AsyncGenerator, Dict, Tuple, TypedDict, Literal, Union

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from src.llm.standard_llm_new import StandardLLM
from src.retrieval.faq_retriever_practice import FAQResult
from src.retrieval.doc_retriever_practice import DocResult
from src.chat.prompt_manager import get_prompt_manager
from src.chat.citation_verifier import verify_citations, format_issues

logger = logging.getLogger(__name__)


class FAQSource(TypedDict):
    type: Literal["faq"]
    question: str
    answer: str


class DocSource(TypedDict):
    type: Literal["doc"]
    doc_name: str
    page: int
    content: str


SourceMap = Dict[str, Union[FAQSource, DocSource]]


@dataclass
class GenerationResult:
    """生成结果（含引用溯源）"""
    answer: str
    citations: List[Dict] = field(default_factory=list)


class ResponseGenerator:
    """回答生成器（无状态，线程安全）"""

    def __init__(self, llm_mode: str = None):
        """
        初始化

        Args:
            llm_mode: LLM 模式（"local"/"api"），None 则先读 LLM_MODE 环境变量，
                     否则用配置默认值
        """
        import os
        self.llm_mode = llm_mode or os.getenv("LLM_MODE")
        self._prompt_mgr = get_prompt_manager()
        self._system_prompt_pure = (
            "你是一位专业的中国历史研究专家。"
            "请基于你的知识回答用户问题，保持客观中立。"
            "如果不确定，请明确说明。"
        )
        # Prompt 组装配置
        self.max_faq_in_prompt = 3
        self.max_doc_in_prompt = 5

    def _build_messages(
        self,
        prompt: str,
        history_messages: Optional[List[BaseMessage]] = None,
        pure: bool = False,
    ) -> List[BaseMessage]:
        """构建消息列表（同步/异步共用）"""
        if pure:
            system = self._system_prompt_pure
        else:
            system = self._prompt_mgr.get_system_prompt()
        messages: List[BaseMessage] = [SystemMessage(content=system)]
        if history_messages:
            messages.extend(history_messages)
        messages.append(HumanMessage(content=prompt))
        return messages

    def _sanitize_citations(self, answer: str, source_map: Dict) -> Tuple[str, List[Dict], List]:
        """
        清洗虚假引用（Citation Sanitization）

        关键修复：不直接删除 CRITICAL 引用标号，而是降级为"（史料记载）"，
        避免"裸奔事实"（有事实声明但无引用支撑）。

        流程：
        1. 提取引用
        2. 校验引用一致性
        3. 对 CRITICAL 引用：先尝试 re-pair（在其他文档中找支撑），
           失败则降级为"（史料记载）"
        4. 重新提取清洗后的引用

        Returns:
            (清洗后的答案, 清洗后的引用列表, 问题列表)
        """
        issues = verify_citations(answer, source_map)
        if not issues:
            citations = self.extract_citations(answer, source_map)
            return answer, citations, []

        # 收集 CRITICAL 级别的虚假引用
        critical_ids = {issue.citation_id for issue in issues if issue.severity == "CRITICAL"}
        warning_ids = {issue.citation_id for issue in issues if issue.severity == "WARNING"}

        if critical_ids:
            logger.warning(
                "[ResponseGenerator] 发现 %d 个 CRITICAL 虚假引用: %s",
                len(critical_ids), critical_ids
            )

        # 对 CRITICAL 引用进行降级处理
        for cid in critical_ids:
            # 策略：先尝试 re-pair（在其他 source 中找支撑）
            repaired = self._try_repair_citation(answer, cid, source_map)
            if repaired:
                logger.info("[ResponseGenerator] 引用 [%s] 已 re-pair 到 [%s]", cid, repaired)
                answer = answer.replace(f"[{cid}]", f"[{repaired}]")
            else:
                # 降级为"（史料记载）"，避免裸奔事实
                logger.info("[ResponseGenerator] 引用 [%s] 降级为'（史料记载）'", cid)
                answer = answer.replace(f"[{cid}]", "（史料记载）")

        # 对 WARNING 引用：保留但加上"（存疑）"标记
        for cid in warning_ids:
            if cid not in critical_ids:  # 避免重复处理
                answer = answer.replace(f"[{cid}]", f"[{cid}·存疑]")

        # 清理格式
        answer = re.sub(r'\s+', ' ', answer).strip()

        # 重新提取引用（只保留通过的）
        citations = self.extract_citations(answer, source_map)
        return answer, citations, issues

    def _try_repair_citation(self, answer: str, cid: str, source_map: Dict) -> Optional[str]:
        """
        尝试在 source_map 的其他文档中为该引用找到支撑。

        策略：提取引用句子的核心关键词，检查其他文档是否包含这些词。
        如果找到匹配的文档，返回其编号；否则返回 None。
        """
        from src.chat.citation_verifier import _extract_key_terms
        import re

        # 提取包含 [cid] 的句子
        pattern = re.compile(r'[^。！？\n]*?\[' + re.escape(cid) + r'\][^。！？\n]*')
        matches = pattern.findall(answer)
        if not matches:
            return None

        sentence = matches[0]
        # 去掉引用标号本身，只保留事实内容
        sentence_clean = re.sub(r'\[\d+\]', '', sentence).strip()
        key_terms = _extract_key_terms(sentence_clean)

        if not key_terms:
            return None

        # 在其他 source 中寻找最佳匹配
        best_match = None
        best_score = 0.0
        min_threshold = 0.5  # 至少 50% 关键词匹配才算 re-pair 成功

        for other_id, other_source in source_map.items():
            if other_id == cid:
                continue
            other_text = other_source.get("content") or other_source.get("answer", "")
            if not other_text:
                continue

            matched = sum(1 for term in key_terms if term in other_text)
            score = matched / len(key_terms) if key_terms else 0

            if score > best_score and score >= min_threshold:
                best_score = score
                best_match = other_id

        return best_match

    def generate(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult],
        history_messages: Optional[List[BaseMessage]] = None,
    ) -> GenerationResult:
        """生成回答（同步）"""
        prompt, source_map = self.build_prompt(query, faq_results, doc_results)
        messages = self._build_messages(prompt, history_messages)
        try:
            resp = StandardLLM.invoke(messages, mode=self.llm_mode)
            answer = resp.content
            # 【修复】先清理 LLM 自发生成的 footer，再从正文中提取引用
            # 避免 footer 中的虚假引用混入 citations
            answer = self._strip_llm_citation_footer(answer)
            answer, citations, _ = self._sanitize_citations(answer, source_map)
            answer, footer, id_remap = self._format_citation_footer(answer, source_map)
            if footer:
                answer += footer
            # 同步更新 citations 中的 id（重映射后的连续编号），
            # 原始数据库 id 保留到 original_id 供后端调试
            if id_remap:
                for c in citations:
                    old_id = c.get("id")
                    if old_id in id_remap:
                        c["original_id"] = old_id
                        c["id"] = id_remap[old_id]
            return GenerationResult(answer=answer, citations=citations)
        except Exception as e:
            logger.error("[ResponseGenerator] LLM 调用失败: %s", e)
            return GenerationResult(answer="抱歉，生成回答时出错了，请稍后再试。")

    async def agenerate(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult],
        history_messages: Optional[List[BaseMessage]] = None,
    ) -> GenerationResult:
        """生成回答（异步）"""
        prompt, source_map = self.build_prompt(query, faq_results, doc_results)
        messages = self._build_messages(prompt, history_messages)
        try:
            resp = await StandardLLM.ainvoke(messages, mode=self.llm_mode)
            answer = resp.content
            # 【修复】先清理 LLM 自发生成的 footer，再从正文中提取引用
            # 避免 footer 中的虚假引用混入 citations
            answer = self._strip_llm_citation_footer(answer)
            answer, citations, _ = self._sanitize_citations(answer, source_map)
            answer, footer, id_remap = self._format_citation_footer(answer, source_map)
            if footer:
                answer += footer
            # 同步更新 citations 中的 id（重映射后的连续编号），
            # 原始数据库 id 保留到 original_id 供后端调试
            if id_remap:
                for c in citations:
                    old_id = c.get("id")
                    if old_id in id_remap:
                        c["original_id"] = old_id
                        c["id"] = id_remap[old_id]
            return GenerationResult(answer=answer, citations=citations)
        except Exception as e:
            logger.error("[ResponseGenerator] LLM 异步调用失败: %s", e)
            return GenerationResult(answer="抱歉，生成回答时出错了，请稍后再试。")

    def generate_stream(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult],
        history_messages: Optional[List[BaseMessage]] = None,
    ) -> Generator[str, None, None]:
        """
        流式生成回答（同步）

        Yields:
            生成的文本片段
        """
        prompt, _ = self.build_prompt(query, faq_results, doc_results)
        messages = self._build_messages(prompt, history_messages)
        for chunk in StandardLLM.stream(messages, mode=self.llm_mode):
            yield chunk.content

    async def agenerate_stream(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult],
        history_messages: Optional[List[BaseMessage]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式生成回答（异步）

        Yields:
            生成的文本片段
        """
        prompt, _ = self.build_prompt(query, faq_results, doc_results)
        messages = self._build_messages(prompt, history_messages)
        async for chunk in StandardLLM.astream(messages, mode=self.llm_mode):
            yield chunk.content

    def generate_pure_llm(self, query: str, history_messages: Optional[List[BaseMessage]] = None) -> str:
        """
        纯 LLM 生成（不走检索，用于兜底/错误恢复）

        Returns:
            str: 生成的回答
        """
        messages = self._build_messages(query, history_messages, pure=True)
        try:
            resp = StandardLLM.invoke(messages, mode=self.llm_mode)
            return resp.content
        except Exception as e:
            logger.error("[ResponseGenerator] 纯 LLM 兜底调用失败: %s", e)
            return "抱歉，服务暂时不可用，请稍后再试。"

    def build_prompt(
        self,
        query: str,
        faq_results: List[FAQResult],
        doc_results: List[DocResult]
    ) -> Tuple[str, SourceMap]:
        """
        构建提示词（XML 结构化包裹）

        Returns:
            (prompt_str, source_map)
            source_map: {"1": {"type": "doc", ...}, "2": {"type": "doc", ...}}
            
        注意：FAQ 不赋予数字编号，仅供 LLM 参考背景信息，
        禁止 LLM 在回答中引用 FAQ。只有文档部分赋予 [1]/[2]/... 编号。
        """
        source_map: Dict[str, dict] = {}
        docs_parts = []

        if faq_results:
            docs_parts.append("  <!-- FAQ 问答对（仅供参考，回答时请勿引用） -->")
            for i, r in enumerate(faq_results[:self.max_faq_in_prompt], 1):
                docs_parts.append(f'  <faq id="F{i}">')
                docs_parts.append(f'    <question>{self._xml_escape(r.question)}</question>')
                docs_parts.append(f'    <answer>{self._xml_escape(r.answer or "")}</answer>')
                docs_parts.append(f'  </faq>')
                # FAQ 不加入 source_map，因此 LLM 引用 [F1] 时会被 verifier 忽略

        if doc_results:
            docs_parts.append("  <!-- 文档片段（回答时可引用 [1] [2] ...） -->")
            idx = 1
            for r in doc_results[:self.max_doc_in_prompt]:
                chapter_title = self._extract_chapter_title(r.content or "")
                # 在 XML 中显示章节标题，帮助 LLM 正确引用
                docs_parts.append(f'  <doc id="{idx}" type="document" source="{self._xml_escape(r.doc_name)}" chapter="{self._xml_escape(chapter_title)}">')
                docs_parts.append(f'    <content>{self._xml_escape(r.content or "")}</content>')
                docs_parts.append(f'  </doc>')
                source_map[str(idx)] = {
                    "type": "doc",
                    "doc_name": r.doc_name,
                    "page": r.doc_page,
                    "content": r.content,
                    "chapter_title": chapter_title,
                }
                idx += 1

        docs_xml = "\n".join(docs_parts)
        prompt = self._prompt_mgr.render_user_prompt(docs_xml=docs_xml, question=self._xml_escape(query))
        return prompt, source_map

    @staticmethod
    def _xml_escape(text: str) -> str:
        """转义 XML 特殊字符，防止破坏 Prompt 结构"""
        return html.escape(text, quote=True)

    @staticmethod
    def _extract_chapter_title(text: str) -> str:
        """
        从 parent_text 开头提取章节标题（支持完整路径）

        新格式（带完整路径）：
        "# 第六章 统一边疆 > 第一节 抗击东突厥"
        
        旧格式（仅小节）：
        "## 第一节 抗击东突厥"
        """
        lines = text.strip().split('\n')
        for line in lines[:5]:  # 只看前 5 行
            line = line.strip()
            if line.startswith('#'):
                # 去掉 markdown 标记和多余空格
                title = line.lstrip('#').strip()
                return title
        return ""

    @staticmethod
    def extract_citations(answer: str, source_map: SourceMap) -> List[Dict]:
        """
        从 LLM 回答中提取 [1]、[2]、[2·存疑] 等引用标记，关联到原始来源。

        Args:
            answer: LLM 生成的回答文本
            source_map: build_prompt 返回的编号映射表

        Returns:
            [{"id": "1", "type": "faq"/"doc", ...}, ...]
        """
        citations = []
        seen = set()
        # 匹配 [1] 或 [1·存疑]
        for match in re.finditer(r'\[(\d+)(?:·存疑)?\]', answer):
            cid = match.group(1)
            if cid not in seen and cid in source_map:
                seen.add(cid)
                citations.append({"id": cid, **source_map[cid]})
        return citations

    @staticmethod
    def _strip_llm_citation_footer(answer: str) -> str:
        """
        清理 LLM 自己在回答末尾生成的引用来源块（避免重复）
        """
        if not answer:
            return answer
        # 匹配各种 LLM 自发生成的引用块格式
        patterns = [
            r'\n---+\s*\n?\*\*引用来源：.*?$',           # --- 分隔符 + 引用来源
            r'\n\*\*引用来源：.*?$',                     # 换行后的引用来源
            r'\n来源[：:].*?$',                           # 换行后的来源
            r'引用来源[：:].*?$',                         # 行尾的引用来源
            r'\n参考文献[：:].*?$',                       # 换行后的参考文献
        ]
        for pat in patterns:
            answer = re.sub(pat, '', answer, flags=re.DOTALL)
        # 清理可能残留的 markdown 标记
        answer = re.sub(r'\*+\s*$', '', answer)
        return answer.strip()

    def _format_citation_footer(self, answer: str, source_map: SourceMap) -> Tuple[str, str, Dict[str, str]]:
        """
        格式化引用来源列表（方案 C：底部列出所有引用）

        在答案底部追加引用来源说明，并按 LLM 实际引用的顺序重新编号为 [1] [2] ...
        （避免用户看到不连续的编号如 [3] [4]）

        ---
        **引用来源：**
        [1] 《唐太宗传》· 第一节 抗击东突厥
        [2·存疑] 《唐太宗传》· 第二节 贞观司法（引用存疑）
        """
        # 提取所有引用标记（包括 [1] 和 [1·存疑]），保持出现顺序
        cited_ids_ordered = []
        seen = set()
        for match in re.finditer(r'\[(\d+)(?:·存疑)?\]', answer):
            cid = match.group(1)
            if cid not in seen and cid in source_map:
                seen.add(cid)
                cited_ids_ordered.append(cid)

        if not cited_ids_ordered:
            return answer, "", {}

        # 创建重映射表：原编号 → 新连续编号
        id_remap = {old_id: str(new_idx) for new_idx, old_id in enumerate(cited_ids_ordered, 1)}

        # 替换 answer 中的所有引用编号（包括 [3] 和 [3·存疑]）
        for old_id, new_id in id_remap.items():
            answer = answer.replace(f"[{old_id}·存疑]", f"[{new_id}·存疑]")
            answer = answer.replace(f"[{old_id}]", f"[{new_id}]")

        # 按新编号构建底部引用列表
        lines = ["\n---\n**引用来源：**"]
        for old_id in cited_ids_ordered:
            new_id = id_remap[old_id]
            source = source_map[old_id]
            chapter = source.get("chapter_title", "")
            doc_name = source.get("doc_name", "未知文献")
            
            # 简化文档名：去掉括号里的元数据
            doc_short = doc_name.split('(')[0].strip() if '(' in doc_name else doc_name
            # 去掉已有的书名号，避免双书名号
            doc_short = doc_short.strip('《》')

            # 检查状态（用新编号检查，因为 answer 已经被替换过了）
            if f"[{new_id}·存疑]" in answer:
                status = "（引用存疑）"
            else:
                status = ""

            if chapter:
                lines.append(f"[{new_id}] 《{doc_short}》· {chapter}{status}")
            else:
                lines.append(f"[{new_id}] 《{doc_short}》{status}")

        return answer, "\n".join(lines), id_remap
