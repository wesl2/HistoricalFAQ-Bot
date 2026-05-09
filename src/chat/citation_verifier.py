#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
引用一致性校验器（Citation Verifier）v2

功能：检查 LLM 生成的答案中，每个 [n] 引用是否确实支撑了它所在句子的核心事实。

核心问题：Citation Hallucination（引用幻觉）
- 模型知道正确答案（来自参数记忆）
- 但检索器没召回支撑该答案的片段
- 模型硬把 [1][2] 贴到无关片段上，假装有据可查

校验策略（当前实现）：
1. 实体硬对齐：用 jieba POS 提取人名、地名、数字、引语等"不可转述的硬事实"
2. 关键词匹配：检查硬事实是否出现在对应 chunk 中
3. 动态阈值：关系/推断类问题放宽，事实类问题收紧

预留扩展：
- 语义相似度兜底（embedding cosine similarity）
- 轻量 LLM Judge 抽检

用法：
    from src.chat.citation_verifier import verify_citations
    issues = verify_citations(answer, source_map)
    if issues:
        # 有虚假引用，需要处理
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# 预留：后续接入语义相似度层时取消注释
# from src.embedding.embedding_local_practice import get_embedding

logger = logging.getLogger(__name__)


@dataclass
class CitationIssue:
    """引用问题记录"""
    citation_id: str           # 引用编号，如 "1"
    sentence: str              # 包含该引用的句子
    claimed_keywords: List[str]  # 句子中声称的关键词
    actual_chunk: str          # 对应 chunk 的实际内容（前200字）
    match_score: float         # 匹配分数（0-1）
    severity: str              # "CRITICAL" | "WARNING"
    reason: str                # 问题原因描述


def _extract_cited_sentences(answer: str) -> List[Tuple[str, str]]:
    """
    提取答案中包含引用的句子

    Args:
        answer: LLM 生成的答案文本

    Returns:
        [(句子文本, 引用编号), ...]
        例如：[("义仓税率为亩税二升 [1]", "1")]
    """
    sentences = []
    # 按句号/问号/感叹号分句
    for sent in re.split(r'(?<=[。！？\n])\s*', answer):
        # 提取该句中的所有引用 [n]
        refs = re.findall(r'\[(\d+)\]', sent)
        for ref in refs:
            sentences.append((sent.strip(), ref))
    return sentences


def _is_core_entity(term: str) -> bool:
    """
    判断一个term是否属于"核心实体"（人名、地名、引语等）。
    核心实体的匹配在分数计算中占更高权重。
    """
    return len(term) >= 3 or '·' in term


def _extract_key_terms(sentence: str) -> List[str]:
    """
    从句子中提取硬事实（Hard Facts）用于引用校验

    策略：提取"不可转述"的实体信息，忽略 LLM 的现代汉语措辞。
    
    硬事实包括：
    1. 人名（nr）：唐太宗、魏征、李靖
    2. 地名/机构（ns/nt）：长安、高昌、三省六部
    3. 数字（m）：贞观二年、亩税二升
    4. 直接引语：加引号的内容（"以人为镜"）
    5. 专有名词（nz/n）：均田令、玄武门、六部
    
    不提取（因为 LLM 会转述）：
    - 动词（v）：视为、帮助、认识
    - 抽象名词：关键人物、关系、措施
    """
    import jieba.posseg as pseg

    # 先去掉引用标记，避免 [1] 被分词成 [、1、]
    sentence = re.sub(r'\[\d+\]', '', sentence)

    # 预定义 skip_words（引语提取和后续过滤都需要）
    skip_words = {
        '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.',
        '第一', '第二', '第三', '第四', '第五',
        '首先', '其次', '最后', '总之', '此外', '同时',
        '根据', '关于', '有关', '以及', '或者', '但是',
        '可以', '进行', '通过', '需要', '成为', '形成',
        '说明', '表明', '反映', '体现', '显示',
        '具有', '存在', '发生', '产生', '导致',
        '因此', '所以', '由于', '因为', '虽然',
        '直接', '间接', '核心', '重要', '主要',
        '具体', '详细', '进一步', '一定程度上',
        '之一',
    }

    terms = []

    # 1. 提取直接引语（加引号的内容，不可转述）
    # 分别处理四种引号：英文双引、中文双引、英文单引、中文单引
    quotes = []
    quotes.extend(re.findall(r'["""](.+?)["""]', sentence))      # 英文双引号
    quotes.extend(re.findall(r'[""](.+?)[""]', sentence))        # 中文双引号
    quotes.extend(re.findall(r"['''](.+?)[''']", sentence))      # 英文单引号
    quotes.extend(re.findall(r'[''](.+?)['']', sentence))        # 中文单引号
    for q in quotes:
        # 引语本身就是原文，不过滤词性，直接保留长度>=2的词
        # （jieba 对古文 POS 标注不准，如"以人为镜"标成 l，"得失"标成 v）
        for word, flag in pseg.cut(q):
            if len(word) >= 2 and word not in skip_words:
                terms.append(word)

    # 2. 提取数字（年份、税率、数量——不可转述）
    numbers = re.findall(
        r'(?:\d{1,4}|(?:[一二三四五六七八九十百千]+))(?:年|月|日|升|石|顷|亩|人|户|卷)?',
        sentence
    )
    terms.extend(numbers)

    # 3. 用 jieba 分词提取硬事实实体
    # 保留：名词(n)、专有名词(nr/ns/nt/nz)、数词、量词
    # 排除：动词(v/vn)、形容词(a/ad)、副词(d)
    keep_flags = {'n', 'nr', 'ns', 'nt', 'nz', 'm', 'q'}
    for word, flag in pseg.cut(sentence):
        if flag[0] in keep_flags and len(word) >= 2:
            terms.append(word)

    # 4. 过滤 skip_words
    filtered_terms = [t for t in terms if t not in skip_words]

    # 5. 去重并保持顺序
    seen = set()
    unique_terms = []
    for t in filtered_terms:
        if t not in seen and len(t) >= 2:
            seen.add(t)
            unique_terms.append(t)

    return unique_terms[:15]  # 最多取 15 个


def _keyword_match(sentence: str, chunk_text: str) -> Tuple[float, List[str], List[str]]:
    """
    关键词匹配校验（硬事实优先版）

    策略：
    1. 提取关键词（硬事实）
    2. 区分"核心实体"（人名、地名、引语）和"辅助词"
    3. 只要核心实体有匹配，就视为有效引用（降低 false negative）

    注意：generic_terms 采用硬编码而非动态计算。
    原因：动态计算会过度过滤（如把"李世民""皇后"也当成通用词），
    导致 false positive 激增。硬编码在特定领域（唐史）中更可控。
    换朝代/加新书时手动更新即可。
    """
    key_terms = _extract_key_terms(sentence)
    if not key_terms:
        return 1.0, [], []

    # 通用干扰词：在所有片段中都出现，匹配了也不说明问题
    # 硬编码：针对《唐太宗传》等唐史资料。换朝代时需更新。
    generic_terms = {
        '贞观', '唐太宗', '太宗', '唐初', '唐朝', '唐代',
        '皇帝', '大臣', '诏令', '政策', '制度',
        '根据参考资料', '参考资料', '史料记载',
    }

    core_terms = [t for t in key_terms if t not in generic_terms]
    if not core_terms:
        return 1.0, [], []

    matched = []
    unmatched = []
    core_matched = 0  # 核心实体匹配数
    core_total = 0    # 核心实体总数
    
    for term in core_terms:
        is_core = _is_core_entity(term)
        if is_core:
            core_total += 1
        if term in chunk_text:
            matched.append(term)
            if is_core:
                core_matched += 1
        else:
            unmatched.append(term)

    # 计算分数：核心实体匹配度优先
    if core_total > 0 and core_matched > 0:
        # 如果有核心实体匹配，基础分至少 0.5
        base_score = max(0.5, core_matched / core_total)
    else:
        base_score = len(matched) / len(core_terms) if core_terms else 1.0
    
    score = base_score
    return score, matched, unmatched


def _resolve_threshold(answer: str, base_threshold: float) -> float:
    """
    动态阈值：关系/推断类问题放宽，事实类问题收紧。
    
    历史 FAQ 中两类问题差异很大：
    - 事实型："义仓税率是多少？" → 要求硬事实严格匹配
    - 关系型："三省六部制与均田制的关系？" → 允许合理推断
    """
    # 关系/推断类关键词
    relation_keywords = ['关系', '影响', '作用', '联系', '区别', '比较', '意义']
    if any(w in answer for w in relation_keywords):
        return max(0.15, base_threshold - 0.1)
    return base_threshold


def verify_citations(
    answer: str,
    source_map: Dict[str, dict],
    doc_results: Optional[List] = None,
    threshold: float = 0.3,
) -> List[CitationIssue]:
    """
    校验答案中的引用一致性

    Args:
        answer: LLM 生成的答案文本
        source_map: build_prompt 返回的 {id: {type, content, ...}} 映射
        doc_results: 检索器返回的 DocResult 列表（fallback 补全用）
        threshold: 关键词匹配最低分数，低于此值视为问题引用

    Returns:
        CitationIssue 列表，空列表表示所有引用都通过校验
    """
    issues = []
    cited_sentences = _extract_cited_sentences(answer)

    if not cited_sentences:
        logger.debug("[CitationVerifier] 答案中无引用标记，跳过校验")
        return issues

    # 动态阈值
    effective_threshold = _resolve_threshold(answer, threshold)

    logger.info(
        "[CitationVerifier] 开始校验 | 引用数=%d | 阈值=%.2f",
        len(cited_sentences), effective_threshold
    )

    for sentence, ref_id in cited_sentences:
        if ref_id not in source_map:
            issues.append(CitationIssue(
                citation_id=ref_id,
                sentence=sentence,
                claimed_keywords=[],
                actual_chunk="引用编号不存在",
                match_score=0.0,
                severity="CRITICAL",
                reason=f"引用 [{ref_id}] 在 source_map 中不存在（可能模型编造了编号）"
            ))
            continue

        source = source_map[ref_id]
        chunk_text = source.get("content") or source.get("answer", "")

        # fallback：如果 source_map 内容为空，尝试从 doc_results 补全
        if not chunk_text.strip() and doc_results:
            for doc in doc_results:
                if str(getattr(doc, 'id', '')) == ref_id:
                    chunk_text = getattr(doc, 'content', '') or getattr(doc, 'text', '')
                    break

        # 边界保护：chunk 为空或过短无法校验
        if not chunk_text or len(chunk_text.strip()) < 20:
            issues.append(CitationIssue(
                citation_id=ref_id,
                sentence=sentence,
                claimed_keywords=[],
                actual_chunk=chunk_text[:200],
                match_score=0.0,
                severity="WARNING",
                reason="引用片段内容为空或过短，无法校验"
            ))
            logger.warning(
                "[CitationVerifier] [WARNING] 引用 [%s] 片段过短 | 长度=%d",
                ref_id, len(chunk_text.strip()) if chunk_text else 0
            )
            continue

        # 关键词匹配
        score, matched, unmatched = _keyword_match(sentence, chunk_text)

        if score < effective_threshold:
            severity = "CRITICAL" if score < 0.1 else "WARNING"
            issues.append(CitationIssue(
                citation_id=ref_id,
                sentence=sentence,
                claimed_keywords=unmatched,
                actual_chunk=chunk_text[:200],
                match_score=score,
                severity=severity,
                reason=f"句子关键词与 chunk 匹配度仅 {score:.0%}。"
                       f"句子声称: {unmatched}，但 chunk 中未找到。"
            ))
            logger.warning(
                "[CitationVerifier] [%s] 引用 [%s] 不匹配 | "
                "句子: %s... | 未匹配: %s",
                severity, ref_id, sentence[:60], unmatched
            )
        else:
            logger.debug(
                "[CitationVerifier] 引用 [%s] 通过 | 匹配度 %.0f%% | 匹配: %s",
                ref_id, score * 100, matched
            )

    return issues


def format_issues(issues: List[CitationIssue]) -> str:
    """
    将校验问题格式化为可读的报告

    Args:
        issues: CitationIssue 列表

    Returns:
        格式化的问题报告字符串
    """
    if not issues:
        return "✅ 所有引用均通过一致性校验"

    lines = [f"⚠️ 发现 {len(issues)} 个引用问题：", ""]
    for i, issue in enumerate(issues, 1):
        lines.append(f"--- 问题 {i} [{issue.severity}] ---")
        lines.append(f"引用编号: [{issue.citation_id}]")
        lines.append(f"问题句子: {issue.sentence[:100]}...")
        lines.append(f"声称内容: {issue.claimed_keywords}")
        lines.append(f"实际片段: {issue.actual_chunk[:150]}...")
        lines.append(f"匹配分数: {issue.match_score:.0%}")
        lines.append(f"原因: {issue.reason}")
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    # 测试用例 1：义仓税率（事实型，应严格匹配）
    test_answer = (
        '根据参考资料，贞观二年四月，唐太宗下诏设置义仓，'
        '其税率规定为"亩税二升" [1][2]。'
    )

    test_source_map = {
        "1": {
            "type": "doc",
            "content": "## 第三节 重农政策的具体措施..."
                       "武德七年四月，唐高祖颁布均田令，"
                       "规定：丁男、中男给（田）一顷...",
        },
        "2": {
            "type": "doc",
            "content": "## 第三节 德化政策..."
                       "唐太宗对边疆少数民族实行德化政策...绥之以德...",
        },
    }

    print("=" * 60)
    print("测试 1：义仓税率（事实型）")
    print("=" * 60)
    issues = verify_citations(test_answer, test_source_map)
    print(format_issues(issues))

    # 测试用例 2：带中文引号（验证引号正则修复）
    test_answer2 = '唐太宗说："以人为镜，可以明得失。" [1]'
    test_source_map2 = {
        "1": {"type": "doc", "content": "...以铜为镜，可以正衣冠..."},
    }
    print("\n" + "=" * 60)
    print("测试 2：中文引号提取")
    print("=" * 60)
    terms = _extract_key_terms('唐太宗说："以人为镜，可以明得失。"')
    print(f"提取的硬事实: {terms}")
    issues2 = verify_citations(test_answer2, test_source_map2)
    print(format_issues(issues2))

    # 测试用例 3：核心实体统计（验证空字符串 bug 修复）
    print("\n" + "=" * 60)
    print("测试 3：核心实体统计")
    print("=" * 60)
    score, matched, unmatched = _keyword_match(
        "魏征谏言二百余事。",
        "魏征以直言敢谏闻名，所陈谏多达二百余事。"
    )
    print(f"score={score:.2f}, matched={matched}, unmatched={unmatched}")
    assert score > 0.5, "核心实体应正确匹配"
    print("✅ 核心实体统计正确")
