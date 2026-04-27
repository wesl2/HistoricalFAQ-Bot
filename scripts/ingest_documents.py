#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档数据导入脚本（生产级 RAG 预处理流水线）

将 PDF/TXT/DOCX 等原始文档经过清洗、去噪、语义分块、上下文增强、批量 Embedding 后，
增量或全量导入 PostgreSQL 的 doc_chunks 表。

用法:
    # 全量导入（默认先清空表）
    python scripts/ingest_documents.py ./data/raw

    # 增量追加（不删除已有数据，仅按文件 MD5 更新变化的文档）
    python scripts/ingest_documents.py ./data/raw --append

    # 指定分块大小和重叠
    python scripts/ingest_documents.py ./data/raw --chunk-size 1024 --overlap 128

【修改留痕 - 2026-04-17】
- 由旧版简单导入脚本升级，融合 /root/autodl-tmp/RAG_Test/FalgEmbedding_test/ 项目中的
  OCR 清洗（wash_ocr）、去噪（denoise）、数据清洗（clean_data）逻辑。
- 引入 2024-2025 工业界 RAG 最佳实践：
  1. 语义化递归分块（RecursiveCharacterTextSplitter）
  2. 上下文增强（Contextual Chunk Headers）
  3. 批量 Embedding（利用 BGE-M3 批处理能力）
  4. 增量同步（--append 模式）
  5. 内容去重与长度过滤
  6. 哈希校验与进度条

【修改留痕 - 2026-04-17（二次修复）】
根据架构师级 code review 反馈，修复 3 个生产环境隐患：
1. 硬编码业务数据 -> 抽离到 config/custom_dict_replace.json。
2. 全零向量兜底 -> 单条 embedding 失败返回 None，上游直接丢弃该 Chunk。
3. MD5 粒度过细 -> 新增文件级 MD5 缓存（.ingest_cache.json），未变文件直接跳过。

【修改留痕 - 2026-04-17（三次修复）】
根据第二次 code review 反馈，修复 3 个代码 bug：
1. wash_ocr：将 `line.replace(" ", "")` 改成条件性删除，只处理全角空格和中文字间的 OCR 空格，
   避免破坏英文、数字、URL 中的正常半角空格。
2. normalize：删除 `text.replace(r"\n", "\n")` 这一无意义的 bug 行。
3. compute_embeddings_batch：修掉 `isinstance(batch, str)` 这个永远为 False 的判断，
   改为通过 result 结构稳健区分批量/单条返回。
"""

# =============================================================================
# 第一部分：标准库导入
# =============================================================================
# argparse    : 用于解析命令行参数，例如 `python ingest.py ./data --append`
# hashlib     : 提供 MD5 等哈希算法，用于计算文件指纹和内容去重
# json        : 读写 JSON 格式文件（缓存、配置）
# logging     : 记录程序运行日志，替代 print 以便控制输出级别
# os          : 操作系统接口，用于路径拼接、环境变量等
# re          : 正则表达式引擎，用于文本去噪中的模式匹配
# sys         : 系统相关功能，例如修改 Python 模块搜索路径 sys.path
# pathlib.Path: 面向对象的路径操作，比字符串拼接更安全
# typing      : 类型提示（List, Tuple, Optional 等），提升代码可读性和 IDE 补全
import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# =============================================================================
# 第二部分：第三方库导入（带降级处理）
# =============================================================================
# tqdm 是一个轻量级进度条库。在批量处理大文档时，能让用户直观看到进度。
# 用 try/except 包裹：如果当前环境没有安装 tqdm，就令 tqdm = None，
# 后续代码检测到 None 时自动回退为普通循环，避免因为缺少依赖就直接崩溃。
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# =============================================================================
# 第三部分：项目模块导入
# =============================================================================
# 这一步非常关键：把项目的根目录（即 HistoricalFAQ-Bot/）插入到 sys.path 的最前面。
# 为什么要在最前面？因为 Python 导入模块时按 sys.path 顺序查找，
# 放在前面可以确保优先加载项目内的 src/、config/，而不是系统 Python 路径里
# 可能存在的同名旧包。
# `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` 的含义：
#   __file__ 是当前脚本的路径 scripts/ingest_documents.py
#   第一层 dirname -> scripts/
#   第二层 dirname -> 项目根目录 HistoricalFAQ-Bot/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# 兼容导入策略：优先正式版模块，失败时回退到 _practice 版本
# ---------------------------------------------------------------------------
# 你的项目经历多次重构，有些模块同时存在正式版和练习版（带 _practice 后缀）。
# 用 try/except 做自适应导入，确保无论当前环境是哪种状态都能跑通。
try:
    # 正式版：本地 BGE-M3 embedding 计算接口
    from src.embedding.embedding_local_practice import get_embedding
    # 正式版：PostgreSQL 连接池
    from src.vectorstore.pg_pool import get_connection
except ImportError:
    # 若正式版不存在（比如你还在练习阶段），自动回退到 practice 版
    from src.embedding.embedding_local_practice import get_embedding
    from src.vectorstore.pg_pool_practice import get_connection

# DocumentProcessor 是项目内置的文档处理器，封装了 LangChain 的加载器和分块器
from src.data_pipeline.document_processor import DocumentProcessor

# 配置导入：PG_DOC_TABLE 是文档片段表名，EMBEDDING_CONFIG 包含模型参数
from config.pg_config import PG_DOC_TABLE
from config.model_config_practice import EMBEDDING_CONFIG

# =============================================================================
# 第四部分：日志配置
# =============================================================================
# logging.basicConfig 是 Python 标准库日志的初始化入口。
# level=logging.INFO 表示只输出 INFO 及以上级别的日志（DEBUG 会被过滤掉）。
# format 定义了日志格式：时间 - 记录器名 - 日志级别 - 消息内容。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# 获取以当前模块名命名的 logger 实例。后续用 logger.info() / logger.warning() 等输出。
logger = logging.getLogger(__name__)

# =============================================================================
# 第五部分：全局常量
# =============================================================================
# CACHE_FILE 定义了文件级 MD5 缓存的本地存储位置。
# Path(__file__) 获取当前脚本的绝对路径对象，.parent.parent 往上跳两级：
#   scripts/ingest_documents.py -> scripts/ -> 项目根目录
# 缓存文件放在项目根目录下，名为 .ingest_cache.json（带点号表示隐藏文件）。
# 它的作用是：记录每个文档文件名及其最近一次成功处理时的 MD5 哈希。
# 下次增量导入时，若文件哈希没变，就可以直接跳过，避免重复调用 BGE 模型。
CACHE_FILE = Path(__file__).parent.parent / ".ingest_cache.json"


# ############################################################################
# 模块一：文件级 MD5 缓存工具
# ############################################################################
#
# 为什么需要文件级缓存？
#   BGE-M3 模型推理非常耗时。如果你有 100 个文档，只修改了其中 1 个，
#   理想情况下应该只重新处理那 1 个，其余 99 个直接跳过。
#   这就是增量同步（Incremental Sync）的核心思想。
#
# 实现方式：
#   在 process_single_file() 的最开头计算整个文件的 MD5（文件级 hash）。
#   和 .ingest_cache.json 里记录的上次 hash 对比，一致则 return 0。


def load_hash_cache() -> dict:
    """
    加载文件哈希缓存，返回一个字典：{doc_name: md5_hash}。
    如果缓存文件不存在或读取失败，返回空字典 {}。
    """
    if not CACHE_FILE.exists():
        return {}  # 第一次运行，还没有缓存
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # 缓存文件损坏（比如写了一半程序挂了），不抛异常，直接当作无缓存
        return {}


def save_hash_cache(cache: dict):
    """
    将文件哈希缓存写回磁盘。
    这个操作放在所有文档处理完成后执行，一次性写入，减少 IO 次数。
    若写入失败只打 warning，不阻断主流程（缓存失败是“可降级”的）。
    """
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"缓存保存失败（非致命）: {e}")


def compute_file_hash(file_path: str) -> str:
    """
    计算文件的 MD5 哈希值（十六进制字符串）。
    采用流式读取（每次 8KB），避免一次性把大文件读进内存。
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        # iter(lambda: f.read(8192), b"") 是 Python 中读取文件直到 EOF 的惯用写法：
        # 每次调用 lambda 读取 8192 字节，直到返回空字节串 b"" 时停止迭代。
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ############################################################################
# 模块二：TextCleaner 文本清洗器
# ############################################################################
# RAG 系统的质量上限取决于原始数据的质量。
# 如果输入文本充满 OCR 断行、PDF 页眉页脚、电子书水印，
# 那么无论检索模型多强，召回的 chunk 里都会混入大量噪声。
# TextCleaner 的职责就是：在进入分块和 Embedding 之前，把文本洗到"人类可读"的水平。


class TextCleaner:
    """
    RAG 预处理文本清洗器。
    包含三个核心阶段：
      1. wash_ocr   : OCR 后处理（断行重组、字间空格处理）
      2. denoise    : 去噪（页眉页脚、水印、出版信息）
      3. normalize  : 格式化（统一空白、压缩多余换行）
    """

    # _custom_replacements 是类级缓存，避免每次 denoise 都重新读 JSON 文件。
    # 第一次调用 _load_custom_replacements() 后会被赋值为 List[dict]，
    # 后续直接复用，减少磁盘 IO。
    _custom_replacements = None

    @classmethod
    def _load_custom_replacements(cls) -> List[dict]:
        """
        加载 config/custom_dict_replace.json 中的特定事实替换规则。
        这是一个"业务纠错字典"，与通用清洗逻辑解耦。
        例如：{"old": "一九七八年...", "new": "一九七六年..."}
        为什么要抽离？因为不同项目的纠错需求完全不同，
        通用代码里不应该硬编码任何特定文档的事实修正。
        """
        # 如果已经加载过，直接返回缓存，不再读文件
        if cls._custom_replacements is not None:
            return cls._custom_replacements

        custom_dict_path = Path(__file__).parent.parent / "config" / "custom_dict_replace.json"
        if not custom_dict_path.exists():
            # 配置文件不存在（比如用户还没创建），返回空列表，不影响主流程
            cls._custom_replacements = []
            return cls._custom_replacements

        try:
            with open(custom_dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cls._custom_replacements = data.get("replacements", [])
        except Exception as e:
            logger.warning(f"加载 custom_dict_replace.json 失败: {e}，跳过自定义替换")
            cls._custom_replacements = []

        return cls._custom_replacements

    @staticmethod
    def wash_ocr(text: str) -> str:
        """
        OCR 后处理：将 OCR 扫描结果中常见的断行、字间空格修复为正常段落。
        设计来源：旧项目 wash_ocr.py

        OCR 扫描的 PDF/图片经常会出现以下症状：
          - 每一行末尾强制换行，把完整句子切成多行
          - 中文字符之间出现半角空格（如 "王 洪 文"）
          - 混入全角空格 \u3000

        本函数的策略：
          1. 按行读取，尝试在句末标点（。！？等）处合并断行，恢复完整句子。
          2. 删除全角空格。
          3. 条件性删除半角空格：仅当中文字符之间出现单个空格时才删除，
             保护英文单词、数字、URL 中的正常空格。
        """
        lines = text.splitlines()  # 按 \n 拆分成行列表
        clean_paragraphs = []      # 存放重组后的完整段落
        buffer = ""                # 暂存还没遇到句末标点的断行片段

        for line in lines:
            line = line.strip()    # 去掉行首行尾空白
            if not line:
                continue           # 跳过空行

            # --- 空格处理（2026-04-17 修复）---
            # 全角空格（\u3000）在任何情况下都应删除
            line = line.replace("\u3000", "")

            # 条件性去除 OCR 字间空格：
            # 正则 `(?<=[\u4e00-\u9fa5]) (?=[\u4e00-\u9fa5])` 的含义：
            #   (?<=...)  是"正向后行断言"，表示左边必须是一个中文字符
            #   [\u4e00-\u9fa5] 是中文 Unicode 范围
            #   空格      表示匹配一个半角空格
            #   (?=...)   是"正向前行断言"，表示右边必须是一个中文字符
            # 合起来：只匹配"中文字符 + 半角空格 + 中文字符"这种模式，
            # 例如 "王 洪 文" 里的空格会被删掉，变成 "王洪文"。
            # 但 "Deep Seek"（英文字母之间）或 "2024 04 17"（数字之间）不会匹配，
            # 因此它们的正常空格会被保留。
            line = re.sub(r'(?<=[\u4e00-\u9fa5]) (?=[\u4e00-\u9fa5])', '', line)

            # --- 断行重组逻辑 ---
            # 定义一组"句末标点"，如果一行以这些标点结尾，就认为是一个完整句子的结束。
            end_punctuations = ("。」", "」", "。", "！", "？", "”", "…", "!", "?"
            )
            if line.endswith(end_punctuations):
                # 这一行是句尾，把 buffer 中暂存的断行片段和它拼起来
                full_sentence = buffer + line
                clean_paragraphs.append(full_sentence)
                buffer = ""  # 清空缓冲区，准备组装下一段
            else:
                # 这一行不是句尾，可能是段落中间被强行换行了，先存进 buffer
                buffer += line

        # 循环结束后，如果 buffer 里还有剩余内容（最后一行没有句末标点），
        # 把它作为独立段落追加进去，避免内容丢失。
        if buffer:
            clean_paragraphs.append(buffer)

        # 用 \n 把重组后的段落重新连接成字符串返回。
        # 注意这里保留 \n，因为 denoise 和 normalize 还会进一步处理。
        return "\n".join(clean_paragraphs)

    @classmethod
    def denoise(cls, text: str) -> str:
        """
        去噪：切除页眉页脚、出版信息、PDF 水印、页码标记等。
        设计来源：旧项目 denoise.py + clean_data.py

        这一层处理的是"版式噪声"，和 wash_ocr 处理的"OCR 断行噪声"是互补的。
        即使你的文本不是 OCR 来的（比如直接复制的电子书），也可能带有：
          - PDF 页码标记（RN(7)、pp《4）
          - 出版信息水印（Anna's Archive）
          - 章节标题和页码粘连在一起
        """
        # --- PDF 页码/页眉标记 ---
        # 匹配模式如：RN(7)、RN《16)、pp《4
        text = re.sub(r"RN[（\(《]\d+[）\)）。]", "", text)
        text = re.sub(r"RN\(\d+\)\)", "", text)
        text = re.sub(r"pp[《<]\d+", "", text)

        # --- 出版信息和页码引用块 ---
        # 匹配模式如：@青野...一九九三年版(Page#508)
        # flags=re.DOTALL 让 . 也能匹配换行符，防止跨行匹配失败
        text = re.sub(r"@.*?\(Page#\d+\)", "", text, flags=re.DOTALL)
        text = re.sub(r"\(Page#\d+\)", "", text)

        # --- 分隔符长线 ---
        # 连续 5 个以上的等号通常是手动分隔线或 PDF 转换残留
        text = re.sub(r"={5,}", "", text)

        # --- 混入正文的章节标题+页码 ---
        # 匹配模式如：第十四章武装叛乱的失获485
        # [一二三四五六七八九十百]+ 匹配中文数字序列
        text = re.sub(r"第[一二三四五六七八九十百]+章.*?\d+", "", text)

        # --- 奇怪章节头（特定项目历史残留）---
        text = re.sub(r"总序站\)", "", text)
        text = re.sub(r"卷首语[;；]", "", text)
        text = re.sub(r"急挡“标准属”", "", text)

        # --- 电子书水印（Anna's Archive 等）---
        text = re.sub(
            r"Document\s*generated\s*by\s*Anna's\s*Archive.*",
            "",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r"Some\s*stats\s*\(more\s*in\s*the\s*PDF\s*attachments\).*",
            "",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(r"filename[:_].*zip", "", text, flags=re.IGNORECASE)

        # --- 特定事实修正（从外部 JSON 加载，不再硬编码）---
        for rule in cls._load_custom_replacements():
            old = rule.get("old")
            new = rule.get("new")
            if old and new and old in text:
                text = text.replace(old, new)

        return text

    @classmethod
    def normalize(cls, text: str) -> str:
        """
        格式化：统一换行、去除首尾空格、删除空行。

        2026-04-17 修复：删除了 bug 行 `text.replace(r"\n", "\n")`。
        原代码的问题：r"\n" 在 Python 中表示"反斜杠 + n"两个字符的字面量，
        它不会匹配真实的换行符（\n 的 ASCII 码是 0x0A），因此这一行没有任何实际作用。
        如果目的是替换文本里出现的字面量 `\n`（两个字符），应该写成 `.replace("\\n", "\n")`。
        但当前数据里没有这种需求，所以直接删除该行。
        """
        # 多个连续换行压缩为单个，避免段落之间出现大量空行
        text = re.sub(r"\n+", "\n", text)
        # 去掉字符串首尾的空格和换行
        text = text.strip()
        return text

    @classmethod
    def clean(cls, text: str) -> str:
        """
        完整清洗流程：OCR 重组 -> 去噪 -> 格式化。
        这是 TextCleaner 的对外统一入口，调用方只需要调用这一个方法。
        """
        text = cls.wash_ocr(text)   # 第一步：修复 OCR 断行和字间空格
        text = cls.denoise(text)    # 第二步：切除页眉页脚、水印等版式噪声
        text = cls.normalize(text)  # 第三步：统一空白、压缩换行
        return text


# ############################################################################
# 模块三：文档加载与语义分块
# ############################################################################


def load_and_clean_documents(file_path: str) -> List:
    """
    加载文档，并对每页/每段内容进行清洗。
    返回 LangChain Document 对象列表（带 metadata）。

    为什么不直接 open() 读 txt？
      因为 DocumentProcessor 封装了多格式支持（PDF、TXT、DOCX），
      并且会自动提取 metadata（如 PDF 的页码）。
    """
    processor = DocumentProcessor()
    # load_document 内部会根据后缀名选择对应的 Loader（PyPDFLoader / TextLoader / DocxLoader）
    raw_docs = processor.load_document(file_path)
    if not raw_docs:
        return []  # 加载失败或文件为空

    # Path(file_path).stem 提取文件名（不带后缀），作为 doc_name
    # 例如 "/data/raw/唐史.txt" -> "唐史"
    doc_name = Path(file_path).stem
    cleaned_docs = []

    for i, doc in enumerate(raw_docs):
        # 对每一页/每一段调用完整清洗流程
        cleaned_text = TextCleaner.clean(doc.page_content)
        if not cleaned_text:
            continue  # 清洗后为空，跳过

        # --- 页码处理 ---
        # PyPDFLoader 会在 metadata 里放 "page" 字段，值是 0 起始的页码。
        # 为了和人类阅读习惯一致，转成 1 起始。
        page_num = doc.metadata.get("page", 0)
        if isinstance(page_num, int):
            page_num += 1
        else:
            page_num = 1

        # 把清洗后的文本写回 Document 对象
        doc.page_content = cleaned_text
        # 补充 metadata，供后续分块和入库使用
        doc.metadata.update({
            "doc_name": doc_name,
            "doc_page": page_num,
            "source": file_path,
        })
        cleaned_docs.append(doc)

    return cleaned_docs


def semantic_chunk_documents(
    documents: List,
    chunk_size: int = 1024,
    chunk_overlap: int = 128
) -> List:
    """
    语义递归分块。
    使用 RecursiveCharacterTextSplitter，优先在段落边界、句子边界切分，
    而不是粗暴的固定长度截断。

    为什么要用递归分块？
      固定长度切分（如每 512 字符一刀切）很容易把一句话从中间切断，
      导致 chunk 的前半句和后半句语义不连贯，向量检索质量下降。
      RecursiveCharacterTextSplitter 会按优先级尝试多种分隔符：
      段落(\n\n) -> 换行(\n) -> 句子(。！？) -> 空格 -> 字符，
      尽量保证切分点落在语义边界上。

    Args:
        documents : LangChain Document 列表
        chunk_size: 每个 chunk 的目标字符长度（不是 token 数，是字符数）
        chunk_overlap: 相邻 chunk 之间的重叠字符数，防止边界信息丢失
    """
    processor = DocumentProcessor()
    chunks = processor.split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter_type="recursive"
    )

    # --- 长度过滤 ---
    # 丢弃过短的碎块（< 50 字符）。
    # 这些碎块通常是标题、页码残留、或清洗后的边角料，
    # 对检索和回答几乎没有任何价值，还会浪费 Embedding 计算资源。
    filtered = [c for c in chunks if len(c.page_content.strip()) >= 50]
    if len(filtered) < len(chunks):
        logger.info(f"过滤短碎块：原始 {len(chunks)} -> 保留 {len(filtered)}")

    return filtered


# ############################################################################
# 模块四：上下文增强（Contextual Chunk Headers）
# ############################################################################
# Anthropic 在 2024 年提出 Contextual Retrieval，核心思想是：
# 原始 chunk 可能缺乏上下文（比如一句"他于当天去世"，模型不知道"他"是谁）。
# 通过在 chunk 前 prepend 来源信息，让 embedding 捕获更多语义。
# 同时这对 BM25 关键词检索也有帮助：用户搜"唐史"时，即使正文没出现这个词，
# 头部信息也能让 BM25 命中。


def enrich_chunk(chunk) -> str:
    """
    为 chunk 添加上下文头部信息，提升检索质量。
    参考 Anthropic 2024 Contextual Retrieval。
    """
    doc_name = chunk.metadata.get("doc_name", "未知文档")
    doc_page = chunk.metadata.get("doc_page", 0)
    content = chunk.page_content.strip()

    # 构建简短的上下文头
    header = f"【来源：《{doc_name}》第{doc_page}页】"
    # 用换行符把头部和正文隔开，既清晰又不影响语义
    return f"{header}\n{content}"


# ############################################################################
# 模块五：批量 Embedding 计算
# ############################################################################
# BGE-M3 等模型支持批量输入（一次传入多条文本，并行编码），
# 比逐条调用快 3-5 倍。但批量也有风险：如果 batch 里某条文本异常（超长、乱码），
# 可能导致整批失败。因此需要"批量优先、失败降级"策略。


def compute_embeddings_batch(
    texts: List[str],
    batch_size: int = None
) -> List[Optional[List[float]]]:
    """
    批量计算 Embedding，带错误降级处理。
    若批量失败，自动降级为逐条计算，避免整批作废。

    2026-04-17 修复：单条失败时返回 None，由上游丢弃该 Chunk，
    不再塞入全零向量（防止 PGVector 余弦相似度除以零）。

    Args:
        texts     : 待编码的文本列表
        batch_size: 每批处理的条数，默认从 EMBEDDING_CONFIG 读取（通常是 8）

    Returns:
        List[Optional[List[float]]]: 每个位置对应一个向量，失败则为 None
    """
    if batch_size is None:
        batch_size = EMBEDDING_CONFIG.get("batch_size", 8)

    all_vectors: List[Optional[List[float]]] = []
    # 把 texts 按 batch_size 切成若干段
    iterator = range(0, len(texts), batch_size)
    if tqdm is not None:
        # 用 tqdm 包装，显示 Embedding 进度条
        iterator = tqdm(list(iterator), desc="Embedding 批次")

    for i in iterator:
        batch = texts[i: i + batch_size]
        try:
            # get_embedding 的接口约定：
            #   输入 List[str] -> 返回 List[List[float]]（批量）
            #   输入 str      -> 返回 List[float]（单条）
            result = get_embedding(batch)

            # 2026-04-17 修复：稳健判断 result 是批量还是单条返回。
            # 逻辑：如果 result 是 list，且第一个元素也是 list，说明是批量返回。
            # 否则就是单条返回（List[float]），用 append 加入。
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                all_vectors.extend(result)  # 展开 8 条向量
            else:
                all_vectors.append(result)   # 单条向量
        except Exception as e:
            # 整批失败（可能某条文本超长触发了模型报错）
            logger.warning(f"批量 embedding 失败（batch {i}）: {e}，降级为逐条处理")
            for single_text in batch:
                try:
                    # 降级：逐条调用。单条失败只影响自己，不影响其他条
                    vec = get_embedding(single_text)
                    all_vectors.append(vec)
                except Exception as e2:
                    # 单条也失败：记录错误，返回 None，让上游丢弃这个 Chunk
                    logger.error(f"单条 embedding 失败: {e2}，已丢弃该 Chunk")
                    all_vectors.append(None)

    return all_vectors


# ############################################################################
# 模块六：数据库写入
# ############################################################################


def insert_chunks_to_db(
    records: List[Tuple],
    append: bool = False,
    doc_name: str = None
):
    """
    将文档块批量写入 PostgreSQL 的 doc_chunks 表。

    Args:
        records : List[Tuple]，每个元组为
                  (chunk_text, chunk_vector_str, doc_name, doc_page, chunk_index)
        append  : True 表示增量模式（先删旧再插新），False 表示全量模式（先 TRUNCATE）
        doc_name: 增量模式下用于定位旧记录的文档名
    """
    if not records:
        logger.info("没有记录需要写入数据库")
        return

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            if not append:
                # 全量模式：清空整个表。CASCADE 表示级联删除依赖它的外键记录。
                cursor.execute(f"TRUNCATE TABLE {PG_DOC_TABLE} CASCADE")
                conn.commit()
                logger.info(f"全量模式：已清空 {PG_DOC_TABLE}")
            elif doc_name:
                # 增量模式：先删除该文档的旧记录，再插入新记录。
                # 这样做保证了"单文档级幂等"：同一文档反复导入不会产生重复 chunk。
                cursor.execute(
                    f"DELETE FROM {PG_DOC_TABLE} WHERE doc_name = %s",
                    (doc_name,)
                )
                deleted = cursor.rowcount
                if deleted > 0:
                    logger.info(f"增量模式：已删除旧记录 {deleted} 条（{doc_name}）")
                conn.commit()

            # 使用 psycopg2 的 execute_values 做真正的批量插入，
            # 比 execute() 循环快 10-100 倍。
            insert_sql = f"""
                INSERT INTO {PG_DOC_TABLE}
                (chunk_text, chunk_vector, doc_name, doc_page, chunk_index)
                VALUES %s
            """
            from psycopg2.extras import execute_values
            execute_values(cursor, insert_sql, records)
            conn.commit()
            logger.info(f"数据库写入成功：{len(records)} 条记录")
            cursor.close()
    except Exception as e:
        logger.error(f"数据库写入失败: {e}", exc_info=True)
        raise


# ############################################################################
# 模块七：单文件处理主流程
# ############################################################################


def process_single_file(
    file_path: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    append: bool = False,
    hash_cache: dict = None,
) -> Tuple[int, str]:
    """
    处理单个文档并导入数据库。

    Returns:
        (成功导入的 chunk 数量, 文件 MD5 哈希)

    2026-04-17 修复：增加文件级 MD5 缓存检查。
    若 append 模式下文件内容未变化，直接跳过整篇文档的解析与 Embedding。
    """
    doc_name = Path(file_path).stem
    # 在最开头就计算整个文件的 MD5，用于判断"这个文件是否被修改过"
    file_hash = compute_file_hash(file_path)

    # --- 增量跳过逻辑 ---
    if append and hash_cache is not None:
        cached_hash = hash_cache.get(doc_name)
        if cached_hash == file_hash:
            # 文件内容和上次完全一致，直接跳过，省掉解析+Embedding 的巨额开销
            logger.info(f"文件未变化，跳过处理：{doc_name}")
            return 0, file_hash

    logger.info(f"开始处理文档：{file_path}")

    # 步骤 1：加载 + 清洗
    cleaned_docs = load_and_clean_documents(file_path)
    if not cleaned_docs:
        logger.warning(f"文档为空或加载失败：{file_path}")
        return 0, file_hash

    # 步骤 2：语义分块
    chunks = semantic_chunk_documents(cleaned_docs, chunk_size, chunk_overlap)
    if not chunks:
        logger.warning(f"分块后无有效内容：{file_path}")
        return 0, file_hash

    # 步骤 3：上下文增强 & 内容去重
    enriched_texts = []
    meta_infos = []
    seen_hashes = set()  # 同一文档内的 chunk 级去重

    for idx, chunk in enumerate(chunks):
        enriched = enrich_chunk(chunk)

        # 基于内容 MD5 去重：如果两个 chunk 的文本完全一样，只保留第一个
        h = hashlib.md5(enriched.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        enriched_texts.append(enriched)
        meta_infos.append({
            "doc_name": chunk.metadata.get("doc_name", ""),
            "doc_page": chunk.metadata.get("doc_page", 0),
            "chunk_index": idx,
        })

    if not enriched_texts:
        logger.warning(f"去重后无有效内容：{file_path}")
        return 0, file_hash

    logger.info(
        f"文档处理完成：{len(cleaned_docs)} 页 -> "
        f"{len(chunks)} 块 -> 去重后 {len(enriched_texts)} 块"
    )

    # 步骤 4：批量 Embedding
    vectors = compute_embeddings_batch(enriched_texts)

    # 步骤 5：组装数据库记录（过滤掉 embedding 失败的 None）
    db_records = []
    for text, vec, meta in zip(enriched_texts, vectors, meta_infos):
        if vec is None:
            continue  # embedding 失败，直接丢弃，不写入数据库
        # 将浮点数列表转换成 PostgreSQL vector 类型接受的字符串格式 "[0.1,0.2,...]"
        vec_str = "[" + ",".join([str(v) for v in vec]) + "]"
        db_records.append((
            text,
            vec_str,
            meta["doc_name"],
            meta["doc_page"],
            meta["chunk_index"]
        ))

    # 步骤 6：写入数据库
    insert_chunks_to_db(db_records, append=append, doc_name=doc_name)

    return len(db_records), file_hash


# ############################################################################
# 模块八：命令行入口
# ############################################################################


def main():
    """
    命令行入口。解析参数后遍历输入路径，逐个文件调用 process_single_file。
    """
    parser = argparse.ArgumentParser(description="文档数据导入脚本（生产级 RAG 预处理）")
    parser.add_argument("input", help="输入文件或目录路径")
    parser.add_argument(
        "--append", action="store_true",
        help="增量追加模式（同名文档会先删除旧记录再插入新记录，文件未变化则跳过）"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1024,
        help="分块大小（默认 1024 字符，约对应 500 中文字）"
    )
    parser.add_argument(
        "--overlap", type=int, default=128,
        help="分块重叠大小（默认 128 字符）"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入路径不存在：{args.input}")
        sys.exit(1)

    # 收集待处理文件：支持单个文件或整个目录
    if input_path.is_file():
        files = [input_path]
    else:
        # 按后缀名匹配，排序以保证处理顺序稳定（方便日志排查）
        files = sorted(input_path.glob("*.txt"))
        files += sorted(input_path.glob("*.pdf"))
        files += sorted(input_path.glob("*.docx"))

    if not files:
        logger.warning("未找到任何支持的文档文件（.txt / .pdf / .docx）")
        sys.exit(0)

    # 仅在增量模式下加载文件级 MD5 缓存
    hash_cache = load_hash_cache() if args.append else {}

    logger.info(f"共发现 {len(files)} 个待处理文档")

    total_chunks = 0
    file_iter = files
    if tqdm is not None:
        file_iter = tqdm(files, desc="文档处理进度")

    for file_path in file_iter:
        try:
            count, file_hash = process_single_file(
                str(file_path),
                chunk_size=args.chunk_size,
                chunk_overlap=args.overlap,
                append=args.append,
                hash_cache=hash_cache,
            )
            total_chunks += count

            # 更新缓存：无论 count 是否为 0，都记录本次 hash
            # count=0 可能表示"文件未变化跳过"或"文档为空"，都需要记录
            if args.append:
                hash_cache[Path(file_path).stem] = file_hash
        except Exception as e:
            logger.error(f"处理失败 [{file_path}]：{e}", exc_info=True)
            continue

    # 所有文件处理完后，一次性保存缓存到磁盘
    if args.append:
        save_hash_cache(hash_cache)

    logger.info(f"全部完成，总计导入 {total_chunks} 个文档块")


# Python 的惯用写法：只有当本脚本被直接运行时（不是被 import）才执行 main()
if __name__ == "__main__":
    main()
