# -*- coding: utf-8 -*-
"""
EPUB → Markdown 解析器

将 EPUB 格式的电子书解析为结构化的 Markdown 文本，保留章节层级。

特性：
1. 基于标准库 zipfile + BeautifulSoup，无需额外依赖
2. 保留章节结构（h1→#, h2→##, h3→###）
3. 自动去噪（去掉图片、脚本、样式、页眉页脚）
4. 扫描版检测（图片过多+文本过少 → 标记为不可用）
5. 质量报告（字符数、章节数、中文占比）

用法：
    parser = EpubParser()
    markdown, quality = parser.parse("book.epub")
    if quality.is_valid:
        print(markdown)
"""

import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString


@dataclass
class EpubQuality:
    """EPUB 质量报告"""
    total_chars: int = 0
    chinese_chars: int = 0
    chapter_count: int = 0
    html_files: int = 0
    jpg_files: int = 0
    is_valid: bool = False
    reason: str = ""

    @property
    def chinese_ratio(self) -> float:
        return self.chinese_chars / self.total_chars if self.total_chars > 0 else 0


class EpubParser:
    """EPUB 解析器：EPUB → Markdown"""

    # HTML 标签 → Markdown 映射
    HEADING_MAP = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### "}

    # 需要完全移除的标签（不保留内容）
    REMOVE_TAGS = {"script", "style", "img", "svg", "nav", "footer", "header",
                   "aside", "form", "input", "button", "iframe", "canvas",
                   "video", "audio", "source", "track", "embed", "object",
                   "noscript", "template"}

    def parse(self, file_path: str) -> Tuple[str, EpubQuality]:
        """
        解析 EPUB 文件，返回 Markdown 文本和质量报告。

        Args:
            file_path: EPUB 文件路径

        Returns:
            (markdown_text, quality_report)
        """
        path = Path(file_path)
        if not path.exists():
            return "", EpubQuality(is_valid=False, reason="文件不存在")

        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                return self._parse_zip(zf)
        except zipfile.BadZipFile:
            return "", EpubQuality(is_valid=False, reason="不是有效的 ZIP/EPUB 文件")
        except Exception as e:
            return "", EpubQuality(is_valid=False, reason=f"解析失败: {e}")

    def _parse_zip(self, zf: zipfile.ZipFile) -> Tuple[str, EpubQuality]:
        """解析 ZIP 内部结构"""
        files = zf.namelist()
        quality = EpubQuality()

        # 统计文件类型
        quality.html_files = sum(1 for f in files if f.endswith(('.html', '.xhtml', '.htm')))
        quality.jpg_files = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))

        # 1. 找到 content.opf 路径
        container_xml = zf.read("META-INF/container.xml").decode('utf-8', errors='ignore')
        rootfile_path = self._extract_rootfile_path(container_xml)
        if not rootfile_path:
            return "", EpubQuality(is_valid=False, reason="无法找到 content.opf")

        # 2. 解析 content.opf，获取 spine（阅读顺序）
        opf_content = zf.read(rootfile_path).decode('utf-8', errors='ignore')
        spine_ids, manifest = self._parse_opf(opf_content)

        # 3. 计算 content.opf 所在目录，用于拼接相对路径
        opf_dir = Path(rootfile_path).parent.as_posix()
        if opf_dir == ".":
            opf_dir = ""

        # 4. 按 spine 顺序读取每个章节
        markdown_parts = []
        for item_id in spine_ids:
            href = manifest.get(item_id)
            if not href:
                continue

            # 拼接完整路径
            full_path = f"{opf_dir}/{href}" if opf_dir else href
            full_path = full_path.lstrip('/')

            if full_path not in files:
                continue

            try:
                html_content = zf.read(full_path).decode('utf-8', errors='ignore')
                md_part = self._html_to_markdown(html_content)
                if md_part.strip():
                    markdown_parts.append(md_part)
            except Exception:
                continue

        # 5. 合并并后处理
        full_md = "\n\n".join(markdown_parts)
        full_md = self._post_process(full_md)

        # 6. 质量统计
        quality.total_chars = len(full_md)
        quality.chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', full_md))
        quality.chapter_count = len(re.findall(r'^#{1,4}\s+', full_md, re.MULTILINE))

        # 7. 质量校验
        is_valid, reason = self._validate(quality, full_md)
        quality.is_valid = is_valid
        quality.reason = reason

        return full_md, quality

    @staticmethod
    def _extract_rootfile_path(container_xml: str) -> Optional[str]:
        """从 container.xml 提取 content.opf 路径"""
        try:
            root = ET.fromstring(container_xml)
            # namespace 处理
            ns = {'ns': 'urn:oasis:names:tc:opendocument:xmlns:container'}
            rootfile = root.find('.//ns:rootfile', ns)
            if rootfile is not None:
                return rootfile.get('full-path')
        except ET.ParseError:
            # 手动正则提取（容错）
            match = re.search(r'full-path=["\']([^"\']+)["\']', container_xml)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _parse_opf(opf_content: str) -> Tuple[list, dict]:
        """
        解析 content.opf，返回 spine 顺序和 manifest 映射。

        Returns:
            (spine_item_ids, manifest_dict)
            manifest_dict: {id -> href}
        """
        try:
            root = ET.fromstring(opf_content)
        except ET.ParseError:
            return [], {}

        # 处理 namespace
        ns = {'opf': 'http://www.idpf.org/2007/opf'}

        # 解析 manifest: id -> href
        manifest = {}
        manifest_elem = root.find('opf:manifest', ns)
        if manifest_elem is not None:
            for item in manifest_elem.findall('opf:item', ns):
                item_id = item.get('id')
                href = item.get('href')
                if item_id and href:
                    manifest[item_id] = href

        # 解析 spine: 读取顺序
        spine_ids = []
        spine_elem = root.find('opf:spine', ns)
        if spine_elem is not None:
            for itemref in spine_elem.findall('opf:itemref', ns):
                item_id = itemref.get('idref')
                if item_id:
                    spine_ids.append(item_id)

        return spine_ids, manifest

    def _html_to_markdown(self, html: str) -> str:
        """将 HTML 内容转为 Markdown"""
        soup = BeautifulSoup(html, 'html.parser')

        # 1. 移除噪声标签（不保留内容）
        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 2. 移除注释
        for comment in soup.find_all(string=lambda text: isinstance(text, NavigableString) and str(text).strip().startswith('<!--')):
            comment.extract()

        # 3. 从 body 或 html 开始转换
        root = soup.body or soup.html or soup
        if not root:
            return ""

        return self._convert_element(root).strip()

    def _convert_element(self, element) -> str:
        """递归转换 BeautifulSoup 元素为 Markdown"""
        if isinstance(element, NavigableString):
            text = str(element)
            # 规范化空白
            return text

        parts = []
        tag_name = element.name

        # 处理子元素
        for child in element.children:
            parts.append(self._convert_element(child))

        content = ''.join(parts)

        # 根据标签类型包装
        if tag_name in self.HEADING_MAP:
            prefix = self.HEADING_MAP[tag_name]
            # 清理标题内的换行和多余空格
            text = ' '.join(content.split())
            return f"\n{prefix}{text}\n"

        elif tag_name == 'p':
            text = ' '.join(content.split())
            if not text:
                return ""
            # 识别"第X章""第X篇""第X卷"等章节标题，转换为 H1
            if re.match(r'^第[一二三四五六七八九十百千零]+(?:章|篇|卷|部|编)$', text.strip()):
                return f"\n# {text}\n\n"
            return f"{text}\n\n"

        elif tag_name == 'br':
            return "\n"

        elif tag_name in ('strong', 'b'):
            text = content.strip()
            return f"**{text}**" if text else ""

        elif tag_name in ('em', 'i'):
            text = content.strip()
            return f"*{text}*" if text else ""

        elif tag_name == 'li':
            text = ' '.join(content.split())
            return f"- {text}\n"

        elif tag_name in ('ul', 'ol'):
            return f"\n{content}\n"

        elif tag_name == 'blockquote':
            lines = content.strip().split('\n')
            quoted = '\n'.join(f"> {line}" for line in lines if line.strip())
            return f"\n{quoted}\n\n"

        elif tag_name in ('div', 'section', 'article', 'span', 'a'):
            return content

        elif tag_name in ('table', 'tr', 'td', 'th'):
            # 表格简化处理：只保留文本
            return content

        else:
            return content

    @staticmethod
    def _post_process(text: str) -> str:
        """Markdown 后处理：清理格式"""
        # 1. 合并多余空行（3个以上 → 2个）
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        # 2. 去掉行首行尾空格
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)

        # 3. 合并标题后的多余空行
        text = re.sub(r'^(#{1,4}\s+.+?)\n{3,}', r'\1\n\n', text, flags=re.MULTILINE)

        # 4. 去掉孤立的数字行（页码）
        text = re.sub(r'\n\d{1,3}\n', '\n', text)

        # 5. 去掉目录页常见的噪声
        noise_patterns = [
            r'^(封面|版权|目录|Contents|Index|索引|参考文献|附录)\s*$',
            r'^第[一二三四五六七八九十百零\d]+[章节节]\s*$',  # 单独的"第一章"保留，但需要上下文
        ]
        # 这些不在这里处理，交给上层 TextCleaner

        return text.strip()

    @staticmethod
    def _validate(quality: EpubQuality, text: str) -> Tuple[bool, str]:
        """质量校验"""
        # 扫描版检测
        if quality.jpg_files > 50 and quality.total_chars < 5000:
            return False, f"扫描版EPUB（{quality.jpg_files}张图片，仅{quality.total_chars}字符），需要OCR处理"

        # 文本量检测
        if quality.total_chars < 3000:
            return False, f"文本量过少（{quality.total_chars}字符），可能是目录/前言"

        # 中文占比
        if quality.total_chars > 0:
            ratio = quality.chinese_chars / quality.total_chars
            if ratio < 0.15:
                return False, f"中文占比过低（{ratio:.1%}），可能是外文书籍或乱码"

        return True, "质量合格"


# =============================================================================
# 便捷函数
# =============================================================================

def parse_epub_to_markdown(file_path: str) -> Tuple[str, EpubQuality]:
    """
    便捷函数：解析 EPUB 为 Markdown。

    Args:
        file_path: EPUB 文件路径

    Returns:
        (markdown_text, quality_report)
    """
    parser = EpubParser()
    return parser.parse(file_path)
