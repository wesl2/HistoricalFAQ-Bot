# -*- coding: utf-8 -*-
"""
文档处理器

使用 LangChain 处理各种格式的文档，包括：
1. 加载文档（PDF、TXT、DOCX等）
2. 文本分块
3. 文档转换
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, DocxLoader, UnstructuredFileLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, CharacterTextSplitter
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    文档处理器类
    """
    
    def __init__(self):
        """
        初始化文档处理器
        """
        pass
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        加载文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档列表
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext == '.docx':
                loader = DocxLoader(file_path)
            else:
                # 尝试使用通用加载器
                loader = UnstructuredFileLoader(file_path)
            
            documents = loader.load()
            logger.info(f"成功加载文档: {file_path}, 共 {len(documents)} 页")
            return documents
        except Exception as e:
            logger.error(f"加载文档失败: {file_path}, 错误: {e}")
            return []
    
    def split_documents(self, 
                      documents: List[Document], 
                      chunk_size: int = 1000, 
                      chunk_overlap: int = 200,
                      splitter_type: str = "recursive") -> List[Document]:
        """
        文本分块
        
        Args:
            documents: 文档列表
            chunk_size: 块大小
            chunk_overlap: 块重叠
            splitter_type: 分块器类型 (recursive 或 character)
            
        Returns:
            分块后的文档列表
        """
        if splitter_type == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True
            )
        else:
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
        
        chunks = splitter.split_documents(documents)
        logger.info(f"文档分块完成: 共 {len(chunks)} 个块")
        return chunks
    
    def process_file(self, 
                    file_path: str, 
                    chunk_size: int = 1000, 
                    chunk_overlap: int = 200) -> List[Document]:
        """
        处理文件（加载 + 分块）
        
        Args:
            file_path: 文件路径
            chunk_size: 块大小
            chunk_overlap: 块重叠
            
        Returns:
            处理后的文档块列表
        """
        documents = self.load_document(file_path)
        if not documents:
            return []
        
        return self.split_documents(documents, chunk_size, chunk_overlap)
    
    def batch_process(self, 
                     file_paths: List[str], 
                     chunk_size: int = 1000, 
                     chunk_overlap: int = 200) -> List[Document]:
        """
        批量处理文件
        
        Args:
            file_paths: 文件路径列表
            chunk_size: 块大小
            chunk_overlap: 块重叠
            
        Returns:
            处理后的文档块列表
        """
        all_chunks = []
        
        for file_path in file_paths:
            chunks = self.process_file(file_path, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
        
        logger.info(f"批量处理完成: 共 {len(all_chunks)} 个块")
        return all_chunks


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python document_processor.py <file_path>")
        sys.exit(1)
    
    processor = DocumentProcessor()
    chunks = processor.process_file(sys.argv[1])
    print(f"处理完成，生成 {len(chunks)} 个文档块")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n块 {i+1}:")
        print(chunk.page_content[:200] + "...")
