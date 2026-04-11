#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理脚本

将 PDF/TXT/DOCX 文档处理为向量库并持久化

用法:
    # 处理单个文件
    python scripts/process_documents.py data/raw/王洪文传.txt
    
    # 处理整个目录
    python scripts/process_documents.py data/raw/ --output-dir vectorstore/
    
    # 使用高级检索器（Multi-Query + Rerank）
    python scripts/process_documents.py data/raw/ --advanced
    
    # 指定向量库类型
    python scripts/process_documents.py data/raw/ --vectorstore-type faiss
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.document_processor import DocumentProcessor
from src.rag.advanced_retriever import get_advanced_retriever
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.model_config import EMBEDDING_CONFIG, LANGCHAIN_CONFIG


def process_single_file(
    file_path: str,
    vectorstore_type: str = "chroma",
    output_dir: str = None,
    use_advanced: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    处理单个文档文件
    
    Args:
        file_path: 文档路径
        vectorstore_type: 向量库类型 (chroma/faiss)
        output_dir: 输出目录
        use_advanced: 是否使用高级检索器
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
    """
    print(f"\n处理文件: {file_path}")
    
    # 1. 加载和分块文档
    processor = DocumentProcessor()
    chunks = processor.process_file(
        file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if not chunks:
        print(f"❌ 文件处理失败或为空: {file_path}")
        return None
    
    print(f"✅ 文档分块完成: {len(chunks)} 个块")
    
    # 2. 创建向量库
    if use_advanced:
        # 使用高级检索器（带持久化）
        retriever = get_advanced_retriever()
        retriever.create_vectorstore(chunks, vectorstore_type=vectorstore_type)
        print(f"✅ 向量库创建完成（使用高级检索器）")
        return retriever
    else:
        # 基础向量库创建
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_CONFIG["model_path"],
            model_kwargs={"device": EMBEDDING_CONFIG["device"]},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        if output_dir is None:
            output_dir = LANGCHAIN_CONFIG["vectorstore"]["persist_directory"]
        
        output_path = Path(output_dir) / vectorstore_type
        output_path.mkdir(parents=True, exist_ok=True)
        
        if vectorstore_type == "faiss":
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            vectorstore.save_local(str(output_path))
        else:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(output_path)
            )
            vectorstore.persist()
        
        print(f"✅ 向量库创建完成: {output_path}")
        return vectorstore


def process_directory(
    input_dir: str,
    vectorstore_type: str = "chroma",
    output_dir: str = None,
    use_advanced: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    批量处理目录中的所有文档
    
    Args:
        input_dir: 输入目录
        vectorstore_type: 向量库类型
        output_dir: 输出目录
        use_advanced: 是否使用高级检索器
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
    """
    input_path = Path(input_dir)
    
    # 支持的文件格式
    supported_extensions = {".txt", ".pdf", ".docx", ".md"}
    
    # 收集所有文档文件
    files = []
    for ext in supported_extensions:
        files.extend(input_path.glob(f"*{ext}"))
    
    if not files:
        print(f"❌ 目录中没有支持的文档文件: {input_dir}")
        print(f"支持的格式: {', '.join(supported_extensions)}")
        return
    
    print(f"\n找到 {len(files)} 个文档文件:")
    for f in files:
        print(f"  - {f.name}")
    
    # 批量处理
    all_chunks = []
    processor = DocumentProcessor()
    
    for file_path in files:
        print(f"\n处理: {file_path.name}")
        chunks = processor.process_file(
            str(file_path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        if chunks:
            all_chunks.extend(chunks)
            print(f"  ✅ {len(chunks)} 个块")
        else:
            print(f"  ❌ 处理失败")
    
    if not all_chunks:
        print("❌ 没有成功处理任何文档")
        return
    
    print(f"\n{'='*60}")
    print(f"总计: {len(all_chunks)} 个文档块")
    print(f"{'='*60}")
    
    # 创建统一向量库
    if use_advanced:
        retriever = get_advanced_retriever()
        retriever.create_vectorstore(all_chunks, vectorstore_type=vectorstore_type)
        print(f"✅ 统一向量库创建完成（使用高级检索器）")
        return retriever
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_CONFIG["model_path"],
            model_kwargs={"device": EMBEDDING_CONFIG["device"]},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        if output_dir is None:
            output_dir = LANGCHAIN_CONFIG["vectorstore"]["persist_directory"]
        
        output_path = Path(output_dir) / vectorstore_type
        output_path.mkdir(parents=True, exist_ok=True)
        
        if vectorstore_type == "faiss":
            vectorstore = FAISS.from_documents(
                documents=all_chunks,
                embedding=embeddings
            )
            vectorstore.save_local(str(output_path))
        else:
            vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory=str(output_path)
            )
            vectorstore.persist()
        
        print(f"✅ 统一向量库创建完成: {output_path}")
        return vectorstore


def main():
    parser = argparse.ArgumentParser(
        description="将文档处理为向量库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s data/raw/王洪文传.txt
  %(prog)s data/raw/ --advanced
  %(prog)s data/raw/ --vectorstore-type faiss --output-dir ./my_vectorstore
        """
    )
    
    parser.add_argument(
        "input",
        help="输入文件或目录路径"
    )
    
    parser.add_argument(
        "--vectorstore-type",
        choices=["chroma", "faiss"],
        default="chroma",
        help="向量库类型 (默认: chroma)"
    )
    
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录 (默认: ./vectorstore)"
    )
    
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="使用高级检索器 (Multi-Query + Rerank)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="文本分块大小 (默认: 1000)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="文本分块重叠 (默认: 200)"
    )
    
    args = parser.parse_args()
    
    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 路径不存在: {args.input}")
        sys.exit(1)
    
    print("="*60)
    print("文档处理脚本")
    print("="*60)
    print(f"输入路径: {args.input}")
    print(f"向量库类型: {args.vectorstore_type}")
    print(f"使用高级检索器: {args.advanced}")
    print(f"分块大小: {args.chunk_size}")
    print(f"分块重叠: {args.chunk_overlap}")
    print("="*60)
    
    # 根据输入类型选择处理方式
    if input_path.is_file():
        result = process_single_file(
            str(input_path),
            vectorstore_type=args.vectorstore_type,
            output_dir=args.output_dir,
            use_advanced=args.advanced,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    else:
        result = process_directory(
            str(input_path),
            vectorstore_type=args.vectorstore_type,
            output_dir=args.output_dir,
            use_advanced=args.advanced,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    
    if result:
        print("\n" + "="*60)
        print("✅ 处理完成！")
        print("="*60)
        print("\n使用方法:")
        print("  1. 启动服务时会自动加载向量库")
        print("  2. 或者在代码中使用:")
        if args.advanced:
            print("     from src.rag.advanced_retriever import get_advanced_retriever")
            print("     retriever = get_advanced_retriever()")
            print("     results = retriever.retrieve('你的查询')")
        else:
            print(f"     from langchain_community.vectorstores import {args.vectorstore_type.capitalize()}")
            print(f"     vs = {args.vectorstore_type.capitalize()}.load_local('{args.output_dir or './vectorstore'}')")
    else:
        print("\n❌ 处理失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
