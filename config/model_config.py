# -*- coding: utf-8 -*-
"""
模型配置模块

本文件配置所有 AI 模型的参数，包括：
1. Embedding 模型（BGE-M3）
2. 重排序模型（BCE-Reranker）
3. LLM 模型（本地 Qwen + API 双模）

支持通过环境变量动态切换本地/API 模式。
"""

import os

# =============================================================================
# Embedding 模型配置
# =============================================================================

EMBEDDING_CONFIG = {
    # 模型路径
    # 支持两种模式：
    # 1. 原始 BGE-M3: "/root/autodl-tmp/models/Xorbits/bge-m3"
    # 2. 微调后模型: "/root/autodl-tmp/models/bge-m3-finetuned-wang"
    "model_path": os.getenv(
        "EMBEDDING_MODEL_PATH", 
        "/root/autodl-tmp/models/bge-m3-finetuned-wang"
    ),
    
    # 计算设备: cuda 或 cpu
    "device": os.getenv("EMBEDDING_DEVICE", "cuda"),
    
    # 最大序列长度
    # BGE-M3 支持 8192 tokens，但 FAQ 通常较短，512 足够
    "max_length": int(os.getenv("EMBEDDING_MAX_LENGTH", "512")),
    
    # 批处理大小
    # 根据 GPU 显存调整，4090 可以设 32 或更大
    "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
    
    # 是否使用 FP16 半精度加速
    "use_fp16": os.getenv("EMBEDDING_USE_FP16", "true").lower() == "true",
    
    # 向量维度（BGE-M3 固定 1024）
    "vector_dim": 1024,
    
    # 是否归一化向量
    # 归一化后，点积 = 余弦相似度
    "normalize": True
}

# =============================================================================
# 重排序模型配置
# =============================================================================

RERANKER_CONFIG = {
    # 模型路径
    "model_path": os.getenv(
        "RERANKER_MODEL_PATH",
        "/root/autodl-tmp/models/maidalun/bce-reranker-base_v1"
    ),
    
    # 计算设备
    "device": os.getenv("RERANKER_DEVICE", "cuda"),
    
    # 返回 top_n 结果
    "top_n": int(os.getenv("RERANKER_TOP_N", "5")),
    
    # 是否使用 FP16
    "use_fp16": os.getenv("RERANKER_USE_FP16", "true").lower() == "true"
}

# 如果 BCE-Reranker 模型不存在，使用兼容模式（不报错）
RERANKER_CONFIG["enabled"] = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

# =============================================================================
# BM25 检索配置（新增）
# =============================================================================

BM25_CONFIG = {
    # 是否启用 BM25 混合检索
    "enabled": os.getenv("BM25_ENABLED", "true").lower() == "true",

    # BM25 权重（0-1）
    # 向量权重 = 1 - bm25_weight
    # 推荐配置：
    # - 通用场景：bm25_weight=0.3（向量权重 0.7 + BM25 权重 0.3）
    # - 关键词场景：bm25_weight=0.7（向量权重 0.3 + BM25 权重 0.7）
    "weight": float(os.getenv("BM25_WEIGHT", "0.3")),

    # BM25 返回数量（通常设大一些，后续重排序）
    "top_k": int(os.getenv("BM25_TOP_K", "20")),

    # 分数阈值（低于此值的结果过滤掉）
    "score_threshold": float(os.getenv("BM25_SCORE_THRESHOLD", "0.1")),

    # 中文分词器选择：jieba, pkuseg, thulac
    "tokenizer": os.getenv("BM25_TOKENIZER", "jieba"),
    
    # =============================================================================
    # RRF（Reciprocal Rank Fusion）配置 - 新增
    # =============================================================================
    
    # 混合检索融合方法："linear"（线性加权）或 "rrf"（倒数排名融合）
    # 推荐：rrf（工业界标准，不需要归一化，对异常值不敏感）
    "fusion_method": os.getenv("HYBRID_FUSION_METHOD", "rrf"),
    
    # RRF 参数 K（通常设为 60，论文推荐值）
    # 公式：score = 1/(K+rank_vector) + 1/(K+rank_bm25)
    # K 值越大，排名差异影响越小；K 值越小，排名差异影响越大
    "rrf_k": int(os.getenv("RRF_K", "60")),
}

# =============================================================================
# LLM 配置（双模架构）
# =============================================================================

LLM_CONFIG = {
    # 默认模式: "local" 或 "api"
    # 可通过环境变量 LLM_MODE 动态切换
    "default_mode": os.getenv("LLM_MODE", "local"),
    
    # -------------------------------------------------------------------------
    # 本地模型配置（Qwen/Llama 等）
    # -------------------------------------------------------------------------
    "local": {
        # 模型路径
        "model_path": os.getenv(
            "LOCAL_LLM_PATH",
            "/root/autodl-tmp/models/qwen/Qwen1.5-7B-Chat"
        ),
        
        # 计算设备
        "device": os.getenv("LOCAL_LLM_DEVICE", "cuda"),
        
        # 设备映射方式
        # "auto": 自动分配到多卡
        # "cuda:0": 强制单卡
        "device_map": os.getenv("LOCAL_LLM_DEVICE_MAP", "auto"),
        
        # 数据类型
        # torch.float16: 半精度，省显存
        # torch.bfloat16: 更好的数值稳定性
        # torch.float32: 全精度，质量最好但慢
        "torch_dtype": "float16",
        
        # 生成参数
        "max_new_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "512")),
        "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.7")),
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        
        # 系统提示词（历史人物专家）
        "system_prompt": """你是一位专业的中国现代史研究专家，尤其擅长"四人帮"相关历史。
        你需要基于提供的参考资料回答用户关于历史人物的问题。
        要求：
        1. 回答必须基于提供的资料，不要编造
        2. 保持客观中立的历史态度
        3. 如果资料不足以回答，请明确说明"根据现有资料无法确定"
        4. 适当引用资料来源"""
    },
    
    # -------------------------------------------------------------------------
    # API 模型配置（OpenAI/DeepSeek/Claude 等）
    # -------------------------------------------------------------------------
    "api": {
        # 服务提供商
        # 支持: openai, deepseek, claude, zhipu, moonshot 等
        "provider": os.getenv("API_PROVIDER", "deepseek"),
        
        # API 密钥（从环境变量读取，不要硬编码）
        "api_key": os.getenv("API_KEY", ""),
        
        # API 基础 URL
        "base_url": os.getenv("API_BASE_URL", "https://api.deepseek.com/v1"),
        
        # 模型名称
        "model": os.getenv("API_MODEL", "deepseek-chat"),
        
        # 生成参数
        "max_tokens": int(os.getenv("API_MAX_TOKENS", "512")),
        "temperature": float(os.getenv("API_TEMPERATURE", "0.7")),
        
        # 超时设置（秒）
        "timeout": 60,
        
        # 重试次数
        "retry": 3,
        
        # 系统提示词
        "system_prompt": """你是一位专业的中国现代史研究专家，尤其擅长"四人帮"相关历史。
你需要基于提供的参考资料回答用户关于历史人物的问题。
要求：
1. 回答必须基于提供的资料，不要编造
2. 保持客观中立的历史态度
3. 如果资料不足以回答，请明确说明
4. 适当引用资料来源"""
    }
}

# =============================================================================
# 厂商特定的 API 配置
# =============================================================================

API_PROVIDER_CONFIG = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-3.5-turbo"
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
    },
    "claude": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-3-sonnet-20240229"
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4"
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k"
    }
}

# =============================================================================
# LangChain 配置
# =============================================================================

LANGCHAIN_CONFIG = {
    # 默认链类型: rag, conversational, conversational_retrieval
    "default_chain_type": os.getenv("LANGCHAIN_CHAIN_TYPE", "rag"),
    
    # 向量存储配置
    "vectorstore": {
        # 类型: chroma, faiss
        "type": os.getenv("VECTORSTORE_TYPE", "chroma"),
        # 持久化路径
        "persist_directory": os.getenv("VECTORSTORE_PERSIST_DIR", "./vectorstore"),
        # 搜索类型: similarity, mmr
        "search_type": os.getenv("VECTORSTORE_SEARCH_TYPE", "similarity"),
        # 搜索结果数量
        "k": int(os.getenv("VECTORSTORE_K", "3"))
    },
    
    # 文档处理配置
    "document": {
        # 分块大小
        "chunk_size": int(os.getenv("DOCUMENT_CHUNK_SIZE", "1000")),
        # 分块重叠
        "chunk_overlap": int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "200")),
        # 分块器类型: recursive, character
        "splitter_type": os.getenv("DOCUMENT_SPLITTER_TYPE", "recursive")
    },
    
    # 记忆配置
    "memory": {
        # 类型: buffer, summary
        "type": os.getenv("MEMORY_TYPE", "buffer"),
        # 是否启用记忆
        "enabled": os.getenv("MEMORY_ENABLED", "true").lower() == "true"
    },
    
    # Agent 配置
    "agent": {
        # 类型: zero-shot-react-description, react-docstore
        "type": os.getenv("AGENT_TYPE", "zero-shot-react-description"),
        # 是否启用详细输出
        "verbose": os.getenv("AGENT_VERBOSE", "true").lower() == "true"
    }
}