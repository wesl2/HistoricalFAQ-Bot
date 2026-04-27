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

EMBEDDING_CONFIG = {
        # 模型路径
    # 支持两种模式：
    # 1. 原始 BGE-M3: "/root/autodl-tmp/models/Xorbits/bge-m3"
    # 2. 微调后模型: "/root/autodl-tmp/models/bge-m3-finetuned-wang"
    "model_path":os.getenv(
        "EMBEDDING_MODEL_PATH",
        "/root/autodl-tmp/models/bge-m3-finetuned-wang"),
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
    "vector_dim":1024,
    # 是否归一化向量
    # 归一化后，点积 = 余弦相似度
    "normalize": True
}

RERANKER_CONFIG = {
    # 模型路径
    "model_path": os.getenv(
        "RERANKER_MODEL_PATH",
        "/root/autodl-tmp/models/maidalun/bce-reranker-base_v1"
    ),
    # 计算设备
    "device": os.getenv("RERANKER_DEVICE", "cuda"),
    # 最大序列长度
    "max_length" : int(os.getenv("RERANKER_MAX_LENGTH","512")),
    # 批处理大小
    "batch_size": int(os.getenv("RERANKER_BATCH_SIZE","8")),
    # 是否使用 FP16 半精度加速
    "use_fp16": os.getenv("RERANKER_USE_FP16","True").lower() == "true",
    
}

# 如果 BCE-Reranker 模型不存在，使用兼容模式（不报错）
RERANKER_CONFIG["enabled"] = os.getenv("RERANKER_ENABLED", "True").lower() == "true"


# BM25 检索配置
BM25_CONFIG = {
        # 是否启用 BM25 混合检索
    "enabled": os.getenv("BM25_ENABLED","True").lower() == "true",
    # BM25 权重（向量权重 = 1 - BM25 权重
    "weight": float(os.getenv("BM25_WEIGHT","0.3")),
    "top_k":int(os.getenv("BM25_TOP_K","100")),
    "score_threshold":float(os.getenv("BM25_SCORE_THRESHOLD","0.2")),
    "tokenizer" : os.getenv("BM25_TOKENIZER","jieba"),
    # RRF（Reciprocal Rank Fusion）配置 - 新增
    # RRF 参数 K（通常设为 60，论文推荐值）
    # 公式：score = 1/(K+rank_vector) + 1/(K+rank_bm25)
    # K 值越大，排名差异影响越小；K 值越小，排名差异影响越大
    "fusion_method": os.getenv("HYBRID_FUSION_METHOD", "rrf"),
    "rrf_k": int(os.getenv("BM25_RRF_K", "60"))
}


LLM_CONFIG = {
    "default_mode": os.getenv("LLM_MODE", "local"),

    # -------------------------------------------------------------------------
    # 本地模型配置（vLLM 服务）
    # -------------------------------------------------------------------------
    # 注意：改用 vLLM 后，模型由独立的推理服务托管，不再在本进程加载权重。
    # 这里的配置本质是"连接本地 vLLM 服务的客户端参数"，跟 API 模式结构一致。
    "local": {
        # vLLM 服务地址（OpenAI-compatible API）
        "base_url": os.getenv(
            "LOCAL_LLM_BASE_URL",
            "http://localhost:8000/v1"
        ),

        # API Key（vLLM 默认不校验，随便填）
        "api_key": os.getenv("LOCAL_LLM_API_KEY", "not-needed"),

        # 模型名称（vLLM 启动时 --model 或 --served-model-name 指定的名称）
        "model": os.getenv("LOCAL_LLM_MODEL_NAME", "qwen-7b-chat"),

        # 生成参数（OpenAI API 标准字段名）
        "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "512")),
        "top_p": 0.9,

        # 连接控制
        "timeout": int(os.getenv("LOCAL_LLM_TIMEOUT", "120")),
        "max_retries": int(os.getenv("LOCAL_LLM_MAX_RETRIES", "2")),


    },

    # -------------------------------------------------------------------------
    # API 模型配置（OpenAI/DeepSeek/Claude 等）
    # -------------------------------------------------------------------------
    # 结构跟 local 保持一致，方便统一用 ChatOpenAI 初始化。
    # 如果没填 base_url，standard_llm_new.py 会根据 provider 自动补全。
    "api": {
        # 服务商标识（用于自动补全 base_url 和 model，可选）
        "provider": os.getenv("API_PROVIDER", "deepseek"),

        # API 密钥
        "api_key": os.getenv("API_KEY", ""),

        # 自定义 base_url（如果填了就直接用，不读 provider 默认值）
        "base_url": os.getenv("API_BASE_URL", ""),

        # 自定义模型名（如果填了就直接用，不读 provider 默认值）
        "model": os.getenv("API_MODEL", ""),

        # 生成参数
        "temperature": float(os.getenv("API_TEMPERATURE", "0.0")),
        "max_tokens": int(os.getenv("API_MAX_TOKENS", "512")),
        "top_p": 0.9,

        # 连接控制
        "timeout": int(os.getenv("API_LLM_TIMEOUT", "60")),
        "max_retries": int(os.getenv("API_LLM_MAX_RETRIES", "2")),


    }
}


# =============================================================================
# 厂商特定的 API 默认配置（base_url + model）
# =============================================================================
# standard_llm_new.py 里，如果 api 配置没填 base_url/model，就从这里补。
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