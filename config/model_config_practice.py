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

    # 本地模型配置（Qwen/Llama 等）
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
    
    # API 模型配置（OpenAI/DeepSeek/Claude 等）
    "api":{
        # API 类型（openai/deepseek/claude）
        "api_type": os.getenv("API_LLM_TYPE", "openai"),
        # OpenAI API 配置
        "openai": {
            "model": os.getenv("OPENAI_MODEL", "gpt-4-0613"),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "512")),
            "top_p": 0.9,
            "top_k": 50,
        },
        # DeepSeek API 配置
        "deepseek": {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-3.5"),
            "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "512")),
            "top_p": 0.9,
            "top_k": 50,
        },
        # Claude API 配置
        "claude": {
            "model": os.getenv("CLAUDE_MODEL", "claude-2"),
            "temperature": float(os.getenv("CLAUDE_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("CLAUDE_MAX_TOKENS", "512")),
            "top_p": 0.9,
            "top_k": 50,
    },
      
}
}