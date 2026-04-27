# -*- coding: utf-8 -*-
"""
本地 Embedding 模型封装 - 练习版本
目标：自己手敲实现 BGE-M3 文本向量化
参考：embedding_local.py
"""
# 需要导入：
# - logging: 日志记录
# - torch: PyTorch 深度学习框架
# - torch.nn.functional as F: 用于向量归一化
# - Union, List from typing: 类型提示
# - AutoTokenizer, AutoModel from transformers: 加载模型
# - EMBEDDING_CONFIG from config.model_config: 配置

# 你的代码：
import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent.parent  # 根据文件层级调整
sys.path.insert(0, str(project_root))


# 添加项目根目录到 Python 路径
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import logging
import torch
import torch.nn.functional as F
from typing import Union, List
from transformers import AutoTokenizer, AutoModel
from config.model_config_practice import EMBEDDING_CONFIG


# 使用 logging.getLogger(__name__) 创建 logger
# 你的代码：
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# 定义三个全局变量，初始值为 None：
# - _tokenizer: 分词器
# - _model: 神经网络模型
# - _device: 计算设备（cpu/cuda）

# 你的代码：
_tokenizer = None
_model = None
_device = None


# 功能：延迟加载模型（只在第一次调用时执行）
# 要点：
# 1. 使用 global 声明修改全局变量
# 2. 如果 _model 为 None 才加载，否则直接返回
# 3. 设置设备：torch.device(EMBEDDING_CONFIG['device'])
# 4. 加载 tokenizer: AutoTokenizer.from_pretrained(...)
# 5. 加载 model: AutoModel.from_pretrained(...).to(_device)
# 6. 设置为评估模式：model.eval()
# 7. 如果是 CUDA 且配置允许，使用半精度：model.half()
# 8. 使用 logger.info() 记录加载过程

def _load_model():
    """延迟加载模型"""
    # 你的代码：
    # 1. global 声明
    global _tokenizer, _model, _device
    # 2. if _model is None 判断
    if _model is None:
        logger.info(f"正在加载 Embedding 模型: {EMBEDDING_CONFIG['model_path']}")
    # 3. 设置 _device
        _device = torch.device(EMBEDDING_CONFIG['device'])
    # 4. 加载 tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_CONFIG['model_path']
        )
    # 5. 加载 model
        _model = AutoModel.from_pretrained(
            EMBEDDING_CONFIG['model_path']
        ).to(_device)
    # 6. model.eval()\
        _model.eval()
    # 7. 可选：model.half()
    if  EMBEDDING_CONFIG['use_fp16'] == "true" and _device.type == 'cuda':
        _model = _model.half()
    # 8. 记录完成
        logger.info("Embedding 模型加载完成")

# 功能：计算文本的 embedding 向量
# 输入：text (str 或 List[str])
# 输出：List[float] (单条) 或 List[List[float]] (批量)，出错返回 None
# 
# 步骤：
# 1. 调用 _load_model() 确保模型已加载
# 2. 使用 try-except 捕获异常
# 3. tokenizer 编码：
#    - padding=True
#    - truncation=True
#    - max_length=EMBEDDING_CONFIG['max_length']
#    - return_tensors='pt'
#    - .to(_device)
# 4. torch.no_grad() 上下文：
#    - model(**encoded_input) 前向传播
#    - 取 [CLS] token: model_output[0][:, 0]
#    - L2 归一化: F.normalize(..., p=2, dim=1)
# 5. 转换为 numpy，再转 list
# 6. 如果是单条 str，返回 [0]，否则返回整个列表
# 7. 异常时 logger.error() 记录，返回 None

def compute_embedding(text):
    """计算文本的嵌入向量"""
    # 你的代码：
    # 1. _load_model()
    _load_model()
    # 2. try:
    # 3.   tokenizer 编码
    # 4.   with torch.no_grad(): 计算向量
    # 5.   取 [CLS] token
    # 6.   L2 归一化
    # 7.   转 numpy -> list
    # 8.   根据输入类型返回
    # 9. except Exception as e:
    # 10.  logger.error()
    # 11.  return None
    try:
        inputs = _tokenizer(text,return_tensors='pt',padding=True,truncation=True,max_length=EMBEDDING_CONFIG['max_length']).to(_device)
        with torch.no_grad():
            outputs = _model(**inputs)
            sentence_embeddings = F.normalize(outputs[0][:,0,:], p=2, dim=1) # p=2 表示 L2 归一化，dim=1 表示对每行进行归一化
            #注 ： 1. isinstance(text, str) 判断输入是单条文本还是批量文本
            # 2. 输入是单条文本则希望得到的是单独一个[x,x,x,...]这样的向量
            # 3. 如果不取[0] 得到的就是[[x,x,x,...]]这样的列表，外面多了一层列表嵌套，不符合预期
            if isinstance(text,str):
                return sentence_embeddings.cpu().numpy()[0].tolist()
            else:
                return sentence_embeddings.cpu().numpy().tolist()
    except Exception as e:
        logger.error(f"编码失败: {e}")
        return None



# 功能：调用 compute_embedding，出错时返回零向量兜底
# 输入：text (str 或 List[str])
# 输出：List[float] 或 List[List[float]]（不会返回 None）
#
# 步骤：
# 1. 调用 compute_embedding(text) 获取 result
# 2. 如果 result 为 None:
#    - 获取维度: EMBEDDING_CONFIG['vector_dim']
#    - 如果是单条 str，返回 [0.0] * dim
#    - 如果是列表，返回 [[0.0] * dim]
# 3. 否则返回 result

def get_embedding(text):
    """获取文本向量（兼容接口，不会返回 None）"""
    # 你的代码：
    # 1. result = compute_embedding(text)
    # 2. if result is None: 返回零向量
    # 3. return result
    result = compute_embedding(text)
    if result is None:
        dim = EMBEDDING_CONFIG['vector_dim']
        if isinstance(text, str):
            return [0.0] * dim
        else:
            return [[0.0] * dim]
    return result



# 功能：验证实现是否正确
# 测试项：
# 1. 单条文本向量化，检查维度是否为 1024
# 2. 批量文本向量化，检查返回数量
# 3. 计算相似度，验证归一化是否正确（模长应为1.0）

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # TODO 7.1: 测试单条文本
    print("=" * 50)
    print("测试1：单条文本")
    print("=" * 50)
    vec = get_embedding("王洪文和毛泽东")
    print(f"向量维度：{len(vec)}")
    print(f"前10个值：{vec[:10]}")
    
    # TODO 7.2: 测试批量文本
    print("\n" + "=" * 50) 
    print("测试2：批量文本")
    print("=" * 50)
    texts = ["王洪文万岁", "王洪文是毛泽东的追随者"]
    vecs = get_embedding(texts)
    print(f"批量向量数量：{len(vecs)}")
    
    # TODO 7.3: 测试相似度计算
    print("\n" + "=" * 50)
    print("测试3：相似度计算")  
    print("=" * 50)
    import numpy as np
    v1 = np.array(vecs[0])
    v2 = np.array(vecs[1])
    similarity = np.dot(v1, v2)
    print(f"相似度：{similarity:.4f}")


# =============================================================================
# 练习检查清单
# =============================================================================
# □ 能正确导入所有依赖
# □ _load_model() 只执行一次（单例模式）
# □ compute_embedding() 能处理 str 和 List[str]
# □ 向量维度正确（1024维）
# □ 向量已归一化（模长为1.0）
# □ 出错时返回 None（compute_embedding）
# □ 出错时返回零向量（get_embedding）
# □ 能正确计算文本相似度
# =============================================================================
