# Historical FAQ Bot

基于RAG架构的历史人物问答系统，结合了王洪文QA微调项目和FAQ项目。

## 项目功能

- **FAQ检索**：快速匹配常见问题
- **文档检索(RAG)**：基于文档内容回答问题
- **混合检索**：结合FAQ和文档结果
- **本地/API LLM支持**：支持本地模型和API模型
- **FastAPI后端**：提供RESTful API接口
- **Web前端**：用户友好的交互界面
- **容器化部署**：支持Docker部署

## 技术栈

- **后端**：FastAPI
- **前端**：HTML5 + Tailwind CSS + JavaScript
- **向量数据库**：PostgreSQL + pgvector
- **嵌入模型**：BGE-M3（支持微调）
- **重排序模型**：BCE-Reranker
- **LLM**：本地Qwen + API双模架构
- **部署**：Docker + docker-compose

## 项目结构

```
HistoricalFAQ-Bot/
├── src/             # 核心代码
│   ├── api/         # FastAPI后端
│   ├── chat/        # 对话引擎
│   ├── data_pipeline/ # 数据处理管道
│   ├── embedding/   # 嵌入模型
│   ├── llm/         # 语言模型
│   └── retrieval/   # 检索系统
├── config/          # 配置文件
├── data/            # 数据目录
│   ├── finetune/    # 微调数据
│   ├── processed/   # 处理后数据
│   ├── qa_pairs/    # QA对
│   └── raw/         # 原始数据
├── scripts/         # 脚本
├── models/          # 模型目录
├── logs/            # 日志目录
├── frontend.html    # 前端界面
├── Dockerfile       # Docker构建文件
├── docker-compose.yml # Docker Compose配置
├── requirements.txt # 依赖文件
└── README.md        # 项目文档
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd HistoricalFAQ-Bot

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

修改 `config/` 目录下的配置文件，设置模型路径和数据库连接信息。

### 3. 启动服务

#### 方法一：直接运行

```bash
# 启动FastAPI服务
python -m src.api.main
```

#### 方法二：Docker部署

```bash
# 构建并启动容器
docker-compose up -d
```

### 4. 访问系统

- **API接口**：http://localhost:8000/docs
- **前端界面**：打开 `frontend.html` 文件

## API接口

### 1. 查询接口

- **URL**：`/api/query`
- **方法**：POST
- **请求体**：
  ```json
  {
    "question": "王洪文的生平",
    "history": []
  }
  ```
- **响应**：
  ```json
  {
    "answer": "王洪文（1935年-1992年），吉林长春人，曾任中共中央副主席...",
    "sources": [
      {
        "type": "faq",
        "question": "王洪文是谁？",
        "confidence": 0.95
      }
    ],
    "search_type": "faq_only",
    "confidence": 0.95
  }
  ```

### 2. 健康检查

- **URL**：`/api/health`
- **方法**：GET
- **响应**：
  ```json
  {
    "status": "healthy",
    "service": "Historical FAQ Bot API"
  }
  ```

### 3. 服务信息

- **URL**：`/api/info`
- **方法**：GET
- **响应**：
  ```json
  {
    "name": "Historical FAQ Bot",
    "version": "1.0.0",
    "description": "基于RAG架构的历史人物问答系统",
    "features": ["FAQ检索", "文档检索(RAG)", "混合检索", "本地/API LLM支持"]
  }
  ```

## 模型配置

### 1. 嵌入模型

- **默认路径**：`/root/autodl-tmp/models/bge-m3-finetuned-wang`
- **支持模型**：BGE-M3（原始或微调）

### 2. 重排序模型

- **默认路径**：`/root/autodl-tmp/models/maidalun/bce-reranker-base_v1`

### 3. LLM模型

- **本地模式**：`/root/autodl-tmp/models/qwen/Qwen1.5-7B-Chat`
- **API模式**：支持DeepSeek、OpenAI、Claude等

## 数据准备

1. 将原始文档放入 `data/raw/` 目录
2. 运行数据处理脚本生成QA对
3. 微调嵌入模型（可选）

## 性能优化

- **批量处理**：调整批处理大小提高效率
- **缓存**：启用模型缓存减少加载时间
- **硬件加速**：使用GPU加速模型推理
- **索引优化**：为向量字段创建索引

## 部署建议

- **生产环境**：使用Docker部署，设置具体的CORS域名
- **监控**：添加日志监控和性能指标
- **备份**：定期备份数据库和模型

## 常见问题

### 1. 模型加载失败

检查模型路径是否正确，确保模型文件存在。

### 2. 数据库连接失败

检查PostgreSQL服务是否运行，连接参数是否正确。

### 3. 响应速度慢

- 检查硬件资源是否充足
- 调整批处理大小和缓存设置
- 考虑使用更轻量的模型

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License