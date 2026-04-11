# # -*- coding: utf-8 -*-
# """
# LangChain 集成模块

# 封装 LangChain 相关功能，提供与原有系统的集成
# """

# import logging
# from pathlib import Path
# from typing import List, Dict, Any, AsyncGenerator, Iterator

# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.documents import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma, FAISS
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_community.document_loaders import Docx2txtLoader as DocxLoader  # 0.2.x 中改名为 Docx2txtLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
# from langchain_community.chat_message_histories import PostgresChatMessageHistory
# # Memory 类在标准模块中重新实现
# # ConversationalRetrievalChain 在 LangChain 1.x 中已移除，推荐使用 LCEL 组合式
# # from langchain_community.memory import ConversationBufferMemory, ConversationSummaryMemory
# # from langchain.chains import ConversationalRetrievalChain
# #TODO: 现代推荐做法（新范式）不要直接用 ConversationalRetrievalChain。现在的标准做法是使用 组合式构建（Conversational RAG）：
# from langchain_core.tools import Tool
# # Agent 模块在 LangChain 1.x 中完全重构，暂时注释
# # from langchain.agents import initialize_agent, AgentType

# from src.embedding.embedding_local import get_embedding
# from src.llm.standard_llm import get_standard_llm
# # llm_factory 已删除，使用 standard_llm
# # from src.llm.llm_factory import LLMFactory
# # advanced_retriever 暂时注释，因为 LangChain 1.x 导入变化
# # from src.rag.advanced_retriever import AdvancedRetriever, get_advanced_retriever
# from src.rag.callbacks import get_callback_manager
# from config.model_config import EMBEDDING_CONFIG, LANGCHAIN_CONFIG
# from config.pg_config import PG_URL, PG_CHAT_TABLE

# logger = logging.getLogger(__name__)


# class LangChainIntegration:
#     """
#     LangChain 集成类
    
#     功能：
#     - 文档加载和分块
#     - 向量库管理（支持持久化）
#     - RAG Chain（支持 Streaming）
#     - 对话记忆
#     - Agent 功能
#     - Callback 监控
#     """
    
#     def __init__(self, llm_mode=None, use_advanced_retriever=True, session_id: str = None):
#         """
#         初始化 LangChain 集成
        
#         Args:
#             llm_mode: LLM 模式
#             use_advanced_retriever: 是否使用高级检索器（multi-query + rerank）
#             session_id: 会话ID，用于PostgreSQL持久化存储对话历史
#         """
#         self.llm = get_standard_llm(llm_mode)
#         self.embeddings = self._init_embeddings()
#         self.session_id = session_id
        
#         # 使用 PostgreSQL 持久化存储对话历史（方案A）
#         if session_id:
#             self.memory = PostgresChatMessageHistory(
#                 connection_string=PG_URL,
#                 session_id=session_id,
#                 table_name=PG_CHAT_TABLE
#             )
#             logger.info(f"使用 PostgreSQL 持久化对话历史: session_id={session_id}")
#         else:
#             # 降级为内存存储
#             self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
#             logger.warning("未提供 session_id，使用内存存储对话历史（非持久化）")
        
#         self.summary_memory = ConversationSummaryMemory(llm=self.llm, memory_key="history")
        
#         # 高级检索器
#         self.use_advanced_retriever = use_advanced_retriever
#         self.advanced_retriever = None
#         if use_advanced_retriever:
#             self.advanced_retriever = get_advanced_retriever()
        
#         # 回调管理器
#         self.callback_manager = get_callback_manager()
        
#         # 提示词缓存
#         self._prompts = {}
#         self._load_prompts()
        
#         logger.info(f"LangChain 集成初始化完成，高级检索器: {use_advanced_retriever}")
    
#     def _init_embeddings(self):
#         """初始化嵌入模型"""
#         return HuggingFaceEmbeddings(
#             model_name=EMBEDDING_CONFIG["model_path"],
#             model_kwargs={"device": EMBEDDING_CONFIG["device"]},
#             encode_kwargs={"normalize_embeddings": EMBEDDING_CONFIG["normalize"]}
#         )
    
#     def _load_prompts(self):
#         """加载提示词模板"""
#         prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        
#         prompt_files = {
#             "rag": "rag_template.txt",
#             "conversational": "conversational_template.txt",
#             "multi_query": "multi_query_template.txt"
#         }
        
#         for key, filename in prompt_files.items():
#             filepath = prompts_dir / filename
#             if filepath.exists():
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     self._prompts[key] = f.read()
#             else:
#                 logger.warning(f"提示词文件不存在: {filepath}")
    
#     def _get_prompt(self, key: str, default: str = None) -> str:
#         """获取提示词模板"""
#         return self._prompts.get(key, default)
    
#     # ==================== 文档处理 ====================
    
#     def load_documents(self, file_path: str) -> List[Document]:
#         """
#         加载文档
        
#         Args:
#             file_path: 文件路径
            
#         Returns:
#             文档列表
#         """
#         if file_path.endswith('.pdf'):
#             loader = PyPDFLoader(file_path)
#         elif file_path.endswith('.docx'):
#             loader = DocxLoader(file_path)
#         else:
#             loader = TextLoader(file_path, encoding='utf-8')
        
#         return loader.load()
    
#     def split_documents(self, documents: List[Document], 
#                        chunk_size: int = None,
#                        chunk_overlap: int = None,
#                        splitter_type: str = None) -> List[Document]:
#         """
#         文本分块
        
#         Args:
#             documents: 文档列表
#             chunk_size: 块大小
#             chunk_overlap: 块重叠
#             splitter_type: 分块器类型
            
#         Returns:
#             分块后的文档
#         """
#         doc_config = LANGCHAIN_CONFIG["document"]
#         chunk_size = chunk_size or doc_config["chunk_size"]
#         chunk_overlap = chunk_overlap or doc_config["chunk_overlap"]
#         splitter_type = splitter_type or doc_config["splitter_type"]
        
#         if splitter_type == "recursive":
#             splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=chunk_size,
#                 chunk_overlap=chunk_overlap,
#                 length_function=len,
#                 add_start_index=True
#             )
#         else:
#             splitter = CharacterTextSplitter(
#                 chunk_size=chunk_size,
#                 chunk_overlap=chunk_overlap,
#                 separator="\n"
#             )
        
#         return splitter.split_documents(documents)
    
#     def process_file(self, file_path: str, **kwargs) -> List[Document]:
#         """
#         处理文件（加载 + 分块）
        
#         Args:
#             file_path: 文件路径
#             **kwargs: 分块参数
            
#         Returns:
#             处理后的文档块列表
#         """
#         documents = self.load_documents(file_path)
#         return self.split_documents(documents, **kwargs)
    
#     # ==================== 向量库管理（支持持久化） ====================
    
#     def create_vectorstore(self, documents: List[Document], 
#                           vectorstore_type: str = None,
#                           persist: bool = True) -> Any:
#         """
#         创建向量存储（支持持久化）
        
#         Args:
#             documents: 文档列表
#             vectorstore_type: 向量存储类型 (chroma 或 faiss)
#             persist: 是否持久化
            
#         Returns:
#             向量存储实例
#         """
#         if self.use_advanced_retriever and persist:
#             # 使用高级检索器的持久化功能
#             self.advanced_retriever.create_vectorstore(documents, vectorstore_type or "chroma")
#             return self.advanced_retriever.vectorstore
        
#         # 传统方式（不持久化或临时使用）
#         vectorstore_type = vectorstore_type or LANGCHAIN_CONFIG["vectorstore"]["type"]
        
#         if vectorstore_type == "faiss":
#             return FAISS.from_documents(
#                 documents=documents,
#                 embedding=self.embeddings
#             )
#         else:
#             return Chroma.from_documents(
#                 documents=documents,
#                 embedding=self.embeddings
#             )
    
#     def load_vectorstore(self, persist_directory: str = None, vectorstore_type: str = "chroma") -> Any:
#         """
#         加载持久化的向量库
        
#         Args:
#             persist_directory: 持久化目录
#             vectorstore_type: 向量库类型
            
#         Returns:
#             向量库实例
#         """
#         persist_directory = persist_directory or LANGCHAIN_CONFIG["vectorstore"]["persist_directory"]
        
#         if vectorstore_type == "chroma":
#             import chromadb
#             return Chroma(
#                 persist_directory=persist_directory,
#                 embedding_function=self.embeddings
#             )
#         else:
#             return FAISS.load_local(persist_directory, self.embeddings)
    
#     # ==================== 检索器 ====================
    
#     def create_retriever(self, vectorstore, k: int = None, search_type: str = None):
#         """
#         创建基础检索器
        
#         Args:
#             vectorstore: 向量存储实例
#             k: 返回结果数量
#             search_type: 搜索类型
            
#         Returns:
#             检索器实例
#         """
#         k = k or LANGCHAIN_CONFIG["vectorstore"]["k"]
#         search_type = search_type or LANGCHAIN_CONFIG["vectorstore"]["search_type"]
        
#         return vectorstore.as_retriever(
#             search_type=search_type,
#             search_kwargs={"k": k}
#         )
    
#     def retrieve(self, query: str, k: int = 5) -> List[Document]:
#         """
#         检索文档（使用高级检索器）
        
#         Args:
#             query: 查询文本
#             k: 返回数量
            
#         Returns:
#             检索到的文档列表
#         """
#         if self.advanced_retriever and self.advanced_retriever.retriever:
#             return self.advanced_retriever.retrieve(query, k=k)
        
#         logger.error("高级检索器未初始化")
#         return []
    
#     # ==================== RAG Chain（支持 Streaming） ====================
    
#     def create_rag_chain(self, retriever=None, output_format: str = "text", stream: bool = False):
#         """
#         创建 RAG 链
        
#         Args:
#             retriever: 检索器（为 None 则使用高级检索器）
#             output_format: 输出格式 (text 或 json)
#             stream: 是否启用流式输出
            
#         Returns:
#             RAG 链实例
#         """
#         # 使用外部提示词
#         template = self._get_prompt("rag", """基于以下资料回答问题：
# {context}

# 问题：{question}""")
        
#         prompt = ChatPromptTemplate.from_template(template)
        
#         # 格式化文档
#         def format_docs(docs):
#             return "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])
        
#         # 选择检索器
#         if retriever is None and self.advanced_retriever:
#             # 使用高级检索器的 retrieve 方法
#             def retrieve_docs(query):
#                 return self.advanced_retriever.retrieve(query, k=5)
#             retriever_runnable = RunnableLambda(retrieve_docs)
#         else:
#             retriever_runnable = retriever
        
#         # 选择输出解析器
#         if output_format == "json":
#             parser = JsonOutputParser()
#         else:
#             parser = StrOutputParser()
        
#         # 创建链
#         chain = (
#             {
#                 "context": retriever_runnable | format_docs,
#                 "question": RunnablePassthrough()
#             }
#             | prompt
#             | RunnableLambda(lambda x: self._invoke_llm(x, stream))
#             | parser
#         )
        
#         return chain
    
#     def _invoke_llm(self, prompt_value, stream: bool = False):
#         """
#         调用 LLM（支持流式）
        
#         Args:
#             prompt_value: 提示词
#             stream: 是否流式输出
            
#         Returns:
#             LLM 输出
#         """
#         messages = [{"role": "user", "content": str(prompt_value)}]
        
#         if stream:
#             # 返回生成器
#             return self.llm.stream(messages)
#         else:
#             return self.llm.chat(messages)
    
#     def create_streaming_chain(self, retriever=None) -> Any:
#         """
#         创建流式输出链
        
#         Returns:
#             流式链实例
#         """
#         return self.create_rag_chain(retriever=retriever, stream=True)
    
#     def stream(self, query: str) -> Iterator[str]:
#         """
#         流式生成回答
        
#         Args:
#             query: 查询文本
            
#         Yields:
#             生成的文本片段
#         """
#         chain = self.create_streaming_chain()
#         callbacks = self.callback_manager.get_callbacks()
        
#         for chunk in chain.stream(query, config={"callbacks": callbacks}):
#             yield chunk
    
#     async def astream(self, query: str) -> AsyncGenerator[str, None]:
#         """
#         异步流式生成回答
        
#         Args:
#             query: 查询文本
            
#         Yields:
#             生成的文本片段
#         """
#         chain = self.create_streaming_chain()
#         callbacks = self.callback_manager.get_callbacks()
        
#         async for chunk in chain.astream(query, config={"callbacks": callbacks}):
#             yield chunk
    
#     # ==================== 对话链 ====================
    
#     def create_conversational_chain(self, retriever=None, use_summary_memory: bool = False):
#         """
#         创建对话链
        
#         Args:
#             retriever: 检索器
#             use_summary_memory: 是否使用摘要记忆
            
#         Returns:
#             对话链实例
#         """
#         template = self._get_prompt("conversational", """对话历史：{history}

# 参考资料：{context}

# 问题：{question}""")
        
#         prompt = ChatPromptTemplate.from_template(template)
        
#         def format_docs(docs):
#             return "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])
        
#         # 选择检索器
#         if retriever is None and self.advanced_retriever:
#             def retrieve_docs(query):
#                 return self.advanced_retriever.retrieve(query, k=5)
#             retriever_runnable = RunnableLambda(retrieve_docs)
#         else:
#             retriever_runnable = retriever
        
#         # 选择记忆
#         memory = self.summary_memory if use_summary_memory else self.memory
        
#         callbacks = self.callback_manager.get_callbacks()
        
#         # 创建链
#         chain = (
#             {
#                 "context": retriever_runnable | format_docs,
#                 "question": RunnablePassthrough(),
#                 "history": RunnableLambda(lambda x: memory.load_memory_variables({})["history"])
#             }
#             | prompt
#             | RunnableLambda(lambda x: self.llm.chat([{"role": "user", "content": str(x)}]))
#             | StrOutputParser()
#         )
        
#         return chain
    
#     def create_conversational_retrieval_chain(self, retriever):
#         """
#         创建对话检索链（传统方式）
        
#         Args:
#             retriever: 检索器
            
#         Returns:
#             对话检索链实例
#         """
#         callbacks = self.callback_manager.get_callbacks()
        
#         return ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=retriever,
#             memory=self.memory,
#             return_source_documents=True,
#             callbacks=callbacks
#         )
    
#     # ==================== 记忆管理 ====================
    
#     def save_context(self, input_text: str, output_text: str):
#         """
#         保存对话上下文
        
#         Args:
#             input_text: 用户输入
#             output_text: 系统输出
#         """
#         self.memory.save_context(
#             {"input": input_text},
#             {"output": output_text}
#         )
#         self.summary_memory.save_context(
#             {"input": input_text},
#             {"output": output_text}
#         )
    
#     def clear_memory(self):
#         """清除对话记忆"""
#         self.memory.clear()
#         self.summary_memory.clear()
    
#     # ==================== Agent ====================
    
#     def create_tool(self, name: str, func, description: str) -> Tool:
#         """
#         创建工具
        
#         Args:
#             name: 工具名称
#             func: 工具函数
#             description: 工具描述
            
#         Returns:
#             工具实例
#         """
#         return Tool(
#             name=name,
#             func=func,
#             description=description
#         )
    
#     def create_agent(self, tools: List[Tool], agent_type=None) -> Any:
#         """
#         创建智能体（暂时不可用，等待 LangChain 1.x Agent API 稳定）

#         Args:
#             tools: 工具列表
#             agent_type: 智能体类型

#         Returns:
#             智能体实例
#         """
#         raise NotImplementedError(
#             "Agent 功能在 LangChain 1.x 中已重构，"
#             "请使用标准模块 src.rag.standard_chain.build_agent_chain"
#         )
    
#     # ==================== 监控指标 ====================
    
#     def get_metrics(self) -> Dict[str, Any]:
#         """获取监控指标"""
#         return self.callback_manager.get_metrics()
    
#     def reset_metrics(self):
#         """重置监控指标"""
#         self.callback_manager.reset()
