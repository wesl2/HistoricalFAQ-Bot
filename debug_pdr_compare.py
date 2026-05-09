"""
对比 LangChain ParentDocumentRetriever 与我们的系统
"""

print("=" * 70)
print("LangChain ParentDocumentRetriever 核心机制")
print("=" * 70)

print("""
【LangChain PDR 架构】

1. 双重切分器:
   - parent_splitter: RecursiveCharacterTextSplitter(chunk_size=2000)
     → 把整本书切成 2000 字的"大 chunk"(parent)
   - child_splitter: RecursiveCharacterTextSplitter(chunk_size=400)  
     → 把每个 parent 再切成 400 字的"小 chunk"(child)

2. 存储结构:
   - vectorstore (Chroma/PGVector): 存 child chunks + embeddings
   - docstore (InMemoryStore): 存 parent chunks, key=uuid
   
   每个 child 的 metadata 里有 parent_id

3. 检索流程:
   query → embedding → 向量检索 child → 拿到 top_k children
   → 提取它们的 parent_id → 去 docstore 查 parent
   → 返回 parent chunks (2000字)

4. 关键设计:
   - parent 是 2000 字，不是整章
   - parent 之间有边界，不会无限长
   - 去重按 parent_id (uuid)，不是按 content 字符串
""")

print("=" * 70)
print("我们的系统 vs LangChain PDR")
print("=" * 70)

comparison = """
| 维度 | LangChain PDR | 我们的系统 |
|------|---------------|-----------|
| Parent 粒度 | 2000 字(可配置) | 7661 字(整章) |
| Child 粒度 | 400 字(可配置) | 300 字(TokenTextSplitter) |
| 存储 | vectorstore + docstore | PostgreSQL 单表 |
| Parent 标识 | uuid | parent_text 字符串 |
| 去重键 | parent_id | parent_text 内容 |
| 检索返回 | parent chunk | parent_text |
| 重叠控制 | parent_splitter 参数 | 无(整章) |

【核心区别】

1. 粒度不同:
   - LangChain: parent=2000字, 一个章节可能拆成多个 parent
   - 我们: parent=7661字, 一个章节=一个 parent
   
   → 他们的 parent 更短, 引用归属更明确

2. 去重方式不同:
   - LangChain: 按 uuid 去重, 精确可靠
   - 我们: 按 parent_text 字符串去重, 如果两个 parent 内容相同会误判

3. 上下文窗口不同:
   - LangChain: 2000字足够 LLM 理解, 又不会太长
   - 我们: 7661字可能包含多个主题, LLM 引用时容易混淆

【LangChain PDR 的优点】

1. parent 粒度可控: 2000字比 7661字更精准
2. 检索-生成分离: child 负责找, parent 负责读
3. 去重简单: uuid 不会出错
4. 工程成熟: 被大量生产环境验证

【LangChain PDR 的局限】

1. 需要维护两套存储(vectorstore + docstore)
2. parent 仍然是粗粒度的, 引用幻觉问题只是缓解, 未根治
3. 对跨 parent 的事实(如"比较A和B"), 可能召回不全
"""
print(comparison)
