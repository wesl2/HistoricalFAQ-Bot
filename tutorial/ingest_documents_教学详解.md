# ingest_documents.py 教学详解

> **目标读者**：已经完成 HistoricalFAQ-Bot 项目基础课程，希望理解"生产级数据导入脚本"设计思想的同学。  
> **配套文件**：`scripts/ingest_documents.py`、`config/custom_dict_replace.json`

---

## 一、为什么需要这个脚本？

在你之前的练习中，文档数据要么是通过 SQL 手工插入的：

```sql
INSERT INTO doc_chunks (chunk_text, doc_name, doc_page) VALUES ('...', '...', 1);
```

要么是通过一个极其简陋的脚本来做"读 txt → 固定长度切 → 生成 embedding → 插表"。

这两种方式在**练手阶段**没问题，但一旦进入真实场景（处理几十上百本 PDF、需要定期更新、不能每次全量重建），就会暴露大量问题：

| 问题 | 后果 |
|------|------|
| 没有清洗 | OCR 断行、PDF 页眉页脚、水印直接进数据库，BM25 和向量检索质量暴跌 |
| 固定长度切分 | 把句子从中间切断，chunk 语义不连贯 |
| 没有上下文增强 | 检索到的 chunk 缺少来源信息，LLM 回答时无法正确引用 |
| 逐条生成 embedding | BGE-M3 等模型支持 batch 推理，逐条调用浪费 3-5 倍 GPU 时间 |
| 没有增量机制 | 每次导入都要全量重建，100 本书只改 1 本也要全部重算 |
| embedding 失败塞零向量 | PGVector 余弦相似度公式除以零，查询结果完全错乱 |

**`ingest_documents.py` 就是为了系统性地解决以上问题而设计的。**

---

## 二、整体架构图

```
原始文档 (PDF/TXT/DOCX)
    │
    ▼
┌─────────────────────────────────────────┐
│  模块 0: 文件级 MD5 缓存检查               │  ← 增量同步的核心
│  （文件未变 → 直接跳过，省 99% 算力）      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  模块 1: 文本清洗 (TextCleaner)           │
│  1.1 wash_ocr   : 断行重组、OCR 字间空格   │
│  1.2 denoise    : 页眉页脚、水印、出版信息   │
│  1.3 normalize  : 统一空白、压缩换行        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  模块 2: 语义递归分块                      │
│  (RecursiveCharacterTextSplitter)        │
│  优先在段落/句子边界切分，避免切断语义      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  模块 3: 上下文增强                        │
│  (Contextual Chunk Headers)              │
│  为每个 chunk  prepend 来源文档名和页码    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  模块 4: 批量 Embedding + 失败降级         │
│  利用 BGE-M3 batch 能力，失败时逐条降级    │
│  单条再失败 → 丢弃该 Chunk（不塞零向量）   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  模块 5: 数据库写入                        │
│  execute_values 批量插入，增量模式先删旧插新 │
└─────────────────────────────────────────┘
```

---

## 三、逐模块详解

### 模块 0：文件级 MD5 缓存

**核心问题**：如果你有 100 个文档，只修改了其中 1 个，理想情况下应该只重新处理那 1 个。  
**实现方式**：在脚本同级的项目根目录下维护一个 `.ingest_cache.json`：

```json
{
  "唐史": "a1b2c3d4...",
  "王洪文传": "e5f6g7h8..."
}
```

键是文档名（`Path.stem`，即不带后缀的文件名），值是文件内容的 MD5 哈希。

**关键设计点**：
- 缓存检查发生在**解析和 Embedding 之前**。如果命中，直接 `return 0`，连 `DocumentProcessor.load_document()` 都不调用。
- 缓存**只在 `--append` 增量模式下生效**。全量模式（默认）会忽略缓存，始终重新处理。
- 缓存文件是"可降级"的：读写失败不会阻断主流程，只是打 warning。

**为什么不用 chunk 级 MD5 做缓存？**  
chunk 级哈希确实能去重，但它无法避免"解析 + 分词 + embedding 计算"这些最耗时的步骤。文件级哈希在流程最前端拦截，是真正的"增量同步"。

---

### 模块 1：TextCleaner 文本清洗

TextCleaner 是三层递进式清洗器，按顺序执行：

#### 1.1 `wash_ocr` —— OCR 断行重组

**症状**：OCR 扫描的 PDF 经常出现：
```
王洪文在1973年中共十大
上晋升为中共中央副主席。
```

**目标**：把它恢复成正常段落：
```
王洪文在1973年中共十大上晋升为中共中央副主席。
```

**实现逻辑**：
1. 按行读取。
2. 检测"句末标点"（`。！？` 等）。如果一行以句末标点结尾，就认为这是一个完整句子的结束，把之前暂存的 buffer 和它拼接起来，形成一个完整段落。
3. 如果不是句末标点，就把这一行追加到 buffer，等待下一句拼接。

**空格处理（重点修复）**：

旧版代码有一行：
```python
line = line.replace(" ", "").replace("\u3000", "")
```

这会把**所有半角空格**全部删掉，包括：
- "Deep Seek" → "DeepSeek"（英文单词粘连）
- "2024 04 17" → "20240417"（日期可读性下降）
- "https://example.com/path to/file" → 链接断裂

**修复后**：
```python
line = line.replace("\u3000", "")  # 全角空格无条件删除
line = re.sub(r'(?<=[\u4e00-\u9fa5]) (?=[\u4e00-\u9fa5])', '', line)  # 只删中文字间的空格
```

正则解释：
- `(?<=[\u4e00-\u9fa5])` ："正向后行断言"，要求空格左边是中文字符
- `(?=[\u4e00-\u9fa5])` ："正向前行断言"，要求空格右边是中文字符
- 合起来：只匹配 `"王 洪 文"` 这种模式，不碰英文和数字。

#### 1.2 `denoise` —— 版式噪声去除

这一层用正则表达式切除各种"版式垃圾"：

| 正则 | 清除目标 |
|------|---------|
| `RN[（\(《]\d+[）\)）。]` | PDF 页码标记如 `RN(7)`、`RN《16` |
| `@.*?\(Page#\d+\)` | 出版信息引用块，如 `@作者...一九九三年版(Page#508)` |
| `={5,}` | 手动分隔线 `==========` |
| `第[一二三四...]+章.*?\d+` | 混入正文的章节标题+页码，如 `第十四章...485` |
| `Document\s*generated\s*by\s*Anna's\s*Archive` | 电子书水印 |

**特定事实修正**：  
有一个特殊需求：某些 OCR 错误需要替换特定文字（如"一九七八年"→"一九七六年"）。这类规则**绝对不能硬编码在 Python 代码里**，因为不同文档的修正需求完全不同。

解决方式：抽离到 `config/custom_dict_replace.json`：

```json
{
  "replacements": [
    {
      "old": "一九七八年在毛泽东去世之后",
      "new": "一九七六年在毛泽东去世之后",
      "note": "王洪文传 OCR 错误修正示例"
    }
  ]
}
```

`TextCleaner.denoise()` 运行时动态读取该 JSON。通用代码与业务数据完全解耦。

#### 1.3 `normalize` —— 格式化

```python
text = re.sub(r"\n+", "\n", text)  # 多个连续换行压缩为单个
text = text.strip()                 # 去首尾空白
```

**注意**：旧版有一行 `text.replace(r"\n", "\n")` 是 bug。`r"\n"` 表示反斜杠 + n 两个**字符**，不是换行符。这行代码没有任何作用，已被删除。

---

### 模块 2：语义递归分块

**为什么不自己写 `while start < len(text)` 来切？**

固定长度切分最大的问题：**切断语义**。比如：
```
chunk 1: "...玄武门之变发生在唐朝"
chunk 2: "初期，与李世民相关。..."
```

"唐朝"和"初期"被切开了，向量模型对 chunk 1 的理解变成"某个变发生在唐朝"，语义严重受损。

**RecursiveCharacterTextSplitter 的策略**：

它维护一个"分隔符优先级列表"，按顺序尝试：
1. `\n\n`（段落边界）
2. `\n`（换行）
3. `。`、`！`、`？`（句子边界）
4. ` `（空格）
5. ``（字符，最后手段）

它会先尝试在最高优先级的分隔符处切分；如果某段 still 太长，就降级到下一个分隔符。这样最大程度保证了切分点落在语义边界上。

**参数含义**：
- `chunk_size=1024`：目标长度是 1024 个字符（约 500 个中文汉字）。
- `chunk_overlap=128`：相邻 chunk 重叠 128 字符。防止关键信息刚好落在边界上被切掉。

**长度过滤**：  
分块后丢弃 `< 50 字符` 的碎块。这些通常是标题残留、页码、或清洗后的边角料，对检索和回答没有价值。

---

### 模块 3：上下文增强（Contextual Chunk Headers）

**问题**：假设 chunk 内容是：
```
"他于当天去世，终年五十岁。"
```

向量模型不知道"他"是谁，用户问"王洪文什么时候去世"时，这个 chunk 的语义匹配度会很低。

**Anthropic 的解决方案**：在每个 chunk 前 prepend 一段"上下文头"，让 embedding 捕获来源信息。

本脚本的实现：
```python
header = f"【来源：《{doc_name}》第{doc_page}页】"
return f"{header}\n{content}"
```

最终入库的 `chunk_text` 变成：
```
【来源：《王洪文传》第128页】
他于当天去世，终年五十岁。
```

**好处**：
1. **向量检索**："王洪文传"这四个字也被编码进向量，提升语义相关性。
2. **BM25 检索**：用户搜"王洪文传"时，即使正文没出现这四个字，头部信息也能让 BM25 命中。
3. **LLM 回答**：检索结果自带来源，方便后续做引用溯源。

---

### 模块 4：批量 Embedding + 失败降级

#### BGE-M3 的 batch 优势

BGE-M3 支持一次传入多条文本，内部并行编码，速度比逐条调用快 **3-5 倍**。

```python
# 批量调用（推荐）
vectors = get_embedding(["文本1", "文本2", "文本3", ...])  # 返回 List[List[float]]

# 逐条调用（慢）
for text in texts:
    vec = get_embedding(text)  # 每次都要重新走模型前向传播的开销
```

#### 批量失败怎么办？

如果 batch 里某条文本异常（比如包含特殊字符导致 tokenizer 报错），整个 batch 会失败。  
**解决方案**：批量优先，失败时降级为逐条。

```python
try:
    result = get_embedding(batch)  # 尝试批量
except Exception:
    # 整批失败，降级为逐条
    for single_text in batch:
        try:
            vec = get_embedding(single_text)
        except Exception:
            vec = None  # 单条也失败，丢弃
```

#### 零向量问题（生产大忌）

旧版代码在失败时塞入 `[0.0] * 1024`。

**数学问题**：PGVector 的余弦相似度公式是：
```
similarity = dot(a, b) / (||a|| * ||b||)
```
如果 `a` 是零向量，`||a|| = 0`，分母为零 → **除以零错误**，查询结果完全不可预期。

**修复后**：失败返回 `None`，上游组装记录时直接跳过：

```python
for text, vec, meta in zip(enriched_texts, vectors, meta_infos):
    if vec is None:
        continue  # 丢弃，不写库
```

---

### 模块 5：数据库写入

#### 批量插入：execute_values

`psycopg2.extras.execute_values` 是真正的批量插入：一次网络往返插入多行，比 `execute()` 循环快 **10-100 倍**。

```python
from psycopg2.extras import execute_values
execute_values(cursor, insert_sql, records)
```

#### 全量 vs 增量模式

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| 全量（默认） | `TRUNCATE TABLE` → 插入所有 | 第一次导入、需要完全重建 |
| 增量（`--append`） | `DELETE WHERE doc_name = ?` → 插入该文档新 chunk | 定期更新、只增改少量文档 |

增量模式保证了"单文档级幂等"：同一个文档反复跑脚本，库里不会出现重复 chunk。

---

## 四、完整使用示例

### 第一步：建表（只跑一次）

```bash
python scripts/init_db.py
```

### 第二步：准备数据

```bash
mkdir -p data/raw
# 放入你的文档
cp ~/唐史.txt data/raw/
cp ~/王洪文传.pdf data/raw/
```

### 第三步：全量导入

```bash
python scripts/ingest_documents.py ./data/raw
```

日志输出示例：
```
2026-04-17 12:00:00 - INFO - 全量模式：已清空 doc_chunks
2026-04-17 12:00:05 - INFO - 文档处理完成：1 页 -> 15 块 -> 去重后 14 块
Embedding 批次: 100%|████████| 2/2 [00:03<00:00]
2026-04-17 12:00:10 - INFO - 数据库写入成功：14 条记录
2026-04-17 12:00:10 - INFO - 全部完成，总计导入 14 个文档块
```

### 第四步：修改一个文档后增量更新

```bash
# 编辑 data/raw/唐史.txt，加了一段新内容
python scripts/ingest_documents.py ./data/raw --append
```

如果 `唐史.txt` 内容没变：
```
2026-04-17 12:05:00 - INFO - 文件未变化，跳过处理：唐史
```

如果变了：
```
2026-04-17 12:05:00 - INFO - 增量模式：已删除旧记录 14 条（唐史）
2026-04-17 12:05:05 - INFO - 数据库写入成功：16 条记录
```

---

## 五、朋友的 Code Review 精华总结

你朋友给出了两次 review，核心要点汇总如下：

### 已修复的问题（当前版本已解决）

| # | 问题 | 修复方式 |
|---|------|---------|
| 1 | 硬编码业务数据 | 抽离到 `config/custom_dict_replace.json` |
| 2 | 全零向量兜底 | 失败返回 `None`，上游丢弃 |
| 3 | Chunk 级 MD5 粒度过细 | 增加文件级 MD5 缓存 `.ingest_cache.json` |
| 4 | `wash_ocr` 删除所有半角空格 | 改为只删全角空格 + 中文字间 OCR 空格 |
| 5 | `normalize` 的 `r"\n"` bug | 删除无意义行 |
| 6 | `isinstance(batch, str)` 永假 | 改为基于 `result[0]` 结构判断 |

### 未来升级方向（生产化 v2）

如果你要把这套脚本推到真正的生产环境，建议继续补齐以下 6 项：

#### 1. 表结构扩展

当前 `doc_chunks` 的字段比较精简。建议增加：

```sql
ALTER TABLE doc_chunks ADD COLUMN chunk_hash VARCHAR(32) UNIQUE;
ALTER TABLE doc_chunks ADD COLUMN source_path TEXT;
ALTER TABLE doc_chunks ADD COLUMN embedding_model VARCHAR(50);
ALTER TABLE doc_chunks ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE doc_chunks ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
```

- `chunk_hash`：做到真正的幂等（即使 doc_name 相同，内容 hash 也唯一）。
- `embedding_model`：记录生成向量的模型版本，方便后续做模型迁移。
- `created_at/updated_at`：数据新鲜度管理。

#### 2. 任务可恢复（断点续传）

当前如果处理到第 50 个文件时程序崩溃，前面 49 个已经入库，但缓存可能没保存。  
建议：每处理完一个文件就 `save_hash_cache()` 一次，而不是最后统一保存。

#### 3. Embedding 模型切换的迁移机制

如果你从 BGE-M3 切换到 text-embedding-3-small，维度可能从 1024 变成 1536。  
建议：入库时记录 `embedding_model` 和 `embedding_dim`，检索时校验兼容性。

#### 4. 数据质量抽检

定期抽样检查入库的 chunk：
- 平均长度分布
- embedding 为 None 的比例
- 重复内容比例

可以用一个简单的 SQL：
```sql
SELECT doc_name, COUNT(*), AVG(LENGTH(chunk_text))
FROM doc_chunks
GROUP BY doc_name;
```

#### 5. 异步 / 队列化

当文档量达到万级时，串行处理太慢。  
建议：用 `multiprocessing.Pool` 并行处理多个文件，或用 Redis/RabbitMQ 做任务队列。

#### 6. 监控指标

记录以下指标并推送至 Prometheus/Grafana：
- `ingest_documents_total`：处理文件总数
- `ingest_chunks_total`：生成 chunk 总数
- `ingest_embedding_failures_total`：embedding 失败数
- `ingest_duration_seconds`：单次导入耗时

---

## 六、自问自答（FAQ）

**Q1：为什么 chunk_size 用 1024 字符，而不是 512 tokens？**  
A：`RecursiveCharacterTextSplitter` 默认按字符数（character）切分，不是 token 数。1024 字符约等于 500 个中文汉字，对应 BGE-M3 的 512 tokens 左右（中文一字约等于 0.5~1 token）。这是经验值，可根据实际召回效果调整。

**Q2：`--append` 模式和文件级缓存有什么区别？**  
A：`--append` 是"数据库层面的增量"（只替换同名文档的旧记录）。文件级缓存是"计算层面的增量"（文件未变就不做解析和 Embedding）。两者配合使用，才能真正做到"只处理变化的文档"。

**Q3：如果我想加一个新的去噪规则，应该改哪里？**  
A：通用规则（如某种新发现的水印模式）加到 `TextCleaner.denoise()` 里。特定文档的事实修正加到 `config/custom_dict_replace.json` 里。

**Q4：`.`ingest_cache.json` 可以删掉吗？**  
A：可以。删掉后下次 `--append` 会重新计算所有文件的 MD5，但不会影响数据库里的数据。

---

## 七、总结

你现在手中的 `ingest_documents.py` 已经具备以下工业界特征：

- ✅ 三层递进式清洗（OCR 重组 → 版式去噪 → 格式化）
- ✅ 语义递归分块（保护句子边界）
- ✅ 上下文增强（提升检索和引用质量）
- ✅ 批量 Embedding + 失败降级（高效且健壮）
- ✅ 增量同步 + 文件级缓存（避免重复计算）
- ✅ 零向量丢弃（保护数据库查询正确性）
- ✅ 通用代码与业务数据解耦（custom_dict_replace.json）

这套脚本已经从"练手版"进化到了"工程化 v1"，再往 v2 走只需要补上文提到的 6 个升级项即可。
