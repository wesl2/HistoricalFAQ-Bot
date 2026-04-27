# HistoricalFAQ-Bot 增强路线图：从 L1.5 → L3

## 一、Query Understanding（查询理解）

### 1.1 意图识别（Intent Classification）

**加在哪里：** `ChatEngine` 最前面，检索之前

```python
class ChatEngine:
    def chat(self, query: str):
        # 新增：意图识别
        intent = self._classify_intent(query)
        if intent == "chitchat":
            return self._handle_chitchat(query)
        elif intent == "knowledge":
            return self._handle_rag(query)
        elif intent == "clarification":
            return self._ask_clarification(query)
```

**实现方式（三选一）：**

| 方案 | 实现 | 优点 | 缺点 | 耗时 |
|-----|------|------|------|------|
| A. 规则匹配 | 关键词正则：`你好|在吗|谢谢` → 闲聊 | 简单、零成本 | 覆盖率低 | 2小时 |
| B. 小模型分类 | BERT 微调 / 开源意图模型 | 准确率高 | 要训练数据 | 1天 |
| C. LLM 判断 | `StandardLLM.ainvoke("判断意图：{query}")` | 无需训练 | 多一次 API 调用 | 2小时 |

**推荐方案 C（最快）：**

```python
async def _classify_intent(self, query: str) -> str:
    prompt = f"""判断以下用户查询的意图，只输出一个标签：
    - chitchat: 闲聊（打招呼、感谢、无意义）
    - knowledge: 知识问答（问人物、事件、历史）
    - clarification: 需要澄清（问题模糊、指代不明）
    
    查询：{query}
    意图："""
    resp = await StandardLLM.ainvoke(prompt, mode="api")
    return resp.content.strip()
```

### 1.2 Query Rewriting（查询改写 / 指代消解）

**加在哪里：** 检索之前，意图识别之后

```python
async def _rewrite_query(self, query: str, history: List[Dict]) -> str:
    """把"他"改成具体人名，把模糊问题改明确"""
    if not history:
        return query  # 没有历史，不用改
    
    prompt = f"""根据对话历史，改写当前查询，消除指代歧义。
    如果当前查询已经明确，直接原样返回。
    
    历史对话：
    {self._format_history(history[-3:])}
    
    当前查询：{query}
    改写后："""
    
    resp = await StandardLLM.ainvoke(prompt, mode="api")
    return resp.content.strip()
```

**例子：**
```
历史：用户问"王洪文是谁？"，AI 回答"王洪文是四人帮成员..."
当前查询："他后来怎么样了？"
改写后："王洪文后来怎么样了？"
```

### 1.3 Query Expansion（查询扩展）

**加在哪里：** 检索之前，改写之后

```python
async def _expand_query(self, query: str) -> List[str]:
    """生成 2-3 个相关查询，多路检索"""
    prompt = f"""基于以下查询，生成 2 个语义相关的扩展查询，用于检索更多资料。
    扩展查询应该覆盖不同的角度或同义表达。
    
    原查询：{query}
    
    扩展查询（每行一个，不要编号）："""
    
    resp = await StandardLLM.ainvoke(prompt, mode="api")
    expanded = [line.strip() for line in resp.content.strip().split('\n') if line.strip()]
    return [query] + expanded[:2]  # 原查询 + 2 个扩展
```

**例子：**
```
原查询："王洪文"
扩展：["王洪文 四人帮", "王洪文 生平事迹"]
```

**多路检索修改：**

```python
# SearchRouter.search() 改造
async def asearch(self, query: str) -> SearchContext:
    # 1. 扩展查询
    queries = await self._expand_query(query)
    
    # 2. 多路检索（每个扩展查询都检索）
    all_faq = []
    all_doc = []
    for q in queries:
        faq = await asyncio.to_thread(self.faq_retriever.retrieve, q)
        doc = await asyncio.to_thread(self.doc_retriever.retrieve, q)
        all_faq.extend(faq)
        all_doc.extend(doc)
    
    # 3. 去重
    all_faq = self._deduplicate(all_faq)
    all_doc = self._deduplicate(all_doc)
    
    # 4. 路由决策（用原 query 判断，不是扩展 query）
    return self._route(query, all_faq, all_doc)
```

---

## 二、动态 System Prompt + Few-shot

### 2.1 动态 System Prompt

**加在哪里：** `ResponseGenerator._build_messages()`

```python
def _build_messages(self, prompt: str, history_messages=None, search_type="hybrid", pure=False):
    # 根据检索类型动态调整 system prompt
    if pure:
        system = self._system_prompt_pure
    elif search_type == "faq_only":
        system = (
            "你是一位中国现代史专家。"
            "用户的问题在 FAQ 中有明确答案，请直接、简洁地回答。"
            "不要扩展无关内容。"
        )
    elif search_type == "doc_only":
        system = (
            "你是一位中国现代史专家。"
            "基于提供的文档片段回答，如果片段不足以回答，请明确说明。"
            "适当引用文档来源。"
        )
    else:  # hybrid
        system = self._system_prompt
    
    messages = [SystemMessage(content=system)]
    ...
```

### 2.2 Few-shot 示例选择

**加在哪里：** `ResponseGenerator._build_messages()`，system prompt 之后、user prompt 之前

```python
def _build_messages(self, prompt, history_messages=None, search_type="hybrid", pure=False):
    ...
    messages = [SystemMessage(content=system)]
    
    # 新增：动态选择 Few-shot 示例
    if not pure:
        few_shots = self._select_few_shots(search_type)
        messages.extend(few_shots)
    
    if history_messages:
        messages.extend(history_messages)
    messages.append(HumanMessage(content=prompt))
    return messages

def _select_few_shots(self, search_type: str) -> List[BaseMessage]:
    """根据检索类型选择预设的 few-shot 示例"""
    if search_type == "faq_only":
        return [
            HumanMessage(content="问：江青是谁？"),
            AIMessage(content="江青（1914-1991）是..."),  # 简洁风格
        ]
    elif search_type == "doc_only":
        return [
            HumanMessage(content="问：文化大革命的起因？"),
            AIMessage(content="根据《文革史》第3章记载..."),  # 引用风格
        ]
    return []  # hybrid 不设 few-shot，避免干扰
```

**更高级：从 FAQ 库中动态选择相似问答作为 few-shot**

```python
def _select_few_shots_from_faq(self, query: str, faq_results: List[FAQResult]) -> List[BaseMessage]:
    """选 1-2 条相似 FAQ 作为 few-shot"""
    shots = []
    for faq in faq_results[:2]:
        shots.append(HumanMessage(content=f"问：{faq.question}"))
        shots.append(AIMessage(content=f"答：{faq.answer[:200]}"))
    return shots
```

---

## 三、后处理（Post-processing）

### 3.1 事实校验（Fact Verification）

**加在哪里：** `ResponseGenerator.generate()` 返回前，或 `ChatEngine.chat()` 返回前

```python
class ResponseGenerator:
    async def _verify_facts(self, answer: str, sources: List[Dict]) -> Tuple[bool, str]:
        """
        用 LLM 自检：回答是否基于提供的资料？
        返回：(是否通过, 修正后的回答)
        """
        source_text = "\n".join([s.get("content", "") for s in sources])
        
        prompt = f"""你是一个严格的事实审核员。
        请判断以下回答是否完全基于提供的资料。
        如果发现回答中有资料未提及的内容，请删除或替换为"根据现有资料无法确定"。
        
        资料：
        {source_text[:2000]}
        
        回答：
        {answer}
        
        请输出审核后的回答（不要解释审核过程）："""
        
        resp = await StandardLLM.ainvoke(prompt, mode="api")
        return True, resp.content.strip()
```

**更轻量的方案：NLI（自然语言推理）小模型**

```python
from sentence_transformers import CrossEncoder

class FactChecker:
    def __init__(self):
        # 加载 NLI 模型
        self.model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    
    def verify(self, claim: str, evidence: str) -> bool:
        """判断 claim 是否被 evidence 支持"""
        score = self.model.predict([(evidence, claim)])[0]
        # entailment (支持) / contradiction (矛盾) / neutral (无关)
        return score[0] > 0.5  # entailment 分数
```

**耗时：NLI 模型方案 1 天，LLM 自检方案 2 小时。**

### 3.2 安全审核（Safety Guardrails）

**加在哪里：** `ChatEngine.chat()` 返回前

```python
class ChatEngine:
    def _safety_check(self, answer: str) -> Tuple[bool, str]:
        """安全审核，返回 (是否安全, 处理后文本)"""
        # 方案 A：关键词过滤（最快）
        blocked_words = ["反动", "推翻", "暴乱"]  # 自定义
        for word in blocked_words:
            if word in answer:
                return False, "该问题涉及敏感内容，无法回答。"
        
        # 方案 B：LLM 审核（更准确）
        prompt = f"""判断以下回答是否包含违法违规或敏感内容。
        只输出 SAFE 或 UNSAFE。
        
        回答：{answer[:500]}
        结果："""
        
        resp = StandardLLM.invoke(prompt, mode="api")
        if "UNSAFE" in resp.content:
            return False, "回答内容未通过安全审核。"
        
        return True, answer
```

**推荐：** 生产环境用关键词 + LLM 双层审核。

### 3.3 引用溯源（Citation Tracking）

**加在哪里：** `ResponseGenerator.build_prompt()` + `ChatEngine.chat()` 后处理

**Step 1：Prompt 里要求标注引用**

```python
def build_prompt(self, query, faq_results, doc_results):
    prompt_parts = ["请参考以下资料回答问题。回答时请在相关陈述后标注引用来源，格式如 [1]、[2]。\n"]
    
    for i, r in enumerate(faq_results[:self.max_faq_in_prompt], 1):
        prompt_parts.append(f"[{i}] FAQ：{r.question}")
        prompt_parts.append(f"    {self._truncate(r.answer)}")
    
    for i, r in enumerate(doc_results[:self.max_doc_in_prompt], 1):
        prompt_parts.append(f"[{i+len(faq_results)}] 文档：{r.doc_name} 第{r.doc_page}页")
        prompt_parts.append(f"    {self._truncate(r.content)}")
    
    prompt_parts.append(f"\n问题：{query}")
    prompt_parts.append("\n请基于以上资料回答，并标注引用来源。")
    return "\n".join(prompt_parts)
```

**Step 2：后处理解析引用标记**

```python
def _extract_citations(self, answer: str, sources: List[Dict]) -> Dict:
    """解析 [1]、[2] 标记，链接到真实来源"""
    import re
    citations = re.findall(r'\[(\d+)\]', answer)
    
    result = {"answer": answer, "citations": []}
    for c in set(citations):
        idx = int(c) - 1
        if 0 <= idx < len(sources):
            result["citations"].append({
                "id": c,
                "source": sources[idx].get("doc_name") or sources[idx].get("question"),
                "page": sources[idx].get("doc_page"),
            })
    return result
```

**返回给前端：**
```json
{
    "answer": "王洪文于1976年被捕[1]，1981年被判无期徒刑[2]。",
    "citations": [
        {"id": "1", "source": "文革史资料", "page": 45},
        {"id": "2", "source": "四人帮审判记录", "page": 128}
    ]
}
```

### 3.4 输出格式化

**加在哪里：** `ResponseGenerator` 或 `ChatEngine`

```python
def _format_output(self, answer: str, format_type: str = "markdown") -> str:
    if format_type == "json":
        return json.dumps({"answer": answer}, ensure_ascii=False)
    elif format_type == "markdown":
        # 自动加粗人名、加标题
        answer = re.sub(r'(王洪文|江青|张春桥|姚文元)', r'**\1**', answer)
        return answer
    return answer
```

---

## 四、加进项目的具体步骤

### 文件修改清单

| 文件 | 修改内容 |
|-----|---------|
| `src/chat/query_understanding.py` | **新增**：意图识别、查询改写、查询扩展 |
| `src/chat/response_generator.py` | **修改**：动态 system prompt、few-shot、引用标记 |
| `src/chat/fact_checker.py` | **新增**：事实校验（NLI 或 LLM 自检）|
| `src/chat/safety_guard.py` | **新增**：安全审核 |
| `src/chat/chat_engine.py` | **修改**：集成所有新模块，调整调用顺序 |

### 调用顺序改造

```python
async def achat(self, query: str) -> Dict:
    # 1. Query Understanding（新增）
    intent = await self._classify_intent(query)
    if intent == "chitchat":
        return await self._chitchat(query)
    
    rewritten = await self._rewrite_query(query, history)
    expanded_queries = await self._expand_query(rewritten)
    
    # 2. 多路检索（改造）
    search_context = await self._multi_search(expanded_queries)
    
    # 3. Reranker 精排（如果已有 BCE-Reranker）
    ranked_results = await self._rerank(rewritten, search_context)
    
    # 4. 生成（改造：动态 prompt + few-shot）
    answer = await self.response_gen.agenerate(
        rewritten, 
        ranked_results.faq_results,
        ranked_results.doc_results,
        history_messages,
        search_type=ranked_results.search_type.value,  # 传给动态 prompt
    )
    
    # 5. 后处理（新增）
    is_safe, answer = self._safety_check(answer)
    if not is_safe:
        return {"answer": answer, "error_code": "SAFETY_BLOCKED"}
    
    is_fact, answer = await self._verify_facts(answer, sources)
    
    # 6. 引用溯源（新增）
    result = self._extract_citations(answer, sources)
    
    # 7. 保存历史
    await self._asave_history("human", query)
    await self._asave_history("ai", result["answer"])
    
    return result
```

---

## 五、耗时估算

| 模块 | 工作量 | 耗时 |
|-----|--------|------|
| **Query Understanding** | 意图识别 + 改写 + 扩展 | 1-2 天 |
| **动态 System Prompt** | 改造 `_build_messages()` | 半天 |
| **Few-shot 选择** | 预设示例 + 动态选择 | 半天 |
| **Reranker 集成** | SearchRouter 调用 BCE-Reranker | 半天 |
| **事实校验** | LLM 自检 或 NLI 模型 | 1 天 |
| **安全审核** | 关键词 + LLM 双层 | 半天 |
| **引用溯源** | Prompt 改造 + 后处理解析 | 半天 |
| **联调测试** | 端到端验证 | 1 天 |
| **总计** | | **5-7 天** |

---

## 六、最小可行版本（MVP）

如果时间紧，只做这 **3 件事**，收益最大：

1. **Query Rewriting（1 天）**：解决"他"指代问题，准确率提升最明显
2. **Reranker 精排（半天）**：配置里已有 BCE-Reranker，SearchRouter 调用即可
3. **引用溯源（半天）**：Prompt 要求 LLM 标注 `[1]`、`[2]`，提升可信度

**总计 2 天，从 L1.5 提升到 L2.5。**
