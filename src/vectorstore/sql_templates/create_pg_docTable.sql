CREATE TABLE IF NOT EXISTS {{table_name}}(
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,        -- 小块文本：用于向量检索和 BM25
    parent_text TEXT,                -- 父级完整文本：召回后给 LLM 看完整上下文
    chunk_vector vector({{vector_dim}}), 
    doc_name VARCHAR(255) NOT NULL,
    doc_page INTEGER,
    chunk_index INTEGER,
    created_by VARCHAR(50) DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)