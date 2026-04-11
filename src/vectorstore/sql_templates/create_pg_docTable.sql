CREATE TABLE IF NOT EXISTS {{table_name}}(
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    chunk_vector vector({{vector_dim}}), 
    doc_name VARCHAR(255) NOT NULL,
    doc_page INTEGER,
    chunk_index INTEGER,
    search_vector tsvector 
        GENERATED ALWAYS AS (to_tsvector('simple', chunk_text)) STORED,
    created_by VARCHAR(50) DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)