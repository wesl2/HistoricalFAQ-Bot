CREATE INDEX IF NOT EXISTS idx_hnsw_docTable on {{table_name}}
    USING hnsw(chunk_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_docTable_search_vector on {{table_name}}
    USING GIN (search_vector);