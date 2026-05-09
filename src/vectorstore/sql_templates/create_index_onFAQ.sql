CREATE INDEX IF NOT EXISTS idx_hnsw_FAQ on {{table_name}}
    USING hnsw (question_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);