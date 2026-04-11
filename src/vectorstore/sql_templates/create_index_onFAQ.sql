CREATE INDEX IF NOT EXISTS idx_hnsw_FAQ on {{table_name}}
    USING hnsw (similar_question_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_faq_search_vector on {{table_name}}
    USING GIN (search_vector);