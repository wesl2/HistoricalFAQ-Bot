CREATE TABLE IF NOT EXISTS {{table_name}} (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    similar_question TEXT NOT NULL,
    similar_question_vector vector({{vector_dim}}), 
    answer TEXT NOT NULL,
    search_vector tsvector 
        GENERATED ALWAYS AS (to_tsvector('simple', similar_question)) STORED,
    category VARCHAR(50),
    source_doc VARCHAR(255),
    confidence FLOAT DEFAULT 0.9,
    created_by VARCHAR(50) DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);