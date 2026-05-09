CREATE TABLE IF NOT EXISTS {{table_name}} (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    question_vector vector({{vector_dim}}),
    answer TEXT NOT NULL,
    category VARCHAR(50),
    source_doc VARCHAR(255),
    created_by VARCHAR(50) DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);