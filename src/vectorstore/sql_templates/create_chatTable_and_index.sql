CREATE TABLE IF NOT EXISTS {{table_name}} (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL,        -- 'human' 或 'ai'
    content TEXT NOT NULL,            -- 消息内容
    created_by VARCHAR(50) DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chat_history_session ON {{table_name}} (session_id);