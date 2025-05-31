-- Single table to track all Celery tasks for document processing
CREATE TABLE IF NOT EXISTS document_tasks (
    id SERIAL PRIMARY KEY,
    
    -- Core identifiers
    document_id VARCHAR(50) NOT NULL,
    case_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    processing_stage VARCHAR(50) NOT NULL,
    celery_task_id VARCHAR(50) UNIQUE NOT NULL,
    
    -- Task information
    task_name VARCHAR(100) NOT NULL,
    task_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    
    -- Control flags
    can_pause BOOLEAN DEFAULT true,
    can_resume BOOLEAN DEFAULT false,
    can_cancel BOOLEAN DEFAULT true,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    
    -- Worker information
    worker_hostname VARCHAR(255),
    worker_pid INTEGER,
    
    -- Flexible data storage
    error_details TEXT,
    checkpoint_data JSONB,
    task_metadata JSONB DEFAULT '{}',
    
    -- Ensure one task per stage per document
    UNIQUE(document_id, processing_stage)
);

-- Indices for efficient querying
CREATE INDEX IF NOT EXISTS idx_document_tasks_document_id ON document_tasks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_tasks_case_id ON document_tasks(case_id);
CREATE INDEX IF NOT EXISTS idx_document_tasks_status ON document_tasks(task_status);
CREATE INDEX IF NOT EXISTS idx_document_tasks_celery_id ON document_tasks(celery_task_id);
CREATE INDEX IF NOT EXISTS idx_document_tasks_stage ON document_tasks(processing_stage);
CREATE INDEX IF NOT EXISTS idx_document_tasks_updated_at ON document_tasks(updated_at);

-- Auto-update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_document_tasks_updated_at ON document_tasks;

CREATE TRIGGER update_document_tasks_updated_at
    BEFORE UPDATE ON document_tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();