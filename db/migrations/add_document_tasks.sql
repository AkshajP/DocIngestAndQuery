CREATE TABLE IF NOT EXISTS document_tasks (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL,
    current_stage VARCHAR(50) NOT NULL DEFAULT 'upload',
    celery_task_id VARCHAR(255),
    task_status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    
    -- Control flags
    can_pause BOOLEAN DEFAULT TRUE,
    can_resume BOOLEAN DEFAULT FALSE,
    can_cancel BOOLEAN DEFAULT TRUE,
    
    -- Request flags
    pause_requested BOOLEAN DEFAULT FALSE,
    cancel_requested BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Worker information
    worker_id VARCHAR(255),
    worker_hostname VARCHAR(255),
    
    -- Progress tracking
    percent_complete INTEGER DEFAULT 0 CHECK (percent_complete >= 0 AND percent_complete <= 100),
    
    -- Checkpoint and metadata storage
    checkpoint_data JSONB DEFAULT '{}',
    stage_metadata JSONB DEFAULT '{}',
    error_details JSONB DEFAULT '{}',
    
    -- Retry information
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Foreign key constraint (adjust table name as needed)
    -- FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
    
    -- Ensure only one active task per document
    UNIQUE(document_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_document_tasks_document_id ON document_tasks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_tasks_status ON document_tasks(task_status);
CREATE INDEX IF NOT EXISTS idx_document_tasks_stage ON document_tasks(current_stage);
CREATE INDEX IF NOT EXISTS idx_document_tasks_celery_task_id ON document_tasks(celery_task_id);
CREATE INDEX IF NOT EXISTS idx_document_tasks_created_at ON document_tasks(created_at);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_document_tasks_updated_at
    BEFORE UPDATE ON document_tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create enum types for better type safety (optional)
DO $$ BEGIN
    CREATE TYPE task_status_enum AS ENUM (
        'PENDING', 'STARTED', 'SUCCESS', 'FAILURE', 
        'PAUSED', 'RESUMED', 'CANCELLED', 'RETRY'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE processing_stage_enum AS ENUM (
        'upload', 'extraction', 'chunking', 'embedding', 
        'tree_building', 'vector_storage', 'completed'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Optional: Add enum constraints (comment out if you prefer VARCHAR flexibility)
-- ALTER TABLE document_tasks 
-- ALTER COLUMN task_status TYPE task_status_enum USING task_status::task_status_enum;
-- 
-- ALTER TABLE document_tasks 
-- ALTER COLUMN current_stage TYPE processing_stage_enum USING current_stage::processing_stage_enum;

-- Insert comment for documentation
COMMENT ON TABLE document_tasks IS 'Tracks Celery task execution for document processing pipeline';
COMMENT ON COLUMN document_tasks.checkpoint_data IS 'JSON storage for task checkpoint information (resume data)';
COMMENT ON COLUMN document_tasks.stage_metadata IS 'JSON storage for stage-specific metadata and progress';
COMMENT ON COLUMN document_tasks.error_details IS 'JSON storage for error information and stack traces';