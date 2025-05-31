#!/usr/bin/env python3
"""
Database initialization script to create the required PostgreSQL tables.
"""

import sys
import os
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_init")

def create_database_if_not_exists(config):
    """
    Create the database if it doesn't exist
    """
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            database="postgres"  # Connect to default database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cursor:
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config.dbname,))
            exists = cursor.fetchone()
            
            if not exists:
                logger.info(f"Creating database {config.dbname}")
                cursor.execute(f"CREATE DATABASE {config.dbname}")
                logger.info(f"Database {config.dbname} created successfully")
            else:
                logger.info(f"Database {config.dbname} already exists")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
        return False

def create_tables(conn):
    """
    Create the required tables if they don't exist
    """
    try:
        with conn.cursor() as cursor:
            # Create chats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id VARCHAR(50) PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    user_id VARCHAR(50) NOT NULL,
                    case_id VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    state VARCHAR(20) NOT NULL,
                    settings JSONB
                )
            """)
            
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id VARCHAR(50) PRIMARY KEY,
                    chat_id VARCHAR(50) NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                    user_id VARCHAR(50) NOT NULL,
                    case_id VARCHAR(50) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    sources JSONB,
                    metadata JSONB,
                    status VARCHAR(20) NOT NULL,
                    token_count INTEGER,
                    model_used VARCHAR(50),
                    error_details JSONB,
                    response_time INTEGER
                )
            """)
            
            # Create chat_documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_documents (
                    id SERIAL PRIMARY KEY,
                    chat_id VARCHAR(50) NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                    document_id VARCHAR(50) NOT NULL,
                    added_at TIMESTAMP NOT NULL,
                    UNIQUE(chat_id, document_id)
                )
            """)
            
            # Create feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id VARCHAR(50) PRIMARY KEY,
                    message_id VARCHAR(50) NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
                    user_id VARCHAR(50) NOT NULL,
                    rating INTEGER,
                    comment TEXT,
                    created_at TIMESTAMP NOT NULL,
                    feedback_type VARCHAR(50),
                    metadata JSONB
                )
            """)
            
            # Create user_case_access table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_case_access (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL,
                    case_id VARCHAR(50) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    UNIQUE(user_id, case_id)
                )
            """)
            
            # Create indices for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_message_id ON feedback(message_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_documents_chat_id ON chat_documents(chat_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_case_access_user_id ON user_case_access(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_case_access_case_id ON user_case_access(case_id)")
            
            # Create indexing for case filtering
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chats_case_id ON chats(case_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_case_id ON messages(case_id)")
            
            # Ensure timestamps are indexed for sorting
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)")
            
            conn.commit()
            logger.info("Tables created successfully")
            
            return True
            
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        conn.rollback()
        return False

def create_task_tables(conn):
    """
    Create task-related tables for Celery integration.
    """
    try:
        with conn.cursor() as cursor:
            # Create document_tasks table for Celery task management
            cursor.execute("""
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
                )
            """)
            
            # Create indices for efficient querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_tasks_document_id ON document_tasks(document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_tasks_case_id ON document_tasks(case_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_tasks_status ON document_tasks(task_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_tasks_celery_id ON document_tasks(celery_task_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_tasks_stage ON document_tasks(processing_stage)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_tasks_updated_at ON document_tasks(updated_at)")
            
            # Auto-update timestamp trigger function (if not exists)
            cursor.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ language 'plpgsql'
            """)
            
            # Drop existing trigger if it exists and create new one
            cursor.execute("DROP TRIGGER IF EXISTS update_document_tasks_updated_at ON document_tasks")
            cursor.execute("""
                CREATE TRIGGER update_document_tasks_updated_at
                    BEFORE UPDATE ON document_tasks
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column()
            """)
            
            conn.commit()
            logger.info("Task tables created successfully")
            return True
            
    except Exception as e:
        logger.error(f"Error creating task tables: {str(e)}")
        conn.rollback()
        return False

def test_connection():
    """Test database connection and create tables if needed"""
    config = get_config().database
    
    # First ensure database exists
    if not create_database_if_not_exists(config):
        logger.error("Failed to create/verify database")
        return False
    
    try:
        # Now connect to the database and create tables
        conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            dbname=config.dbname,
            user=config.user,
            password=config.password
        )
        
        logger.info("Successfully connected to database")
        
        # Create existing tables
        if create_tables(conn):
            logger.info("Chat tables created successfully")
        else:
            logger.error("Failed to create chat tables")
            return False
            
        # Create task tables
        if create_task_tables(conn):
            logger.info("Task tables created successfully")
        else:
            logger.error("Failed to create task tables")
            return False
            
        logger.info("Database initialization completed successfully")
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        return False

def add_sample_data():
    """Add some sample data for testing"""
    config = get_config().database
    
    try:
        conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            dbname=config.dbname,
            user=config.user,
            password=config.password
        )
        
        with conn.cursor() as cursor:
            # Check if we already have sample data
            cursor.execute("SELECT COUNT(*) FROM chats")
            count = cursor.fetchone()[0]
            
            if count > 0:
                logger.info("Sample data already exists, skipping")
                conn.close()
                return True
            
            # Add a sample case
            case_id = "case_demo123"
            user_id = "user_test123"
            
            # Add user-case access
            cursor.execute("""
                INSERT INTO user_case_access (user_id, case_id, role, created_at)
                VALUES (%s, %s, %s, NOW())
            """, (user_id, case_id, "owner"))
            
            # Add a sample chat
            cursor.execute("""
                INSERT INTO chats (chat_id, title, user_id, case_id, created_at, updated_at, state, settings)
                VALUES (%s, %s, %s, %s, NOW(), NOW(), %s, %s)
            """, ("chat_sample123", "Sample Chat", user_id, case_id, "open", '{"use_tree": false, "top_k": 5}'))
            
            # Add some sample messages
            cursor.execute("""
                INSERT INTO messages (message_id, chat_id, user_id, case_id, role, content, created_at, status)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s)
            """, ("msg_user1", "chat_sample123", user_id, case_id, "user", "What is RAG?", "completed"))
            
            cursor.execute("""
                INSERT INTO messages (message_id, chat_id, user_id, case_id, role, content, created_at, status, 
                                      token_count, model_used)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s)
            """, ("msg_assistant1", "chat_sample123", "system", case_id, "assistant", 
                 "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant documents with text generation. It allows language models to access and use external knowledge to generate more accurate and informed responses.", 
                 "completed", 245, "llama3.2"))
            
            conn.commit()
            logger.info("Sample data added successfully")
            
            return True
            
    except Exception as e:
        logger.error(f"Error adding sample data: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting database initialization")
    
    # Test connection and create tables
    if test_connection():
        logger.info("Database connection successful")
        
        # Add sample data if requested
        if len(sys.argv) > 1 and sys.argv[1] == "--with-sample-data":
            add_sample_data()
    else:
        logger.error("Database initialization failed")
        sys.exit(1)
    
    logger.info("Database initialization completed")