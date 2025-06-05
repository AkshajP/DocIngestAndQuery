import os
import logging
from celery import Celery
from celery.signals import setup_logging
from core.config import get_config

logger = logging.getLogger(__name__)

# Initialize config
config = get_config()

# Create Celery app instance
celery_app = Celery(
    'docrag',
    broker=config.celery.broker_url,
    backend=config.celery.result_backend,
    include=[
        'services.celery.tasks.document_tasks',  # Will be created in later steps
    ]
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task routing
    task_routes={
        'services.celery.tasks.document_tasks.*': {'queue': 'document_processing'},
        'services.celery.tasks.document_tasks.extract_document_task': {'queue': 'document_processing'},
        'services.celery.tasks.document_tasks.chunk_document_task': {'queue': 'document_processing'},
        'services.celery.tasks.document_tasks.embed_document_task': {'queue': 'document_processing'},
        'services.celery.tasks.document_tasks.build_tree_task': {'queue': 'document_processing'},
        'services.celery.tasks.document_tasks.store_vectors_task': {'queue': 'document_processing'},
        'services.celery.tasks.document_tasks.start_document_processing_chain': {'queue': 'document_processing'},
    },
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Task execution settings
    task_always_eager=False,  # Set to True for synchronous testing
    task_eager_propagates=True,
    task_ignore_result=False,
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Setup logging for Celery
@setup_logging.connect
def config_loggers(*args, **kwargs):
    from logging.config import dictConfig
    dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
            },
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console'],
        },
        'loggers': {
            'celery': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False,
            },
        },
    })

# Simple test task for initial verification
@celery_app.task(name='core.celery_app.ping')
def ping():
    """Simple ping task for testing Celery setup"""
    logger.info("Ping task executed successfully")
    return "pong"

# Task to verify database connectivity
@celery_app.task(name='core.celery_app.test_db_connection')
def test_db_connection():
    """Test database connectivity from worker"""
    try:
        from db.document_store.repository import DocumentMetadataRepository
        repo = DocumentMetadataRepository()
        stats = repo.get_statistics()
        logger.info(f"Database connection successful, found {stats['total_documents']} documents")
        return {"status": "success", "total_documents": stats['total_documents']}
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise

# Task to verify vector database connectivity  
@celery_app.task(name='core.celery_app.test_vector_db_connection')
def test_vector_db_connection():
    """Test vector database connectivity from worker"""
    try:
        from db.vector_store.adapter import VectorStoreAdapter
        vector_store = VectorStoreAdapter()
        # Simple test - this will fail gracefully if vector DB is not ready
        collections = vector_store.list_collections()
        logger.info(f"Vector database connection successful, found {len(collections)} collections")
        return {"status": "success", "collections_count": len(collections)}
    except Exception as e:
        logger.error(f"Vector database connection failed: {str(e)}")
        raise

# Auto-discover tasks
celery_app.autodiscover_tasks([
    'services.celery.tasks',
])

# Health check for Celery workers
@celery_app.task(name='core.celery_app.health_check')
def health_check():
    """Health check task for monitoring worker status"""
    import time
    import platform
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "worker_host": platform.node(),
    }

logger.info("Celery app initialized successfully")