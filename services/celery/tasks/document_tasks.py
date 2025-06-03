import logging
import time
from typing import Dict, Any

from core.celery_app import celery_app
from .base_tasks import BaseDocumentTask
from services.document.stage_processors import (
    ExtractionProcessor, ChunkingProcessor, EmbeddingProcessor,
    TreeBuildingProcessor, VectorStorageProcessor
)

logger = logging.getLogger(__name__)

@celery_app.task(base=BaseDocumentTask, bind=True, queue='document_processing')
def extract_document_task(self, document_id: str, case_id: str, user_id: str, stage: str, context: Dict[str, Any]):
    """
    Celery task for document content extraction.
    """
    try:
        self.update_progress(5, "Initializing extraction task")
        
        # Check for cancellation
        self.check_for_cancellation()
        
        # Get state manager
        state_manager = self.get_state_manager(document_id, case_id, user_id)
        
        # Initialize processor
        from core.config import get_config
        config = get_config()
        processor = ExtractionProcessor(config)
        
        self.update_progress(10, "Starting content extraction")
        
        # Add task context with fresh storage adapter
        enhanced_context = context.copy()
        enhanced_context.update({
            'celery_task_id': self.request.id,
            'task_instance': self,
            'storage_adapter': self.get_storage_adapter()  # Fresh storage adapter
        })
        
        # Check cancellation before execution
        self.check_for_cancellation()
        
        # Execute extraction
        result = processor.execute(state_manager, enhanced_context)
        
        if result["status"] == "success":
            self.update_progress(100, "Extraction completed successfully")
            logger.info(f"Extraction task completed for document {document_id}")
            return result
        else:
            logger.error(f"Extraction task failed for document {document_id}: {result.get('error')}")
            raise Exception(result.get('error', 'Extraction failed'))
            
    except Exception as e:
        logger.error(f"Extraction task error for document {document_id}: {str(e)}")
        raise

@celery_app.task(base=BaseDocumentTask, bind=True, queue='document_processing')
def chunk_document_task(self, document_id: str, case_id: str, user_id: str, stage: str, context: Dict[str, Any]):
    """
    Celery task for document chunking.
    """
    try:
        self.update_progress(5, "Initializing chunking task")
        
        # Check for cancellation
        self.check_for_cancellation()
        
        # Get state manager
        state_manager = self.get_state_manager(document_id, case_id, user_id)
        
        # Initialize processor
        from core.config import get_config
        config = get_config()
        processor = ChunkingProcessor(config)
        
        self.update_progress(10, "Starting document chunking")
        
        # Add task context with fresh storage adapter
        enhanced_context = context.copy()
        enhanced_context.update({
            'celery_task_id': self.request.id,
            'task_instance': self,
            'storage_adapter': self.get_storage_adapter()  # Fresh storage adapter
        })
        
        # Check cancellation before execution
        self.check_for_cancellation()
        
        # Execute chunking
        result = processor.execute(state_manager, enhanced_context)
        
        if result["status"] == "success":
            self.update_progress(100, "Chunking completed successfully")
            logger.info(f"Chunking task completed for document {document_id}")
            return result
        else:
            logger.error(f"Chunking task failed for document {document_id}: {result.get('error')}")
            raise Exception(result.get('error', 'Chunking failed'))
            
    except Exception as e:
        logger.error(f"Chunking task error for document {document_id}: {str(e)}")
        raise

@celery_app.task(base=BaseDocumentTask, bind=True, queue='document_processing')
def embed_document_task(self, document_id: str, case_id: str, user_id: str, stage: str, context: Dict[str, Any]):
    """
    Celery task for embedding generation.
    """
    try:
        self.update_progress(5, "Initializing embedding task")
        
        # Check for cancellation
        self.check_for_cancellation()
        
        # Get state manager
        state_manager = self.get_state_manager(document_id, case_id, user_id)
        
        # Initialize processor
        from core.config import get_config
        config = get_config()
        processor = EmbeddingProcessor(config)
        
        self.update_progress(10, "Starting embedding generation")
        
        # Add task context with fresh storage adapter
        enhanced_context = context.copy()
        enhanced_context.update({
            'celery_task_id': self.request.id,
            'task_instance': self,
            'storage_adapter': self.get_storage_adapter()  # Fresh storage adapter
        })
        
        # Check cancellation before execution
        self.check_for_cancellation()
        
        # Execute embedding generation
        result = processor.execute(state_manager, enhanced_context)
        
        if result["status"] == "success":
            self.update_progress(100, "Embedding generation completed successfully")
            logger.info(f"Embedding task completed for document {document_id}")
            return result
        else:
            logger.error(f"Embedding task failed for document {document_id}: {result.get('error')}")
            raise Exception(result.get('error', 'Embedding generation failed'))
            
    except Exception as e:
        logger.error(f"Embedding task error for document {document_id}: {str(e)}")
        raise

@celery_app.task(base=BaseDocumentTask, bind=True, queue='document_processing')
def build_tree_task(self, document_id: str, case_id: str, user_id: str, stage: str, context: Dict[str, Any]):
    """
    Celery task for RAPTOR tree building.
    """
    try:
        self.update_progress(5, "Initializing tree building task")
        
        # Check for cancellation
        self.check_for_cancellation()
        
        # Get state manager
        state_manager = self.get_state_manager(document_id, case_id, user_id)
        
        # Initialize processor
        from core.config import get_config
        config = get_config()
        processor = TreeBuildingProcessor(config)
        
        self.update_progress(10, "Starting RAPTOR tree building")
        
        # Add task context with fresh storage adapter
        enhanced_context = context.copy()
        enhanced_context.update({
            'celery_task_id': self.request.id,
            'task_instance': self,
            'storage_adapter': self.get_storage_adapter()  # Fresh storage adapter
        })
        
        # Check cancellation before execution
        self.check_for_cancellation()
        
        # Execute tree building
        result = processor.execute(state_manager, enhanced_context)
        
        if result["status"] == "success":
            self.update_progress(100, "Tree building completed successfully")
            logger.info(f"Tree building task completed for document {document_id}")
            return result
        else:
            logger.error(f"Tree building task failed for document {document_id}: {result.get('error')}")
            raise Exception(result.get('error', 'Tree building failed'))
            
    except Exception as e:
        logger.error(f"Tree building task error for document {document_id}: {str(e)}")
        raise

@celery_app.task(base=BaseDocumentTask, bind=True, queue='document_processing')
def store_vectors_task(self, document_id: str, case_id: str, user_id: str, stage: str, context: Dict[str, Any]):
    """
    Celery task for vector storage.
    """
    try:
        self.update_progress(5, "Initializing vector storage task")
        
        # Check for cancellation
        self.check_for_cancellation()
        
        # Get state manager
        state_manager = self.get_state_manager(document_id, case_id, user_id)
        
        # Initialize processor
        from core.config import get_config
        config = get_config()
        processor = VectorStorageProcessor(config)
        
        self.update_progress(10, "Starting vector storage")
        
        # Add task context with fresh storage adapter
        enhanced_context = context.copy()
        enhanced_context.update({
            'celery_task_id': self.request.id,
            'task_instance': self,
            'storage_adapter': self.get_storage_adapter()  # Fresh storage adapter
        })
        
        # Check cancellation before execution
        self.check_for_cancellation()
        
        # Execute vector storage
        result = processor.execute(state_manager, enhanced_context)
        
        if result["status"] == "success":
            self.update_progress(100, "Vector storage completed successfully")
            logger.info(f"Vector storage task completed for document {document_id}")
            return result
        else:
            logger.error(f"Vector storage task failed for document {document_id}: {result.get('error')}")
            raise Exception(result.get('error', 'Vector storage failed'))
            
    except Exception as e:
        logger.error(f"Vector storage task error for document {document_id}: {str(e)}")
        raise

# Test task for basic Celery functionality
@celery_app.task(queue='document_processing')
def test_document_task(message: str):
    """Simple test task for document processing queue"""
    logger.info(f"Test document task received: {message}")
    time.sleep(2)  # Simulate work
    return f"Document task completed: {message}"