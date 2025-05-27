import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from core.config import get_config
from db.document_store.repository import DocumentMetadataRepository
from db.vector_store.adapter import VectorStoreAdapter
from services.document.storage import LocalStorageAdapter, S3StorageAdapter
from services.document.chunker import Chunker
from services.pdf.extractor import PDFExtractor
from services.ml.embeddings import EmbeddingService
from services.retrieval.raptor import Raptor
from services.document.persistent_upload_service import PersistentUploadService


logger = logging.getLogger(__name__)
def upload_document(
    file_path: str,
    document_id: Optional[str] = None,
    case_id: str = "default",
    metadata: Optional[Dict[str, Any]] = None,
    config = None,
    force_restart: bool = False
) -> Dict[str, Any]:
    """
    Upload and process a document through the complete pipeline with persistence.
    
    This function now uses the PersistentUploadService for stage-based processing
    with retry capabilities. It maintains backward compatibility with the existing API.
    
    Args:
        file_path: Path to the document file
        document_id: Optional custom ID (generated if not provided)
        case_id: Case ID for document grouping (required)
        metadata: Optional metadata about the document
        config: Optional config override
        force_restart: If True, restart processing from the beginning
        
    Returns:
        Dictionary with processing status and information
    """
    logger.info(f"Starting document upload for file: {file_path}")
    
    # Use provided config or get default
    config = config or get_config()
    
    try:
        # Initialize the persistent upload service
        upload_service = PersistentUploadService(config)
        
        # Process the document
        result = upload_service.upload_document(
            file_path=file_path,
            document_id=document_id,
            case_id=case_id,
            metadata=metadata,
            force_restart=force_restart
        )
        
        # Log the result
        if result["status"] == "success":
            logger.info(f"Document {result['document_id']} processed successfully in {result['processing_time']:.2f} seconds")
        else:
            logger.error(f"Document processing failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in upload_document: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "processing_time": 0
        }

def retry_document_processing(
    document_id: str,
    from_stage: Optional[str] = None,
    case_id: Optional[str] = None,
    config = None
) -> Dict[str, Any]:
    """
    Retry document processing from a specific stage.
    
    Args:
        document_id: Document ID to retry
        from_stage: Optional stage to retry from (if None, uses current stage)
        case_id: Optional case ID for verification
        config: Optional config override
        
    Returns:
        Dictionary with retry status and information
    """
    logger.info(f"Retrying document processing for {document_id} from stage {from_stage}")
    
    # Use provided config or get default
    config = config or get_config()
    
    try:
        # Initialize the persistent upload service
        upload_service = PersistentUploadService(config)
        
        # Retry the document processing
        result = upload_service.retry_document_processing(
            document_id=document_id,
            from_stage=from_stage,
            case_id=case_id
        )
        
        # Log the result
        if result["status"] == "success":
            logger.info(f"Document {document_id} retry completed successfully")
        else:
            logger.error(f"Document retry failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in retry_document_processing: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

def get_document_processing_status(
    document_id: str,
    config = None
) -> Dict[str, Any]:
    """
    Get detailed processing status for a document.
    
    Args:
        document_id: Document ID
        config: Optional config override
        
    Returns:
        Dictionary with processing status information
    """
    # Use provided config or get default
    config = config or get_config()
    
    try:
        # Initialize the persistent upload service
        upload_service = PersistentUploadService(config)
        
        # Get processing status
        result = upload_service.get_document_processing_status(document_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting processing status for {document_id}: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

# Legacy function names for backward compatibility
def _initialize_storage_adapter(config):
    """Initialize the appropriate storage adapter based on configuration"""
    # This function is now handled within PersistentUploadService
    # Keeping for backward compatibility if other code references it
    from services.document.storage import LocalStorageAdapter, S3StorageAdapter
    
    storage_type = config.storage.storage_type.lower()
    
    if storage_type == "s3":
        logger.info(f"Initializing S3 storage adapter with bucket {config.storage.s3_bucket}")
        return S3StorageAdapter(
            bucket_name=config.storage.s3_bucket,
            prefix=config.storage.s3_prefix,
            region=config.storage.aws_region
        )
    else:
        logger.info(f"Initializing local storage adapter with directory {config.storage.storage_dir}")
        return LocalStorageAdapter()

def _handle_processing_failure(
    doc_repository,
    document_id: str, 
    stage: str, 
    error_message: str
) -> Dict[str, Any]:
    """
    Handle document processing failure.
    
    This function is now handled within the stage processors and state manager,
    but keeping for backward compatibility.
    
    Args:
        doc_repository: Document metadata repository
        document_id: Document ID
        stage: Processing stage that failed
        error_message: Error message
        
    Returns:
        Error response dictionary
    """
    failure_metadata = {
        "status": "failed",
        "failure_stage": stage,
        "error_message": error_message,
        "failure_time": datetime.now().isoformat()
    }
    
    # Update document metadata
    doc_repository.update_document(document_id, failure_metadata)
    
    logger.error(f"Document {document_id} processing failed at {stage}: {error_message}")
    
    return {
        "status": "error",
        "document_id": document_id,
        "stage": stage,
        "error": error_message
    }