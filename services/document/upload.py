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
    Upload and process a document through the complete Celery pipeline.
    
    This function now delegates to Celery tasks instead of processing synchronously.
    Maintains backward compatibility with the existing API while using async processing.
    
    Args:
        file_path: Path to the document file
        document_id: Optional custom ID (generated if not provided)
        case_id: Case ID for document grouping (required)
        metadata: Optional metadata about the document
        config: Optional config override
        force_restart: If True, restart processing from the beginning
        
    Returns:
        Dictionary with processing status and task information
    """
    logger.info(f"Starting document upload for file: {file_path}")
    
    # Use provided config or get default
    config = config or get_config()
    
    try:
        # Initialize the persistent upload service
        from services.document.persistent_upload_service import PersistentUploadService
        upload_service = PersistentUploadService(config)
        
        # Initialize document and get basic setup
        result = upload_service._initialize_document_processing(
            document_id, case_id, file_path, metadata, force_restart
        )
        
        if result["status"] == "error":
            return result
        
        document_id = result.get("document_id") or document_id
        
        # Generate document ID if not provided
        if not document_id:
            timestamp = int(time.time())
            base_name = os.path.basename(file_path)
            safe_name = ''.join(c for c in base_name.split('.')[0].replace(' ', '_') 
                              if c.isalnum() or c == '_')
            document_id = f"doc_{timestamp}_{safe_name}"
        
        # Start Celery processing chain
        celery_result = upload_document_with_celery(
            file_path=file_path,
            document_id=document_id,
            case_id=case_id,
            metadata=metadata,
            force_restart=force_restart
        )
        
        return celery_result
        
    except Exception as e:
        logger.error(f"Error in upload_document: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "processing_time": 0
        }

def upload_document_with_celery(
    file_path: str,
    document_id: str,
    case_id: str = "default",
    metadata: Optional[Dict[str, Any]] = None,
    force_restart: bool = False
) -> Dict[str, Any]:
    """Upload document with ROBUST file storage handling."""
    start_time = time.time()
    config = get_config()
    
    logger.info(f"Starting Celery upload for document {document_id}")
    
    try:
        # STEP 1: Initialize repositories and services FIRST
        doc_repository = DocumentMetadataRepository()
        
        # STEP 2: Ensure base storage directory exists with detailed checking
        base_storage_dir = config.storage.storage_dir
        logger.info(f"Base storage directory: {base_storage_dir}")
        
        # Convert to absolute path to avoid relative path issues
        base_storage_dir = os.path.abspath(base_storage_dir)
        logger.info(f"Absolute base storage directory: {base_storage_dir}")
        
        if not os.path.exists(base_storage_dir):
            try:
                os.makedirs(base_storage_dir, mode=0o755, exist_ok=True)
                logger.info(f"Created base storage directory: {base_storage_dir}")
            except Exception as e:
                logger.error(f"Failed to create base storage directory: {str(e)}")
                return {
                    "status": "error",
                    "error": f"Storage directory creation failed: {str(e)}",
                    "document_id": document_id
                }
        
        # Verify base directory is writable
        if not os.access(base_storage_dir, os.W_OK):
            return {
                "status": "error",
                "error": f"Base storage directory not writable: {base_storage_dir}",
                "document_id": document_id
            }
        
        # STEP 3: Verify and resolve source file path
        logger.info(f"Source file path: {file_path}")
        
        # Convert to absolute path
        abs_file_path = os.path.abspath(file_path)
        logger.info(f"Absolute source file path: {abs_file_path}")
        
        if not os.path.exists(abs_file_path):
            return {
                "status": "error",
                "error": f"Source file not found: {abs_file_path} (original: {file_path})",
                "document_id": document_id
            }
        
        if not os.access(abs_file_path, os.R_OK):
            return {
                "status": "error", 
                "error": f"Source file not readable: {abs_file_path}",
                "document_id": document_id
            }
        
        # Get file size for verification
        file_size = os.path.getsize(abs_file_path)
        logger.info(f"Source file size: {file_size} bytes")
        
        if file_size == 0:
            return {
                "status": "error",
                "error": f"Source file is empty: {abs_file_path}",
                "document_id": document_id
            }
        
        # STEP 4: Check for existing processing
        existing_doc = doc_repository.get_document(document_id)
        if existing_doc and existing_doc.get("status") == "processing" and not force_restart:
            return {
                "status": "error",
                "error": "Document is already being processed",
                "document_id": document_id,
                "existing_status": existing_doc.get("status")
            }
        
        # STEP 5: Create document directory structure
        doc_dir = os.path.join(base_storage_dir, document_id)
        doc_dir = os.path.abspath(doc_dir)  # Ensure absolute path
        
        logger.info(f"Document directory: {doc_dir}")
        
        # Create document directory
        try:
            os.makedirs(doc_dir, mode=0o755, exist_ok=True)
            logger.info(f"Created/verified document directory: {doc_dir}")
        except Exception as e:
            logger.error(f"Failed to create document directory: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to create document directory {doc_dir}: {str(e)}",
                "document_id": document_id
            }
        
        # Verify document directory is writable
        if not os.access(doc_dir, os.W_OK):
            return {
                "status": "error",
                "error": f"Document directory not writable: {doc_dir}",
                "document_id": document_id
            }
        
        # STEP 6: Store the file with robust error handling
        target_filename = "original.pdf"
        target_file_path = os.path.join(doc_dir, target_filename)
        target_file_path = os.path.abspath(target_file_path)
        
        logger.info(f"Target file path: {target_file_path}")
        
        # Read source file
        try:
            logger.info(f"Reading source file: {abs_file_path}")
            with open(abs_file_path, 'rb') as f:
                file_content = f.read()
            
            logger.info(f"Read {len(file_content)} bytes from source file")
            
            if len(file_content) == 0:
                return {
                    "status": "error",
                    "error": f"Source file is empty when read: {abs_file_path}",
                    "document_id": document_id
                }
                
        except Exception as e:
            logger.error(f"Error reading source file {abs_file_path}: {str(e)}")
            return {
                "status": "error",
                "error": f"Error reading source file: {str(e)}",
                "document_id": document_id
            }
        
        # Write to target location
        try:
            logger.info(f"Writing file to: {target_file_path}")
            with open(target_file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Wrote {len(file_content)} bytes to target file")
            
        except Exception as e:
            logger.error(f"Error writing to target file {target_file_path}: {str(e)}")
            return {
                "status": "error",
                "error": f"Error writing to target file: {str(e)}",
                "document_id": document_id
            }
        
        # STEP 7: Verify file was stored correctly
        if not os.path.exists(target_file_path):
            return {
                "status": "error",
                "error": f"File verification failed - file does not exist: {target_file_path}",
                "document_id": document_id
            }
        
        stored_size = os.path.getsize(target_file_path)
        if stored_size != file_size:
            return {
                "status": "error",
                "error": f"File size mismatch - original: {file_size}, stored: {stored_size}",
                "document_id": document_id
            }
        
        if not os.access(target_file_path, os.R_OK):
            return {
                "status": "error",
                "error": f"Stored file not readable: {target_file_path}",
                "document_id": document_id
            }
        
        logger.info(f"File storage verification successful: {target_file_path} ({stored_size} bytes)")
        
        # STEP 8: Create document metadata and ADD TO REPOSITORY
        doc_metadata = {
            "document_id": document_id,
            "case_id": case_id,
            "original_filename": os.path.basename(abs_file_path),
            "original_file_path": abs_file_path,  # Store absolute path
            "stored_file_path": target_file_path,  # Store absolute path
            "doc_dir": doc_dir,
            "file_type": os.path.splitext(abs_file_path)[1].lower()[1:],
            "file_size": stored_size,
            "processing_start_time": datetime.now().isoformat(),
            "status": "initializing",
            "user_metadata": metadata or {},
            "can_pause": True,
            "can_resume": False,
            "can_cancel": True
        }
        
        # ADD/UPDATE document in repository BEFORE anything else
        if existing_doc:
            logger.info(f"Updating existing document {document_id} in repository")
            success = doc_repository.update_document(document_id, doc_metadata)
        else:
            logger.info(f"Adding new document {document_id} to repository")
            success = doc_repository.add_document(doc_metadata)
        
        if not success:
            return {
                "status": "error",
                "error": "Failed to register document in repository",
                "document_id": document_id
            }
        
        # VERIFY document was added
        verification_doc = doc_repository.get_document(document_id)
        if not verification_doc:
            return {
                "status": "error",
                "error": "Document registration verification failed",
                "document_id": document_id
            }
        
        # Verify stored file path in metadata
        verified_path = verification_doc.get("stored_file_path")
        if not verified_path or not os.path.exists(verified_path):
            return {
                "status": "error",
                "error": f"Document metadata has invalid stored_file_path: {verified_path}",
                "document_id": document_id
            }
        
        logger.info(f"Document {document_id} successfully registered with file at {verified_path}")
        
        # STEP 9: Initialize processing state manager and mark upload complete
        from services.document.processing_state_manager import ProcessingStateManager, ProcessingStage
        
        # Initialize storage adapter for state manager
        from services.document.persistent_upload_service import PersistentUploadService
        upload_service = PersistentUploadService(config)
        
        state_manager = ProcessingStateManager(
            document_id=document_id,
            storage_adapter=upload_service.storage_adapter,
            doc_dir=doc_dir,
            doc_repository=doc_repository,
            task_state_manager=None  # Will be set later
        )
        
        # Mark upload as complete
        if not state_manager.mark_stage_complete(ProcessingStage.UPLOAD.value):
            logger.warning(f"Failed to mark upload stage complete for {document_id}")
        
        # STEP 10: Create task state management entry
        task_state_manager = None
        try:
            from services.celery.task_state_manager import TaskStateManager
            task_state_manager = TaskStateManager()
            
            # Create workflow task entry
            task_db_id = task_state_manager.create_workflow_task(
                document_id=document_id,
                initial_stage="extraction",
                celery_task_id=None,  # Will be updated when Celery task starts
                max_retries=3
            )
            logger.info(f"Created workflow task {task_db_id} for document {document_id}")
            
        except Exception as e:
            logger.warning(f"Could not initialize task state manager: {str(e)}")
        
        # STEP 11: Update document status to "processing" before starting Celery
        doc_repository.update_document(document_id, {
            "status": "processing",
            "processing_mode": "celery_async"
        })
        
        # STEP 12: Start Celery processing chain
        from services.celery.tasks.document_tasks import start_document_processing_chain
        
        logger.info(f"Starting Celery processing chain for {document_id}")
        chain_result = start_document_processing_chain.delay(document_id)
        
        # STEP 13: Update with Celery task information
        final_metadata = {
            "celery_task_id": chain_result.id,
            "celery_task_started_at": datetime.now().isoformat()
        }
        
        doc_repository.update_document(document_id, final_metadata)
        
        # Update task state manager with Celery task ID
        if task_state_manager:
            try:
                task_state_manager.start_task(
                    document_id=document_id,
                    celery_task_id=chain_result.id,
                    worker_info={"initiated_from": "upload_service"}
                )
            except Exception as e:
                logger.warning(f"Failed to update task state manager: {str(e)}")
        
        processing_time = time.time() - start_time
        
        logger.info(f"Successfully started Celery processing for {document_id} with task {chain_result.id}")
        
        return {
            "status": "processing",
            "document_id": document_id,
            "case_id": case_id,
            "celery_task_id": chain_result.id,
            "message": "Document processing started asynchronously",
            "processing_time": processing_time,
            "stored_file_path": target_file_path,
            "doc_dir": doc_dir,
            "file_size": stored_size,
            "task_control": {
                "can_pause": True,
                "can_resume": False,
                "can_cancel": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error in upload_document_with_celery for {document_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to clean up any partial files
        try:
            doc_dir = os.path.join(config.storage.storage_dir, document_id)
            if os.path.exists(doc_dir):
                import shutil
                shutil.rmtree(doc_dir)
                logger.info(f"Cleaned up partial document directory: {doc_dir}")
        except:
            pass
        
        return {
            "status": "error",
            "document_id": document_id,
            "error": str(e),
            "processing_time": time.time() - start_time
        }


def retry_document_processing(
    document_id: str,
    from_stage: Optional[str] = None,
    case_id: Optional[str] = None,
    config = None
) -> Dict[str, Any]:
    """
    Retry document processing using Celery tasks.
    
    Args:
        document_id: Document ID to retry
        from_stage: Optional stage to retry from (if None, uses current stage)
        case_id: Optional case ID for verification
        config: Optional config override
        
    Returns:
        Dictionary with retry status and information
    """
    logger.info(f"Retrying document processing for {document_id} from stage {from_stage}")
    
    config = config or get_config()
    
    try:
        doc_repository = DocumentMetadataRepository()
        document = doc_repository.get_document(document_id)
        
        if not document:
            return {"status": "error", "error": "Document not found"}
        
        # Verify case_id if provided
        if case_id and document.get("case_id") != case_id:
            return {"status": "error", "error": "Document does not belong to specified case"}
        
        # Determine retry strategy based on from_stage
        if from_stage:
            # Specific stage retry - restart the chain from that stage
            if from_stage == "extraction":
                from services.celery.tasks.document_tasks import extract_document_task
                task_result = extract_document_task.delay(document_id)
            elif from_stage == "chunking":
                from services.celery.tasks.document_tasks import chunk_document_task
                task_result = chunk_document_task.delay(document_id)
            elif from_stage == "embedding":
                from services.celery.tasks.document_tasks import embed_document_task
                task_result = embed_document_task.delay(document_id)
            elif from_stage == "tree_building":
                from services.celery.tasks.document_tasks import build_tree_task
                task_result = build_tree_task.delay(document_id)
            elif from_stage == "vector_storage":
                from services.celery.tasks.document_tasks import store_vectors_task
                task_result = store_vectors_task.delay(document_id)
            else:
                return {"status": "error", "error": f"Invalid stage: {from_stage}"}
        else:
            # Full retry - restart the entire chain
            from services.celery.tasks.document_tasks import start_document_processing_chain
            task_result = start_document_processing_chain.delay(document_id)
        
        # Update document metadata
        retry_metadata = {
            "status": "processing",
            "celery_task_id": task_result.id,
            "retry_time": datetime.now().isoformat(),
            "retry_count": document.get("retry_count", 0) + 1,
            "retry_from_stage": from_stage,
            "processing_mode": "celery_async"
        }
        
        doc_repository.update_document(document_id, retry_metadata)
        
        logger.info(f"Started retry for document {document_id} with task {task_result.id}")
        
        return {
            "status": "processing",
            "document_id": document_id,
            "celery_task_id": task_result.id,
            "retry_from_stage": from_stage,
            "retry_count": retry_metadata["retry_count"],
            "message": "Document retry started"
        }
        
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
    Get detailed processing status for a document with Celery task information.
    
    Args:
        document_id: Document ID
        config: Optional config override
        
    Returns:
        Dictionary with processing status information
    """
    config = config or get_config()
    
    try:
        from services.document.persistent_upload_service import PersistentUploadService
        upload_service = PersistentUploadService(config)
        
        # Get comprehensive status including Celery task info
        result = upload_service.get_document_processing_status(document_id)
        
        # Add Celery-specific information
        if result["status"] == "success":
            doc_metadata = result["document_metadata"]
            celery_task_id = doc_metadata.get("celery_task_id")
            
            if celery_task_id:
                # Try to get Celery task status
                try:
                    from celery.result import AsyncResult
                    task = AsyncResult(celery_task_id)
                    
                    result["celery_status"] = {
                        "task_id": celery_task_id,
                        "state": task.state,
                        "info": task.info if hasattr(task, 'info') else None,
                        "ready": task.ready(),
                        "successful": task.successful() if task.ready() else None,
                        "failed": task.failed() if task.ready() else None
                    }
                except Exception as e:
                    result["celery_status"] = {
                        "task_id": celery_task_id,
                        "error": f"Could not get Celery status: {str(e)}"
                    }
        
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
    """Handle document processing failure."""
    failure_metadata = {
        "status": "failed",
        "failure_stage": stage,
        "error_message": error_message,
        "failure_time": datetime.now().isoformat()
    }
    
    doc_repository.update_document(document_id, failure_metadata)
    
    logger.error(f"Document {document_id} processing failed at {stage}: {error_message}")
    
    return {
        "status": "error",
        "document_id": document_id,
        "stage": stage,
        "error": error_message
    }