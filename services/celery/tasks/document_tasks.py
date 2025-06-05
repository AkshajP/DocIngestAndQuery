# services/celery/tasks/document_tasks.py
import os
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from celery import Task
from celery.exceptions import Ignore

from core.celery_app import celery_app
from core.config import get_config
from services.celery.task_utils import (
    TaskCheckpointManager, check_task_control_signals, 
    ControlledProcessor, update_task_progress
)
from services.document.processing_state_manager import ProcessingStateManager, ProcessingStage
from services.document.storage import LocalStorageAdapter, S3StorageAdapter
from services.document.stage_processors import (
    ExtractionProcessor, ChunkingProcessor, EmbeddingProcessor,
    TreeBuildingProcessor, VectorStorageProcessor
)
from db.document_store.repository import DocumentMetadataRepository

logger = logging.getLogger(__name__)

class BaseDocumentTask(Task):
    """
    Base class for document processing tasks with pause/resume/cancel capabilities.
    Provides common functionality for state management and control signal handling.
    """
    
    def __init__(self):
        self.config = get_config()
        self.doc_repository = DocumentMetadataRepository()
        self.task_state_manager = None
        self.checkpoint_manager = None
        
        # Initialize task state manager if available
        try:
            from services.celery.task_state_manager import TaskStateManager
            self.task_state_manager = TaskStateManager()
            self.checkpoint_manager = TaskCheckpointManager(self.task_state_manager)
            logger.info("BaseDocumentTask initialized with task state manager")
        except ImportError:
            logger.warning("Task state manager not available in BaseDocumentTask")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task executes successfully"""
        document_id = args[0] if args else kwargs.get('document_id')
        if document_id and self.task_state_manager:
            try:
                # Don't mark as complete here - let the task itself handle completion
                # since we might be chaining to the next stage
                logger.info(f"Task {task_id} succeeded for document {document_id}")
            except Exception as e:
                logger.error(f"Error in on_success for {document_id}: {str(e)}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        document_id = args[0] if args else kwargs.get('document_id')
        if document_id and self.task_state_manager:
            try:
                self.task_state_manager.complete_task(
                    document_id=document_id,
                    success=False,
                    error_details={
                        "task_id": task_id,
                        "error": str(exc),
                        "traceback": str(einfo)
                    }
                )
                logger.error(f"Task {task_id} failed for document {document_id}: {str(exc)}")
            except Exception as e:
                logger.error(f"Error in on_failure for {document_id}: {str(e)}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        document_id = args[0] if args else kwargs.get('document_id')
        logger.warning(f"Task {task_id} retrying for document {document_id}: {str(exc)}")
    
    def _initialize_storage_adapter(self):
        """Initialize storage adapter based on config"""
        storage_type = self.config.storage.storage_type.lower()
        
        if storage_type == "s3":
            return S3StorageAdapter(
                bucket_name=self.config.storage.s3_bucket,
                prefix=self.config.storage.s3_prefix,
                region=self.config.storage.aws_region
            )
        else:
            return LocalStorageAdapter()
    
    def _get_processing_context(self, document_id: str) -> Dict[str, Any]:
        """Get processing context with ROBUST file path resolution"""
        
        # Get document from repository
        document = self.doc_repository.get_document(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found in repository")
        
        logger.info(f"Retrieved document {document_id} from repository")
        logger.debug(f"Document metadata: {document}")
        
        # Get all possible file paths
        stored_file_path = document.get("stored_file_path")
        original_file_path = document.get("original_file_path")
        doc_dir = document.get("doc_dir")
        
        logger.info(f"File paths from metadata - stored: {stored_file_path}, original: {original_file_path}, doc_dir: {doc_dir}")
        
        # If we don't have doc_dir, construct it
        if not doc_dir:
            doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
            doc_dir = os.path.abspath(doc_dir)
        else:
            doc_dir = os.path.abspath(doc_dir)
        
        # Resolve file path with multiple fallback strategies
        resolved_file_path = None
        path_attempts = []
        
        # Strategy 1: Use stored_file_path if it exists
        if stored_file_path:
            # Try as absolute path first
            abs_stored_path = os.path.abspath(stored_file_path)
            path_attempts.append(("stored_file_path (absolute)", abs_stored_path, os.path.exists(abs_stored_path)))
            
            if os.path.exists(abs_stored_path):
                resolved_file_path = abs_stored_path
                logger.info(f"Found file using stored_file_path (absolute): {resolved_file_path}")
            elif os.path.exists(stored_file_path):
                resolved_file_path = stored_file_path
                logger.info(f"Found file using stored_file_path (relative): {resolved_file_path}")
            else:
                path_attempts.append(("stored_file_path (relative)", stored_file_path, os.path.exists(stored_file_path)))
        
        # Strategy 2: Use original_file_path if stored path failed
        if not resolved_file_path and original_file_path:
            abs_original_path = os.path.abspath(original_file_path)
            path_attempts.append(("original_file_path (absolute)", abs_original_path, os.path.exists(abs_original_path)))
            
            if os.path.exists(abs_original_path):
                resolved_file_path = abs_original_path
                logger.info(f"Found file using original_file_path (absolute): {resolved_file_path}")
            elif os.path.exists(original_file_path):
                resolved_file_path = original_file_path
                logger.info(f"Found file using original_file_path (relative): {resolved_file_path}")
            else:
                path_attempts.append(("original_file_path (relative)", original_file_path, os.path.exists(original_file_path)))
        
        # Strategy 3: Look for common file names in doc_dir
        if not resolved_file_path and os.path.exists(doc_dir):
            common_names = ["original.pdf", "document.pdf", "file.pdf"]
            for name in common_names:
                candidate_path = os.path.join(doc_dir, name)
                path_attempts.append((f"doc_dir/{name}", candidate_path, os.path.exists(candidate_path)))
                
                if os.path.exists(candidate_path):
                    resolved_file_path = candidate_path
                    logger.info(f"Found file using common name strategy: {resolved_file_path}")
                    break
        
        # Strategy 4: List files in doc_dir and take first PDF
        if not resolved_file_path and os.path.exists(doc_dir):
            try:
                files_in_dir = os.listdir(doc_dir)
                pdf_files = [f for f in files_in_dir if f.lower().endswith('.pdf')]
                path_attempts.append((f"files in doc_dir", str(files_in_dir), len(pdf_files) > 0))
                
                if pdf_files:
                    resolved_file_path = os.path.join(doc_dir, pdf_files[0])
                    logger.info(f"Found file using directory scan: {resolved_file_path}")
            except Exception as e:
                logger.warning(f"Could not scan doc_dir {doc_dir}: {str(e)}")
                path_attempts.append(("doc_dir scan", doc_dir, False))
        
        # If no file found, provide detailed error
        if not resolved_file_path:
            attempts_summary = []
            for attempt_name, attempt_path, exists in path_attempts:
                attempts_summary.append(f"{attempt_name}: {attempt_path} (exists: {exists})")
            
            error_message = (
                f"No valid file path found for document {document_id}. "
                f"Checked paths: {'; '.join(attempts_summary)}"
            )
            
            # Additional debugging info
            logger.error(f"File resolution failed for document {document_id}")
            logger.error(f"Document metadata: {document}")
            logger.error(f"Config storage dir: {self.config.storage.storage_dir}")
            logger.error(f"Computed doc_dir: {doc_dir}")
            logger.error(f"Doc_dir exists: {os.path.exists(doc_dir)}")
            
            if os.path.exists(doc_dir):
                try:
                    dir_contents = os.listdir(doc_dir)
                    logger.error(f"Doc_dir contents: {dir_contents}")
                except:
                    logger.error("Could not list doc_dir contents")
            
            raise ValueError(error_message)
        
        # Verify final file is readable
        if not os.access(resolved_file_path, os.R_OK):
            raise ValueError(f"File is not readable: {resolved_file_path}")
        
        # Get file size for verification
        try:
            file_size = os.path.getsize(resolved_file_path)
            if file_size == 0:
                raise ValueError(f"File is empty: {resolved_file_path}")
            logger.info(f"Resolved file size: {file_size} bytes")
        except Exception as e:
            raise ValueError(f"Cannot access file info for {resolved_file_path}: {str(e)}")
        
        # Initialize storage adapter
        storage_adapter = self._initialize_storage_adapter()
        
        # Ensure document directory exists and is writable
        if not os.path.exists(doc_dir):
            logger.info(f"Creating missing doc_dir: {doc_dir}")
            if not storage_adapter.create_directory(doc_dir):
                raise ValueError(f"Failed to create document directory: {doc_dir}")
        
        if not os.access(doc_dir, os.W_OK):
            raise ValueError(f"Document directory not writable: {doc_dir}")
        
        # Create context with resolved paths
        context = {
            "document_id": document_id,
            "case_id": document.get("case_id", "default"),
            "file_path": resolved_file_path,
            "stored_file_path": resolved_file_path,  # Use resolved path
            "doc_dir": doc_dir,
            "storage_adapter": storage_adapter,
            "metadata": document.get("user_metadata", {}),
            "celery_task_id": self.request.id,
            "document_metadata": document,  # Include full document metadata
            "file_size": file_size
        }
        
        logger.info(f"Successfully created processing context for {document_id}")
        logger.debug(f"Context: {context}")
        
        return context
    
    def _get_state_manager(self, document_id: str, context: Dict[str, Any]) -> ProcessingStateManager:
        """Get initialized processing state manager with error handling"""
        try:
            return ProcessingStateManager(
                document_id=document_id,
                storage_adapter=context["storage_adapter"],
                doc_dir=context["doc_dir"],
                doc_repository=self.doc_repository,
                task_state_manager=self.task_state_manager
            )
        except Exception as e:
            logger.error(f"Failed to initialize ProcessingStateManager for {document_id}: {str(e)}")
            raise ValueError(f"ProcessingStateManager initialization failed: {str(e)}")
  
@celery_app.task(base=BaseDocumentTask, bind=True, name='services.celery.tasks.document_tasks.extract_document_task')
def extract_document_task(self, document_id: str):
    """Extract content from document with proper task state management"""
    logger.info(f"Starting document extraction for {document_id}")
    
    try:
        # IMMEDIATELY mark task as STARTED in task state manager
        if self.task_state_manager:
            try:
                worker_info = {
                    "worker_id": os.getenv("CELERY_WORKER_NAME", "unknown"),
                    "worker_hostname": os.getenv("HOSTNAME", "unknown"),
                    "task_id": self.request.id
                }
                
                success = self.task_state_manager.start_task(
                    document_id=document_id,
                    celery_task_id=self.request.id,
                    worker_info=worker_info
                )
                
                if success:
                    logger.info(f"Marked task {self.request.id} as STARTED for document {document_id}")
                else:
                    logger.warning(f"Failed to mark task as STARTED for document {document_id}")
                    
            except Exception as e:
                logger.error(f"Error updating task state to STARTED: {str(e)}")
        
        # Check for immediate control signals
        signal = check_task_control_signals(self.task_state_manager, document_id)
        if signal == 'cancel':
            return {"status": "cancelled", "message": "Task cancelled before extraction"}
        elif signal == 'pause':
            return {"status": "paused", "message": "Task paused before extraction"}
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "extraction", 5, "Starting extraction")
        
        # Get processing context with better error handling
        try:
            context = self._get_processing_context(document_id)
        except Exception as e:
            error_msg = f"Failed to get processing context: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
        
        # Verify file exists before processing
        file_path = context.get("stored_file_path") or context.get("file_path")
        if not file_path or not os.path.exists(file_path):
            error_msg = f"File not found for processing: {file_path}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
        
        # Verify we can create the document directory structure
        doc_dir = context["doc_dir"]
        storage_adapter = context["storage_adapter"]
        
        if not storage_adapter.create_directory(doc_dir):
            error_msg = f"Failed to create document directory: {doc_dir}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
        
        try:
            state_manager = self._get_state_manager(document_id, context)
        except Exception as e:
            error_msg = f"Failed to initialize state manager: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
        
        # Initialize processor with control wrapper
        processor = ExtractionProcessor(self.config)
        controlled_processor = ControlledProcessor(processor, self.task_state_manager, document_id)
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "extraction", 10, "Extracting content")
        
        # Execute extraction with control checks
        result = controlled_processor.execute_with_checkpoints(state_manager, context)
        
        # Handle control signals
        if result["status"] == "cancelled":
            return result
        elif result["status"] == "paused":
            # Save checkpoint
            checkpoint_data = {
                "stage": "extraction", 
                "paused_at": datetime.now().isoformat(),
                "percent_complete": 50
            }
            if self.checkpoint_manager:
                self.checkpoint_manager.save_checkpoint(document_id, "extraction", checkpoint_data)
            return result
        elif result["status"] != "success":
            # Task failed
            logger.error(f"Extraction failed for {document_id}: {result.get('error')}")
            return result
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "extraction", 90, "Extraction complete")
        
        # Chain to next task
        chunking_result = chunk_document_task.delay(document_id)
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "extraction", 100, "Chaining to chunking")
        
        logger.info(f"Extraction completed successfully for {document_id}, chaining to {chunking_result.id}")
        
        return {
            "status": "success",
            "stage": "extraction",
            "next_task_id": chunking_result.id,
            **result
        }
        
    except Exception as e:
        error_msg = f"Error in extract_document_task for {document_id}: {str(e)}"
        logger.error(error_msg)
        
        # Mark task as failed in task state manager
        if self.task_state_manager:
            try:
                self.task_state_manager.complete_task(
                    document_id=document_id,
                    success=False,
                    error_details={"error": str(e), "stage": "extraction"}
                )
            except Exception as tm_error:
                logger.error(f"Failed to update task manager with failure: {str(tm_error)}")
        
        return {"status": "error", "error": error_msg}

@celery_app.task(base=BaseDocumentTask, bind=True, name='services.celery.tasks.document_tasks.chunk_document_task')
def chunk_document_task(self, document_id: str):
    """Chunk document content with pause/resume/cancel support"""
    logger.info(f"Starting document chunking for {document_id}")
    
    try:
        # Check for immediate control signals
        signal = check_task_control_signals(self.task_state_manager, document_id)
        if signal == 'cancel':
            return {"status": "cancelled", "message": "Task cancelled before chunking"}
        elif signal == 'pause':
            return {"status": "paused", "message": "Task paused before chunking"}
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "chunking", 5, "Starting chunking")
        
        # Get processing context
        context = self._get_processing_context(document_id)
        state_manager = self._get_state_manager(document_id, context)
        
        # Initialize processor with control wrapper
        processor = ChunkingProcessor(self.config)
        controlled_processor = ControlledProcessor(processor, self.task_state_manager, document_id)
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "chunking", 10, "Processing chunks")
        
        # Execute chunking with control checks
        result = controlled_processor.execute_with_checkpoints(state_manager, context)
        
        # Handle control signals
        if result["status"] == "cancelled":
            return result
        elif result["status"] == "paused":
            # Save checkpoint
            checkpoint_data = {
                "stage": "chunking",
                "paused_at": datetime.now().isoformat(),
                "percent_complete": 50
            }
            self.checkpoint_manager.save_checkpoint(document_id, "chunking", checkpoint_data)
            return result
        elif result["status"] != "success":
            return result
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "chunking", 90, "Chunking complete")
        
        # Chain to next task
        embedding_result = embed_document_task.delay(document_id)
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "chunking", 100, "Chaining to embedding")
        
        return {
            "status": "success",
            "stage": "chunking",
            "next_task_id": embedding_result.id,
            **result
        }
        
    except Exception as e:
        logger.error(f"Error in chunk_document_task for {document_id}: {str(e)}")
        return {"status": "error", "error": str(e)}

@celery_app.task(base=BaseDocumentTask, bind=True, name='services.celery.tasks.document_tasks.embed_document_task')
def embed_document_task(self, document_id: str):
    """Generate embeddings for document chunks with pause/resume/cancel support"""
    logger.info(f"Starting document embedding for {document_id}")
    
    try:
        # Check for immediate control signals
        signal = check_task_control_signals(self.task_state_manager, document_id)
        if signal == 'cancel':
            return {"status": "cancelled", "message": "Task cancelled before embedding"}
        elif signal == 'pause':
            return {"status": "paused", "message": "Task paused before embedding"}
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "embedding", 5, "Starting embedding")
        
        # Get processing context
        context = self._get_processing_context(document_id)
        state_manager = self._get_state_manager(document_id, context)
        
        # Initialize processor with control wrapper
        processor = EmbeddingProcessor(self.config)
        controlled_processor = ControlledProcessor(processor, self.task_state_manager, document_id)
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "embedding", 10, "Generating embeddings")
        
        # Execute embedding with control checks
        result = controlled_processor.execute_with_checkpoints(state_manager, context)
        
        # Handle control signals
        if result["status"] == "cancelled":
            return result
        elif result["status"] == "paused":
            # Save checkpoint
            checkpoint_data = {
                "stage": "embedding",
                "paused_at": datetime.now().isoformat(),
                "percent_complete": 50
            }
            self.checkpoint_manager.save_checkpoint(document_id, "embedding", checkpoint_data)
            return result
        elif result["status"] != "success":
            return result
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "embedding", 90, "Embedding complete")
        
        # Chain to next task
        tree_result = build_tree_task.delay(document_id)
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "embedding", 100, "Chaining to tree building")
        
        return {
            "status": "success",
            "stage": "embedding",
            "next_task_id": tree_result.id,
            **result
        }
        
    except Exception as e:
        logger.error(f"Error in embed_document_task for {document_id}: {str(e)}")
        return {"status": "error", "error": str(e)}

@celery_app.task(base=BaseDocumentTask, bind=True, name='services.celery.tasks.document_tasks.build_tree_task')
def build_tree_task(self, document_id: str):
    """Build RAPTOR tree with pause/resume/cancel support"""
    logger.info(f"Starting tree building for {document_id}")
    
    try:
        # Check for immediate control signals
        signal = check_task_control_signals(self.task_state_manager, document_id)
        if signal == 'cancel':
            return {"status": "cancelled", "message": "Task cancelled before tree building"}
        elif signal == 'pause':
            return {"status": "paused", "message": "Task paused before tree building"}
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "tree_building", 5, "Starting tree building")
        
        # Get processing context
        context = self._get_processing_context(document_id)
        state_manager = self._get_state_manager(document_id, context)
        
        # Initialize processor with control wrapper
        processor = TreeBuildingProcessor(self.config)
        controlled_processor = ControlledProcessor(processor, self.task_state_manager, document_id)
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "tree_building", 10, "Building RAPTOR tree")
        
        # Execute tree building with control checks
        result = controlled_processor.execute_with_checkpoints(state_manager, context)
        
        # Handle control signals
        if result["status"] == "cancelled":
            return result
        elif result["status"] == "paused":
            # Save checkpoint
            checkpoint_data = {
                "stage": "tree_building",
                "paused_at": datetime.now().isoformat(),
                "percent_complete": 50
            }
            self.checkpoint_manager.save_checkpoint(document_id, "tree_building", checkpoint_data)
            return result
        elif result["status"] != "success":
            return result
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "tree_building", 90, "Tree building complete")
        
        # Chain to final task
        storage_result = store_vectors_task.delay(document_id)
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "tree_building", 100, "Chaining to vector storage")
        
        return {
            "status": "success",
            "stage": "tree_building",
            "next_task_id": storage_result.id,
            **result
        }
        
    except Exception as e:
        logger.error(f"Error in build_tree_task for {document_id}: {str(e)}")
        return {"status": "error", "error": str(e)}

@celery_app.task(base=BaseDocumentTask, bind=True, name='services.celery.tasks.document_tasks.store_vectors_task')
def store_vectors_task(self, document_id: str):
    """Store vectors in database with pause/resume/cancel support"""
    logger.info(f"Starting vector storage for {document_id}")
    
    try:
        # Check for immediate control signals
        signal = check_task_control_signals(self.task_state_manager, document_id)
        if signal == 'cancel':
            return {"status": "cancelled", "message": "Task cancelled before vector storage"}
        elif signal == 'pause':
            return {"status": "paused", "message": "Task paused before vector storage"}
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "vector_storage", 5, "Starting vector storage")
        
        # Get processing context
        context = self._get_processing_context(document_id)
        state_manager = self._get_state_manager(document_id, context)
        
        # Initialize processor with control wrapper
        processor = VectorStorageProcessor(self.config)
        controlled_processor = ControlledProcessor(processor, self.task_state_manager, document_id)
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "vector_storage", 10, "Storing vectors")
        
        # Execute vector storage with control checks
        result = controlled_processor.execute_with_checkpoints(state_manager, context)
        
        # Handle control signals
        if result["status"] == "cancelled":
            return result
        elif result["status"] == "paused":
            # Save checkpoint
            checkpoint_data = {
                "stage": "vector_storage",
                "paused_at": datetime.now().isoformat(),
                "percent_complete": 50
            }
            self.checkpoint_manager.save_checkpoint(document_id, "vector_storage", checkpoint_data)
            return result
        elif result["status"] != "success":
            return result
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "vector_storage", 90, "Vector storage complete")
        
        # Mark entire workflow as complete
        if self.task_state_manager:
            self.task_state_manager.complete_task(document_id, success=True)
        
        # Update progress
        update_task_progress(self.task_state_manager, document_id, "vector_storage", 100, "Document processing complete")
        
        return {
            "status": "success",
            "stage": "vector_storage", 
            "workflow_complete": True,
            **result
        }
        
    except Exception as e:
        logger.error(f"Error in store_vectors_task for {document_id}: {str(e)}")
        return {"status": "error", "error": str(e)}

# Task chain starter
@celery_app.task(base=BaseDocumentTask, bind=True, name='services.celery.tasks.document_tasks.start_document_processing_chain')
def start_document_processing_chain(self, document_id: str):
    """Start the complete document processing chain with proper task state management"""
    logger.info(f"Starting document processing chain for {document_id}")
    
    try:
        # Update the workflow task with this Celery task ID
        if self.task_state_manager:
            try:
                # Check if workflow task exists, if not create it
                existing_task = self.task_state_manager.get_task_status(document_id)
                if not existing_task:
                    logger.info(f"Creating workflow task for {document_id}")
                    self.task_state_manager.create_workflow_task(
                        document_id=document_id,
                        initial_stage="extraction",
                        celery_task_id=self.request.id
                    )
                else:
                    # Update existing task with this Celery task ID
                    logger.info(f"Updating existing workflow task for {document_id}")
                    self.task_state_manager.start_task(
                        document_id=document_id,
                        celery_task_id=self.request.id,
                        worker_info={
                            "worker_id": os.getenv("CELERY_WORKER_NAME", "unknown"),
                            "task_type": "processing_chain"
                        }
                    )
                    
            except Exception as e:
                logger.warning(f"Could not update task state for {document_id}: {str(e)}")
        
        # Start with extraction
        extraction_result = extract_document_task.delay(document_id)
        
        logger.info(f"Processing chain started for {document_id}, first task: {extraction_result.id}")
        
        return {
            "status": "started",
            "document_id": document_id,
            "first_task_id": extraction_result.id,
            "message": "Document processing chain started"
        }
        
    except Exception as e:
        error_msg = f"Error starting processing chain for {document_id}: {str(e)}"
        logger.error(error_msg)
        
        # Mark as failed in task state manager
        if self.task_state_manager:
            try:
                self.task_state_manager.complete_task(
                    document_id=document_id,
                    success=False,
                    error_details={"error": str(e), "stage": "chain_start"}
                )
            except:
                pass
                
        return {"status": "error", "error": error_msg}