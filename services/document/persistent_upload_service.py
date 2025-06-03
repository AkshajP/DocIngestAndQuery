import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from core.config import get_config
from db.document_store.repository import DocumentMetadataRepository
from db.task_store.repository import TaskRepository, TaskStatus
from services.document.storage import LocalStorageAdapter, S3StorageAdapter
from .processing_state_manager import ProcessingStateManager, ProcessingStage
from .stage_processors import (
    ExtractionProcessor, ChunkingProcessor, EmbeddingProcessor,
    TreeBuildingProcessor, VectorStorageProcessor
)

logger = logging.getLogger(__name__)

class PersistentUploadService:
    """
    Enhanced service for persistent document upload and processing with Celery integration.
    Manages document processing through stages with task tracking, pause/resume/cancel capabilities.
    """
    
    def __init__(self, config=None):
        """Initialize the persistent upload service with task management"""
        self.config = config or get_config()
        
        # Initialize storage adapter
        self.storage_adapter = self._initialize_storage_adapter()
        
        # Initialize repositories
        self.doc_repository = DocumentMetadataRepository()
        self.task_repository = TaskRepository()
        
        # Initialize stage processors
        self.processors = {
            ProcessingStage.EXTRACTION.value: ExtractionProcessor(self.config),
            ProcessingStage.CHUNKING.value: ChunkingProcessor(self.config),
            ProcessingStage.EMBEDDING.value: EmbeddingProcessor(self.config),
            ProcessingStage.TREE_BUILDING.value: TreeBuildingProcessor(self.config),
            ProcessingStage.VECTOR_STORAGE.value: VectorStorageProcessor(self.config)
        }
        
        self._cleanup_stale_locks_on_startup()
        
        logger.info("Initialized persistent upload service with Celery task integration")
    
    def _initialize_storage_adapter(self):
        """Initialize the appropriate storage adapter based on configuration"""
        storage_type = self.config.storage.storage_type.lower()
        
        if storage_type == "s3":
            logger.info(f"Initializing S3 storage adapter with bucket {self.config.storage.s3_bucket}")
            return S3StorageAdapter(
                bucket_name=self.config.storage.s3_bucket,
                prefix=self.config.storage.s3_prefix,
                region=self.config.storage.aws_region
            )
        else:
            logger.info(f"Initializing local storage adapter with directory {self.config.storage.storage_dir}")
            return LocalStorageAdapter()
    
    def upload_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        case_id: str = "default",
        user_id: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
        force_restart: bool = False,
        celery_task_ids: Optional[Dict[str, str]] = None,
        use_celery: bool = False  # New parameter to enable Celery
    ) -> Dict[str, Any]:
        """
        Upload and process a document through the complete pipeline with optional Celery integration.
        
        Args:
            file_path: Path to the document file
            document_id: Optional custom ID (generated if not provided)
            case_id: Case ID for document grouping (required for task management)
            user_id: User ID for task tracking (required for task management)
            metadata: Optional metadata about the document
            force_restart: If True, restart processing from the beginning
            celery_task_ids: Optional mapping of stage names to Celery task IDs
            use_celery: If True, use Celery tasks for async processing
            
        Returns:
            Dictionary with processing status and information
        """
        process_start_time = time.time()
        
        # Generate document ID if not provided
        if not document_id:
            timestamp = int(time.time())
            base_name = os.path.basename(file_path)
            safe_name = ''.join(c for c in base_name.split('.')[0].replace(' ', '_') 
                              if c.isalnum() or c == '_')
            document_id = f"doc_{timestamp}_{safe_name}"
        
        try:
            # Check for concurrent operations
            if not self._check_concurrent_processing(document_id, force_restart):
                return {
                    "status": "error",
                    "document_id": document_id,
                    "error": "Document is already being processed by another operation",
                    "processing_time": 0
                }
            
            # Initialize document processing setup
            result = self._initialize_document_processing(
                document_id, case_id, user_id, file_path, metadata, force_restart
            )
            
            if result["status"] == "error":
                return result
            
            doc_dir = result["doc_dir"]
            stored_file_path = result["stored_file_path"]
            
            # Initialize processing state manager with task integration
            state_manager = ProcessingStateManager(
                document_id=document_id,
                storage_adapter=self.storage_adapter,
                doc_dir=doc_dir,
                doc_repository=self.doc_repository,
                case_id=case_id,  # Required for task management
                user_id=user_id   # Required for task management
            )
            
            # Reset processing if force_restart is True
            if force_restart:
                self._handle_force_restart(state_manager, document_id)
            
            # Create processing lock
            self._create_processing_lock(doc_dir)
            
            # Create enhanced processing context with task integration
            context = self._create_processing_context(
                document_id, case_id, user_id, file_path, stored_file_path, 
                doc_dir, metadata, celery_task_ids
            )
            
            # Choose processing method based on use_celery parameter
            if use_celery:
                pipeline_result = self._execute_async_processing_pipeline(state_manager, context)
            else:
                # Add storage adapter for synchronous processing
                context["storage_adapter"] = self.storage_adapter
                pipeline_result = self._execute_processing_pipeline_with_tasks(state_manager, context)
            
            # Calculate total processing time
            total_processing_time = time.time() - process_start_time
            
            # Update final document metadata
            final_result = self._finalize_document_processing(
                document_id, state_manager, pipeline_result, total_processing_time
            )
            
            # Remove processing lock
            self._remove_processing_lock(doc_dir)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            
            # Cleanup on error
            self._cleanup_on_error(document_id, str(e))
            
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e),
                "processing_time": time.time() - process_start_time
            }
    
    def _execute_async_processing_pipeline(
        self,
        state_manager: ProcessingStateManager,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute processing pipeline using Celery tasks (async).
        """
        from services.celery.task_utils import DocumentTaskOrchestrator
        
        document_id = context['document_id']
        case_id = context['case_id']
        user_id = context['user_id']
        
        logger.info(f"Starting async processing pipeline for document {document_id}")
        
        try:
            # CRITICAL: Ensure file is stored before async processing
            stored_file_path = context.get('stored_file_path')
            original_file_path = context.get('file_path')
            
            # Verify the stored file exists and is accessible
            if not stored_file_path or not self.storage_adapter.file_exists(stored_file_path):
                # If stored file doesn't exist, we need to store it from the original
                if original_file_path and os.path.exists(original_file_path):
                    logger.warning(f"Stored file not found, copying from original: {original_file_path}")
                    
                    # Create target path
                    doc_dir = context['doc_dir']
                    target_path = os.path.join(doc_dir, "original.pdf")
                    
                    # Copy file to permanent storage
                    with open(original_file_path, 'rb') as f:
                        file_content = f.read()
                    
                    success = self.storage_adapter.write_file(file_content, target_path)
                    if success:
                        stored_file_path = target_path
                        context['stored_file_path'] = stored_file_path
                        logger.info(f"File stored for async processing: {stored_file_path}")
                    else:
                        raise Exception(f"Failed to store file for async processing: {target_path}")
                else:
                    raise Exception(f"Neither stored file nor original file is accessible for async processing")
            
            # Update context to use stored file path for workers
            async_context = context.copy()
            async_context['file_path'] = stored_file_path  # Workers should use stored file
            
            # Initialize task orchestrator
            orchestrator = DocumentTaskOrchestrator()
            
            # Create and submit processing chain
            chain_result = orchestrator.create_processing_chain(
                document_id=document_id,
                case_id=case_id,
                user_id=user_id,
                context=async_context,
                start_from_stage=state_manager.get_current_stage()
            )
            
            # Return immediately with chain information
            return {
                "status": "processing",
                "chain_id": chain_result.id,
                "message": "Async processing chain submitted",
                "async_mode": True,
                "stored_file_path": stored_file_path
            }
            
        except Exception as e:
            logger.error(f"Error starting async processing: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to start async processing: {str(e)}",
                "async_mode": True
            }
            
    def _check_concurrent_processing(self, document_id: str, force_restart: bool) -> bool:
        """Check if document is already being processed"""
        existing_doc = self.doc_repository.get_document(document_id)
        if existing_doc and existing_doc.get("status") == "processing" and not force_restart:
            doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
            lock_file = os.path.join(doc_dir, "stages", "processing.lock")
            
            if self.storage_adapter.file_exists(lock_file):
                try:
                    lock_content = self.storage_adapter.read_file(lock_file)
                    if lock_content:
                        lock_data = json.loads(lock_content)
                        locked_at = datetime.fromisoformat(lock_data["locked_at"])
                        if (datetime.now() - locked_at).total_seconds() < 300:  # 5 minutes
                            return False
                except:
                    pass  # If we can't read the lock, proceed
        return True
    
    def _handle_force_restart(self, state_manager: ProcessingStateManager, document_id: str):
        """Handle force restart scenario"""
        state_manager.reset_to_stage(ProcessingStage.UPLOAD.value)
        # Mark upload as complete again after reset since file is already stored
        state_manager.mark_stage_complete(ProcessingStage.UPLOAD.value)
        
        # Cancel any active tasks
        active_tasks = self.task_repository.get_tasks_by_document(document_id)
        for task in active_tasks:
            if task['task_status'] in ['pending', 'running', 'paused']:
                self.task_repository.update_task_status(
                    task['celery_task_id'], 
                    TaskStatus.CANCELLED,
                    error_details="Force restart requested"
                )
        
        logger.info(f"Force restarting processing for document {document_id}")
    
    def _create_processing_context(
        self, 
        document_id: str, 
        case_id: str, 
        user_id: str, 
        file_path: str, 
        stored_file_path: str, 
        doc_dir: str, 
        metadata: Optional[Dict[str, Any]], 
        celery_task_ids: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Create enhanced processing context with task integration"""
        context = {
            "document_id": document_id,
            "case_id": case_id,
            "user_id": user_id,
            "file_path": file_path,
            "stored_file_path": stored_file_path,
            "doc_dir": doc_dir,
            "storage_adapter": self.storage_adapter,
            "metadata": metadata or {},
            "celery_task_ids": celery_task_ids or {},
            # Add test detection
            "is_test": (metadata or {}).get("is_integration_test", False) or "test" in case_id.lower()
        }
        return context
    
    def _execute_processing_pipeline_with_tasks(
        self,
        state_manager: ProcessingStateManager,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the processing pipeline with enhanced Celery task coordination.
        """
        
        # Define stage execution order
        stage_order = [
            ProcessingStage.EXTRACTION.value,
            ProcessingStage.CHUNKING.value,
            ProcessingStage.EMBEDDING.value,
            ProcessingStage.TREE_BUILDING.value,
            ProcessingStage.VECTOR_STORAGE.value
        ]
        
        current_stage = state_manager.get_current_stage()
        completed_stages = state_manager.get_completed_stages()
        
        # Find starting point in pipeline
        start_index = self._find_pipeline_start_index(current_stage, stage_order, state_manager)
        
        logger.info(f"Starting processing pipeline for document {context['document_id']} from stage {current_stage}")
        
        # Execute stages in order with task coordination
        last_result = {}
        for i in range(start_index, len(stage_order)):
            stage = stage_order[i]
            
            # Skip completed stages
            if stage in completed_stages:
                logger.info(f"Skipping completed stage: {stage}")
                continue
            
            # Check for task cancellation before each stage
            if self._is_document_cancelled(context['document_id']):
                return {
                    "status": "cancelled",
                    "error": "Processing cancelled by user request",
                    "failed_stage": stage
                }
            
            # AUTO-ADVANCE: Try to advance to this stage if needed
            if not self._advance_to_next_stage_if_needed(state_manager, stage):
                if state_manager.get_current_stage() != stage:
                    error_msg = f"Cannot advance to stage {stage} from {state_manager.get_current_stage()}"
                    logger.error(error_msg)
                    return {
                        "status": "error",
                        "error": error_msg,
                        "failed_stage": stage
                    }
            
            # Get processor for this stage
            processor = self.processors.get(stage)
            if not processor:
                error_msg = f"No processor found for stage: {stage}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "failed_stage": stage
                }
            
            # Prepare context with Celery task ID if available
            stage_context = context.copy()
            celery_task_ids = context.get("celery_task_ids", {})
            if stage in celery_task_ids:
                stage_context["celery_task_id"] = celery_task_ids[stage]
            
            # Validate processor can execute
            if not processor.can_execute(state_manager, stage_context):
                error_msg = f"Stage {stage} processor reports it cannot execute"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "failed_stage": stage
                }
            
            # Validate dependencies
            if not processor.validate_dependencies(state_manager, stage_context):
                error_msg = f"Dependencies not met for stage {stage}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "failed_stage": stage
                }
            
            # Execute stage with task tracking
            logger.info(f"Executing stage: {stage}")
            result = processor.execute(state_manager, stage_context)
            
            if result["status"] != "success":
                logger.error(f"Stage {stage} failed: {result.get('error')}")
                return {
                    "status": "error",
                    "error": result.get("error"),
                    "failed_stage": stage,
                    "processing_time": result.get("processing_time", 0)
                }
            
            last_result = result
            logger.info(f"Stage {stage} completed successfully")
        
        logger.info(f"Processing pipeline completed successfully for document {context['document_id']}")
        
        return {
            "status": "success",
            **last_result
        }
    
    def _find_pipeline_start_index(self, current_stage: str, stage_order: List[str], state_manager: ProcessingStateManager) -> int:
        """Find starting point in pipeline based on current stage"""
        start_index = 0
        if current_stage in stage_order:
            start_index = stage_order.index(current_stage)
        elif current_stage == ProcessingStage.UPLOAD.value:
            if state_manager.is_stage_complete(ProcessingStage.UPLOAD.value):
                start_index = 0  # Start from extraction
                # Advance to extraction stage
                state_manager.processing_state["current_stage"] = ProcessingStage.EXTRACTION.value
                state_manager._save_processing_state()
                logger.info(f"Advanced from upload to extraction for document {state_manager.document_id}")
            else:
                raise ValueError("Upload stage not completed")
        return start_index
    
    def _is_document_cancelled(self, document_id: str) -> bool:
        """Check if document processing has been cancelled"""
        try:
            active_tasks = self.task_repository.get_tasks_by_document(document_id)
            for task in active_tasks:
                if task['task_status'] == 'cancelled':
                    return True
            return False
        except Exception as e:
            logger.warning(f"Error checking cancellation status: {str(e)}")
            return False
    
    def _advance_to_next_stage_if_needed(self, state_manager: ProcessingStateManager, target_stage: str) -> bool:
        """Advance processing state to target stage if prerequisites are met"""
        current_stage = state_manager.get_current_stage()
        
        # Define stage progression
        stage_progression = {
            ProcessingStage.EXTRACTION.value: ProcessingStage.UPLOAD.value,
            ProcessingStage.CHUNKING.value: ProcessingStage.EXTRACTION.value,
            ProcessingStage.EMBEDDING.value: ProcessingStage.CHUNKING.value,
            ProcessingStage.TREE_BUILDING.value: ProcessingStage.EMBEDDING.value,
            ProcessingStage.VECTOR_STORAGE.value: ProcessingStage.TREE_BUILDING.value
        }
        
        # If already at target stage, no advancement needed
        if current_stage == target_stage:
            return True
        
        # Check if we can advance from current stage to target stage
        required_previous_stage = stage_progression.get(target_stage)
        
        if required_previous_stage and current_stage == required_previous_stage:
            # Check if current stage is complete
            if state_manager.is_stage_complete(current_stage):
                # Advance to target stage
                state_manager.processing_state["current_stage"] = target_stage
                state_manager._save_processing_state()
                logger.info(f"Advanced document {state_manager.document_id} from {current_stage} to {target_stage}")
                return True
            else:
                logger.error(f"Cannot advance to {target_stage}: {current_stage} is not complete")
                return False
        
        # Special case: handle upload to extraction transition
        if (current_stage == ProcessingStage.UPLOAD.value and 
            target_stage == ProcessingStage.EXTRACTION.value):
            
            if state_manager.is_stage_complete(ProcessingStage.UPLOAD.value):
                # Advance to extraction stage
                state_manager.processing_state["current_stage"] = ProcessingStage.EXTRACTION.value
                state_manager._save_processing_state()
                logger.info(f"Advanced document {state_manager.document_id} from upload to extraction")
                return True
            else:
                logger.error(f"Upload stage not complete for document {state_manager.document_id}")
                return False
        
        # Check if target stage is already completed (skip case)
        if state_manager.is_stage_complete(target_stage):
            logger.info(f"Stage {target_stage} already completed for document {state_manager.document_id}")
            return False  # Will be skipped in main loop
        
        logger.error(f"Cannot advance from {current_stage} to {target_stage}")
        return False
    
    def _finalize_document_processing(
        self, 
        document_id: str, 
        state_manager: ProcessingStateManager, 
        pipeline_result: Dict[str, Any], 
        total_processing_time: float
    ) -> Dict[str, Any]:
        """Finalize document processing and update metadata"""
        
        final_metadata = {
            "total_processing_time": total_processing_time,
            "processing_state": state_manager.get_comprehensive_status()
        }
        
        if pipeline_result["status"] == "success":
            final_metadata.update({
                "status": "processed",
                "processing_date": datetime.now().isoformat()
            })
            
            # Mark processing as completed
            state_manager.mark_stage_complete(ProcessingStage.COMPLETED.value)
        else:
            final_metadata.update({
                "status": "failed",
                "failure_stage": pipeline_result.get("failed_stage"),
                "error_message": pipeline_result.get("error"),
                "failure_time": datetime.now().isoformat()
            })
        
        self.doc_repository.update_document(document_id, final_metadata)
        
        # Prepare response
        response = {
            "status": pipeline_result["status"],
            "document_id": document_id,
            "case_id": state_manager.case_id,
            "processing_time": total_processing_time,
            "stored_file_path": final_metadata.get("stored_file_path"),
            "doc_dir": state_manager.doc_dir,
            "processing_state": final_metadata["processing_state"]
        }
        
        if pipeline_result["status"] == "success":
            response.update({
                "chunks_count": pipeline_result.get("chunks_count", 0),
                "tree_nodes_count": pipeline_result.get("tree_nodes_count", 0),
                "raptor_levels": pipeline_result.get("raptor_levels", [])
            })
        else:
            response.update({
                "error": pipeline_result.get("error"),
                "failed_stage": pipeline_result.get("failed_stage")
            })
        
        return response
    
    def _cleanup_on_error(self, document_id: str, error_message: str):
        """Cleanup resources on processing error"""
        try:
            doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
            self._remove_processing_lock(doc_dir)
        except:
            pass
        
        # Update document with error status
        self.doc_repository.update_document(document_id, {
            "status": "failed",
            "failure_stage": "initialization",
            "error_message": error_message,
            "failure_time": datetime.now().isoformat()
        })
    
    # === TASK CONTROL METHODS ===
    
    def pause_document_processing(self, document_id: str, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Pause document processing at specified stage or all active stages.
        
        Args:
            document_id: Document ID
            stage: Optional specific stage to pause (if None, pauses all active)
            
        Returns:
            Dictionary with pause operation results
        """
        try:
            # Get active tasks for the document
            active_tasks = self.task_repository.get_active_tasks()
            document_tasks = [task for task in active_tasks if task['document_id'] == document_id]
            
            if not document_tasks:
                return {
                    "status": "error",
                    "error": "No active tasks found for document"
                }
            
            paused_tasks = []
            
            for task in document_tasks:
                task_stage = task['processing_stage']
                
                # Skip if specific stage requested and this isn't it
                if stage and task_stage != stage:
                    continue
                
                # Only pause if task can be paused
                if task['can_pause']:
                    success = self.task_repository.update_task_status(
                        task['celery_task_id'],
                        TaskStatus.PAUSED,
                        checkpoint_data={"paused_at": datetime.now().isoformat()}
                    )
                    
                    if success:
                        paused_tasks.append(task_stage)
                        logger.info(f"Paused task for document {document_id}, stage {task_stage}")
            
            if paused_tasks:
                return {
                    "status": "success",
                    "message": f"Paused tasks for stages: {', '.join(paused_tasks)}",
                    "paused_stages": paused_tasks
                }
            else:
                return {
                    "status": "error",
                    "error": "No tasks could be paused"
                }
                
        except Exception as e:
            logger.error(f"Error pausing document processing: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def resume_document_processing(self, document_id: str, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume paused document processing.
        
        Args:
            document_id: Document ID
            stage: Optional specific stage to resume (if None, resumes all paused)
            
        Returns:
            Dictionary with resume operation results
        """
        try:
            # Get paused tasks for the document
            all_tasks = self.task_repository.get_tasks_by_document(document_id)
            paused_tasks = [task for task in all_tasks if task['task_status'] == 'paused']
            
            if not paused_tasks:
                return {
                    "status": "error",
                    "error": "No paused tasks found for document"
                }
            
            resumed_tasks = []
            
            for task in paused_tasks:
                task_stage = task['processing_stage']
                
                # Skip if specific stage requested and this isn't it
                if stage and task_stage != stage:
                    continue
                
                # Only resume if task can be resumed
                if task['can_resume']:
                    success = self.task_repository.update_task_status(
                        task['celery_task_id'],
                        TaskStatus.RUNNING,
                        checkpoint_data={"resumed_at": datetime.now().isoformat()}
                    )
                    
                    if success:
                        resumed_tasks.append(task_stage)
                        logger.info(f"Resumed task for document {document_id}, stage {task_stage}")
            
            if resumed_tasks:
                return {
                    "status": "success",
                    "message": f"Resumed tasks for stages: {', '.join(resumed_tasks)}",
                    "resumed_stages": resumed_tasks
                }
            else:
                return {
                    "status": "error",
                    "error": "No tasks could be resumed"
                }
                
        except Exception as e:
            logger.error(f"Error resuming document processing: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def cancel_document_processing(self, document_id: str, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel document processing.
        
        Args:
            document_id: Document ID
            stage: Optional specific stage to cancel (if None, cancels all active)
            
        Returns:
            Dictionary with cancellation results
        """
        try:
            # Get active tasks for the document
            all_tasks = self.task_repository.get_tasks_by_document(document_id)
            active_tasks = [task for task in all_tasks if task['task_status'] in ['pending', 'running', 'paused']]
            
            if not active_tasks:
                return {
                    "status": "error",  
                    "error": "No active tasks found for document"
                }
            
            cancelled_tasks = []
            
            for task in active_tasks:
                task_stage = task['processing_stage']
                
                # Skip if specific stage requested and this isn't it
                if stage and task_stage != stage:
                    continue
                
                # Only cancel if task can be cancelled
                if task['can_cancel']:
                    success = self.task_repository.update_task_status(
                        task['celery_task_id'],
                        TaskStatus.CANCELLED,
                        error_details="Cancelled by user request"
                    )
                    
                    if success:
                        cancelled_tasks.append(task_stage)
                        logger.info(f"Cancelled task for document {document_id}, stage {task_stage}")
            
            # Update document status
            if cancelled_tasks:
                self.doc_repository.update_document(document_id, {
                    "status": "cancelled",
                    "cancellation_time": datetime.now().isoformat(),
                    "cancelled_stages": cancelled_tasks
                })
                
                return {
                    "status": "success",
                    "message": f"Cancelled tasks for stages: {', '.join(cancelled_tasks)}",
                    "cancelled_stages": cancelled_tasks
                }
            else:
                return {
                    "status": "error",
                    "error": "No tasks could be cancelled"
                }
                
        except Exception as e:
            logger.error(f"Error cancelling document processing: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_document_task_status(self, document_id: str) -> Dict[str, Any]:
        """
        Get comprehensive task status for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Dictionary with detailed task status information
        """
        try:
            # Get document metadata
            document = self.doc_repository.get_document(document_id)
            if not document:
                return {
                    "status": "error",
                    "error": "Document not found"
                }
            
            # Get all tasks for the document
            tasks = self.task_repository.get_tasks_by_document(document_id)
            
            # Get state manager for comprehensive status
            doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
            state_manager = ProcessingStateManager(
                document_id=document_id,
                storage_adapter=self.storage_adapter,
                doc_dir=doc_dir,
                doc_repository=self.doc_repository,
                case_id=document.get("case_id", "default"),
                user_id=document.get("user_id", "system")
            )
            
            comprehensive_status = state_manager.get_comprehensive_status()
            
            return {
                "status": "success",
                "document_id": document_id,
                "document_metadata": document,
                "tasks": tasks,
                "comprehensive_status": comprehensive_status,
                "task_controllability": state_manager.get_task_controllability()
            }
            
        except Exception as e:
            logger.error(f"Error getting task status for {document_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    # === EXISTING METHODS (keeping for backward compatibility) ===
    
    def retry_document_processing(
        self,
        document_id: str,
        from_stage: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retry document processing from a specific stage or current failed stage"""
        # ... existing implementation with minor updates for task integration ...
        try:
            # ... existing retry logic ...
            
            # Also cancel any stuck tasks
            if document_id:
                self.cancel_document_processing(document_id)
            
            # ... rest of existing implementation ...
        except Exception as e:
            logger.error(f"Error retrying document {document_id}: {str(e)}")
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e)
            }
    
    def get_document_processing_status(self, document_id: str) -> Dict[str, Any]:
        """Get detailed processing status for a document"""
        # Enhanced version that includes task status
        return self.get_document_task_status(document_id)
    
    # === UTILITY METHODS ===
    
    def _create_processing_lock(self, doc_dir: str):
        """Create a processing lock file to prevent concurrent operations"""
        try:
            stages_dir = os.path.join(doc_dir, "stages")
            self.storage_adapter.create_directory(stages_dir)
            
            lock_file = os.path.join(stages_dir, "processing.lock")
            lock_data = {
                "locked_at": datetime.now().isoformat(),
                "process_id": os.getpid()
            }
            
            self.storage_adapter.write_file(json.dumps(lock_data), lock_file)
            logger.debug(f"Created processing lock: {lock_file}")
        except Exception as e:
            logger.warning(f"Failed to create processing lock: {str(e)}")
    
    def _remove_processing_lock(self, doc_dir: str):
        """Remove the processing lock file"""
        try:
            lock_file = os.path.join(doc_dir, "stages", "processing.lock")
            if self.storage_adapter.file_exists(lock_file):
                self.storage_adapter.delete_file(lock_file)
                logger.debug(f"Removed processing lock: {lock_file}")
        except Exception as e:
            logger.warning(f"Failed to remove processing lock: {str(e)}")
    
    def _cleanup_stale_locks_on_startup(self):
        """Clean up stale locks when service starts (after server restart)"""
        try:
            cleanup_result = self.cleanup_stale_locks()
            if cleanup_result["cleaned_documents"] > 0:
                logger.info(f"Startup cleanup: Reset {cleanup_result['cleaned_documents']} documents and removed {cleanup_result['locks_removed']} stale locks")
        except Exception as e:
            logger.error(f"Error during startup lock cleanup: {str(e)}")

    def cleanup_stale_locks(self, max_lock_age_minutes: int = 10) -> Dict[str, Any]:
        """Clean up stale processing locks across all documents"""
        # ... existing implementation ...
        try:
            # Get all documents with processing status
            all_documents = self.doc_repository.list_documents()
            processing_documents = [
                doc for doc in all_documents 
                if doc.get("status") == "processing"
            ]
            
            cleaned_documents = 0
            locks_removed = 0
            errors = []
            
            cutoff_time = datetime.now() - timedelta(minutes=max_lock_age_minutes)
            
            for document in processing_documents:
                document_id = document.get("document_id")
                if not document_id:
                    continue
                
                try:
                    # Also clean up stale task states
                    tasks = self.task_repository.get_tasks_by_document(document_id)
                    for task in tasks:
                        if task['task_status'] in ['running', 'pending'] and task['started_at']:
                            if task['started_at'] < cutoff_time:
                                self.task_repository.update_task_status(
                                    task['celery_task_id'],
                                    TaskStatus.FAILURE,
                                    error_details="Stale task cleanup - server restart"
                                )
                    
                    # ... existing file lock cleanup logic ...
                    doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
                    
                    # Check processing lock
                    processing_lock = os.path.join(doc_dir, "stages", "processing.lock")
                    state_lock = os.path.join(doc_dir, "stages", "state.lock")
                    
                    should_cleanup = False
                    
                    # Check if locks are stale
                    for lock_file in [processing_lock, state_lock]:
                        if self.storage_adapter.file_exists(lock_file):
                            try:
                                lock_content = self.storage_adapter.read_file(lock_file)
                                if lock_content:
                                    lock_data = json.loads(lock_content)
                                    locked_at = datetime.fromisoformat(lock_data["locked_at"])
                                    
                                    if locked_at < cutoff_time:
                                        should_cleanup = True
                                        break
                            except:
                                should_cleanup = True
                                break
                    
                    if should_cleanup:
                        # Remove stale locks
                        for lock_file in [processing_lock, state_lock]:
                            if self.storage_adapter.file_exists(lock_file):
                                self.storage_adapter.delete_file(lock_file)
                                locks_removed += 1
                        
                        # Reset document status
                        processing_state = document.get("processing_state", {})
                        current_stage = processing_state.get("current_stage", "upload")
                        
                        # Determine appropriate status based on completed stages
                        completed_stages = processing_state.get("completed_stages", [])
                        
                        if ProcessingStage.UPLOAD.value in completed_stages:
                            new_status = "failed"
                            error_message = f"Stale lock cleanup - can retry from {current_stage}"
                        else:
                            new_status = "pending"
                            error_message = "Stale lock cleanup - needs re-upload"
                        
                        self.doc_repository.update_document(document_id, {
                            "status": new_status,
                            "error_message": error_message,
                            "stale_lock_cleanup_time": datetime.now().isoformat()
                        })
                        
                        cleaned_documents += 1
                        logger.info(f"Cleaned up stale locks for document {document_id}")
                        
                except Exception as e:
                    error_msg = f"Error cleaning document {document_id}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            result = {
                "cleaned_documents": cleaned_documents,
                "locks_removed": locks_removed,
                "total_processing_documents": len(processing_documents),
                "errors": errors
            }
            
            logger.info(f"Stale lock cleanup completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during stale lock cleanup: {str(e)}")
            return {
                "cleaned_documents": 0,
                "locks_removed": 0,
                "total_processing_documents": 0,
                "errors": [str(e)]
            }
    
    def _initialize_document_processing(
        self,
        document_id: str,
        case_id: str,
        user_id: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]],
        force_restart: bool
    ) -> Dict[str, Any]:
        """Initialize document processing setup"""
        try:
            # Create document directory path
            doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
            target_filename = "original.pdf"
            target_path = os.path.join(doc_dir, target_filename)
            
            # Create the directory if it doesn't exist
            self.storage_adapter.create_directory(doc_dir)
            
            # Check if document already exists in registry
            existing_doc = self.doc_repository.get_document(document_id)
            
            storage_success = False
            stored_file_path = file_path  # Default fallback
            
            if existing_doc and not force_restart:
                # Document exists, return existing paths
                stored_file_path = existing_doc.get("stored_file_path", target_path)
                logger.info(f"Document {document_id} already exists, using existing setup")
                
                # Verify file exists
                if self.storage_adapter.file_exists(stored_file_path):
                    storage_success = True
                else:
                    logger.warning(f"Stored file not found at {stored_file_path}, will re-store")
            
            # Store the original file if needed
            if not storage_success:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                storage_success = self.storage_adapter.write_file(file_content, target_path)
                
                if not storage_success:
                    logger.warning(f"Failed to save original PDF to {target_path}")
                    stored_file_path = file_path  # Fall back to original path
                else:
                    logger.info(f"Successfully saved original PDF to {target_path}")
                    stored_file_path = target_path
            
            # Initialize processing state manager and mark upload complete
            temp_state_manager = ProcessingStateManager(
                document_id=document_id,
                storage_adapter=self.storage_adapter,
                doc_dir=doc_dir,
                doc_repository=self.doc_repository,
                case_id=case_id,
                user_id=user_id
            )
            
            # Mark upload stage as complete if file exists
            file_exists = (storage_success or 
                          self.storage_adapter.file_exists(stored_file_path) or 
                          os.path.exists(stored_file_path))
                          
            if file_exists:
                temp_state_manager.mark_stage_complete(ProcessingStage.UPLOAD.value)
                logger.info(f"Marked upload stage as complete for document {document_id}")
            else:
                logger.error(f"Cannot mark upload complete - file not accessible: {stored_file_path}")
                return {
                    "status": "error",
                    "error": f"File not accessible after upload: {stored_file_path}"
                }
            
            # Initialize or update document metadata
            doc_metadata = {
                "document_id": document_id,
                "case_id": case_id,
                "user_id": user_id,  # Add user_id to document metadata
                "original_filename": os.path.basename(file_path),
                "original_file_path": file_path,
                "stored_file_path": stored_file_path,
                "file_type": os.path.splitext(file_path)[1].lower()[1:],
                "processing_start_time": datetime.now().isoformat(),
                "status": "processing",
                "user_metadata": metadata or {},
                "processing_state": temp_state_manager.get_comprehensive_status()
            }
            
            if existing_doc:
                self.doc_repository.update_document(document_id, doc_metadata)
            else:
                self.doc_repository.add_document(doc_metadata)
            
            return {
                "status": "success",
                "doc_dir": doc_dir,
                "stored_file_path": stored_file_path
            }
            
        except Exception as e:
            logger.error(f"Error initializing document processing: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }