import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from core.config import get_config
from db.document_store.repository import DocumentMetadataRepository
from services.document.storage import LocalStorageAdapter, S3StorageAdapter
from .processing_state_manager import ProcessingStateManager, ProcessingStage
from .stage_processors import (
    ExtractionProcessor, ChunkingProcessor, EmbeddingProcessor,
    TreeBuildingProcessor, VectorStorageProcessor
)

logger = logging.getLogger(__name__)

class PersistentUploadService:
    """
    Service for persistent document upload and processing.
    Manages document processing through stages with retry capabilities.
    Now integrates with Celery task management for pause/resume/cancel support.
    """
    
    def __init__(self, config=None):
        """Initialize the persistent upload service"""
        self.config = config or get_config()
        
        # Initialize storage adapter
        self.storage_adapter = self._initialize_storage_adapter()
        
        # Initialize document metadata repository
        self.doc_repository = DocumentMetadataRepository()
        
        # Initialize Celery task state manager
        self.task_state_manager = None
        try:
            from services.celery.task_state_manager import TaskStateManager
            self.task_state_manager = TaskStateManager()
            logger.info("Initialized Celery task state manager")
        except ImportError:
            logger.warning("Celery task state manager not available - running without task management")
        
        # Initialize stage processors
        self.processors = {
            ProcessingStage.EXTRACTION.value: ExtractionProcessor(self.config),
            ProcessingStage.CHUNKING.value: ChunkingProcessor(self.config),
            ProcessingStage.EMBEDDING.value: EmbeddingProcessor(self.config),
            ProcessingStage.TREE_BUILDING.value: TreeBuildingProcessor(self.config),
            ProcessingStage.VECTOR_STORAGE.value: VectorStorageProcessor(self.config)
        }
        
        self._cleanup_stale_locks_on_startup()
        
        logger.info("Initialized persistent upload service with task management")
    
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
        metadata: Optional[Dict[str, Any]] = None,
        force_restart: bool = False,
        celery_task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload and process a document through the complete pipeline with persistence.
        Now integrates with Celery task management.
        
        Args:
            file_path: Path to the document file
            document_id: Optional custom ID (generated if not provided)
            case_id: Case ID for document grouping (required)
            metadata: Optional metadata about the document
            force_restart: If True, restart processing from the beginning
            celery_task_id: Optional Celery task ID for tracking
            
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
            # Create workflow task in Celery task manager
            if self.task_state_manager:
                try:
                    task_id = self.task_state_manager.create_workflow_task(
                        document_id=document_id,
                        initial_stage=ProcessingStage.UPLOAD.value,
                        celery_task_id=celery_task_id
                    )
                    logger.info(f"Created Celery workflow task {task_id} for document {document_id}")
                except Exception as e:
                    logger.warning(f"Failed to create Celery task: {str(e)} - proceeding without task management")
            
            # Check if document is already being processed by another operation
            existing_doc = self.doc_repository.get_document(document_id)
            if existing_doc and existing_doc.get("status") == "processing" and not force_restart:
                # Check if there's an active lock file
                doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
                lock_file = os.path.join(doc_dir, "stages", "processing.lock")
                
                if self.storage_adapter.file_exists(lock_file):
                    # Check if lock is recent (within last 5 minutes)
                    try:
                        lock_content = self.storage_adapter.read_file(lock_file)
                        if lock_content:
                            lock_data = json.loads(lock_content)
                            locked_at = datetime.fromisoformat(lock_data["locked_at"])
                            if (datetime.now() - locked_at).total_seconds() < 300:  # 5 minutes
                                return {
                                    "status": "error",
                                    "document_id": document_id,
                                    "error": "Document is already being processed by another operation",
                                    "processing_time": 0
                                }
                    except:
                        pass  # If we can't read the lock, proceed
            
            # Initialize or load existing processing state
            result = self._initialize_document_processing(
                document_id, case_id, file_path, metadata, force_restart
            )
            
            if result["status"] == "error":
                return result
            
            doc_dir = result["doc_dir"]
            stored_file_path = result["stored_file_path"]
            
            # Initialize processing state manager with document repository AND task state manager
            state_manager = ProcessingStateManager(
                document_id=document_id,
                storage_adapter=self.storage_adapter,
                doc_dir=doc_dir,
                doc_repository=self.doc_repository,
                task_state_manager=self.task_state_manager  # NEW: Pass task state manager
            )
            
            # Reset processing if force_restart is True
            if force_restart:
                state_manager.reset_to_stage(ProcessingStage.UPLOAD.value)
                # Mark upload as complete again after reset since file is already stored
                state_manager.mark_stage_complete(ProcessingStage.UPLOAD.value, celery_task_id)
                logger.info(f"Force restarting processing for document {document_id}")
            
            # Start the task in Celery task manager
            if self.task_state_manager and celery_task_id:
                worker_info = {
                    "worker_id": os.getenv("CELERY_WORKER_NAME", "unknown"),
                    "worker_hostname": os.getenv("HOSTNAME", "unknown")
                }
                self.task_state_manager.start_task(document_id, celery_task_id, worker_info)
            
            # Create processing lock
            self._create_processing_lock(doc_dir)
            
            # Create processing context
            context = {
                "document_id": document_id,
                "case_id": case_id,
                "file_path": file_path,
                "stored_file_path": stored_file_path,
                "doc_dir": doc_dir,
                "storage_adapter": self.storage_adapter,
                "metadata": metadata or {},
                "celery_task_id": celery_task_id
            }
            
            # Execute processing pipeline with pause/cancel support
            pipeline_result = self._execute_processing_pipeline_with_control(state_manager, context)
            
            # Calculate total processing time
            total_processing_time = time.time() - process_start_time
            
            # Update final document metadata
            final_metadata = {
                "total_processing_time": total_processing_time,
                "processing_state": state_manager.get_stage_status()
            }
            
            if pipeline_result["status"] == "success":
                final_metadata.update({
                    "status": "processed",
                    "processing_date": datetime.now().isoformat()
                })
                
                # Mark processing as completed in Celery task manager
                if self.task_state_manager:
                    self.task_state_manager.complete_task(document_id, success=True)
                
                state_manager.mark_stage_complete(ProcessingStage.COMPLETED.value, celery_task_id)
            elif pipeline_result["status"] == "paused":
                final_metadata.update({
                    "status": "paused",
                    "pause_stage": pipeline_result.get("paused_stage"),
                    "pause_time": datetime.now().isoformat()
                })
                # Task state manager handles pause status internally
            elif pipeline_result["status"] == "cancelled":
                final_metadata.update({
                    "status": "cancelled",
                    "cancel_stage": pipeline_result.get("cancelled_stage"),
                    "cancel_time": datetime.now().isoformat()
                })
                # Task state manager handles cancel status internally
            else:
                final_metadata.update({
                    "status": "failed",
                    "failure_stage": pipeline_result.get("failed_stage"),
                    "error_message": pipeline_result.get("error"),
                    "failure_time": datetime.now().isoformat()
                })
                
                # Mark as failed in Celery task manager
                if self.task_state_manager:
                    self.task_state_manager.complete_task(
                        document_id, 
                        success=False,
                        error_details={"error": pipeline_result.get("error")}
                    )
            
            self.doc_repository.update_document(document_id, final_metadata)
            
            # Remove processing lock
            self._remove_processing_lock(doc_dir)
            
            # Prepare response
            response = {
                "status": pipeline_result["status"],
                "document_id": document_id,
                "case_id": case_id,
                "processing_time": total_processing_time,
                "stored_file_path": stored_file_path,
                "doc_dir": doc_dir,
                "processing_state": state_manager.get_stage_status()
            }
            
            if pipeline_result["status"] == "success":
                response.update({
                    "chunks_count": pipeline_result.get("chunks_count", 0),
                    "tree_nodes_count": pipeline_result.get("tree_nodes_count", 0),
                    "raptor_levels": pipeline_result.get("raptor_levels", [])
                })
            elif pipeline_result["status"] in ["paused", "cancelled"]:
                response.update({
                    "control_action": pipeline_result["status"],
                    f"{pipeline_result['status']}_stage": pipeline_result.get(f"{pipeline_result['status']}_stage")
                })
            else:
                response.update({
                    "error": pipeline_result.get("error"),
                    "failed_stage": pipeline_result.get("failed_stage")
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            
            # Mark as failed in Celery task manager
            if self.task_state_manager:
                self.task_state_manager.complete_task(
                    document_id,
                    success=False,
                    error_details={"error": str(e), "stage": "initialization"}
                )
            
            # Remove processing lock on error
            try:
                doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
                self._remove_processing_lock(doc_dir)
            except:
                pass
            
            # Update document with error status
            self.doc_repository.update_document(document_id, {
                "status": "failed",
                "failure_stage": "initialization",
                "error_message": str(e),
                "failure_time": datetime.now().isoformat()
            })
            
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e),
                "processing_time": time.time() - process_start_time
            }
    
    def pause_document_processing(self, document_id: str) -> Dict[str, Any]:
        """
        Pause document processing.
        
        Args:
            document_id: Document ID to pause
            
        Returns:
            Dictionary with pause status
        """
        try:
            if not self.task_state_manager:
                return {
                    "status": "error",
                    "error": "Task management not available"
                }
            
            success = self.task_state_manager.pause_task(document_id)
            
            if success:
                # Update document repository
                self.doc_repository.mark_task_as_paused(document_id)
                return {
                    "status": "success",
                    "message": f"Pause requested for document {document_id}"
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to pause task"
                }
                
        except Exception as e:
            logger.error(f"Error pausing document {document_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def resume_document_processing(
        self, 
        document_id: str, 
        celery_task_id: str
    ) -> Dict[str, Any]:
        """
        Resume paused document processing.
        
        Args:
            document_id: Document ID to resume
            celery_task_id: New Celery task ID for resumed processing
            
        Returns:
            Dictionary with resume status
        """
        try:
            if not self.task_state_manager:
                return {
                    "status": "error",
                    "error": "Task management not available"
                }
            
            success = self.task_state_manager.resume_task(document_id, celery_task_id)
            
            if success:
                # Update document repository
                self.doc_repository.mark_task_as_resumed(document_id)
                return {
                    "status": "success",
                    "message": f"Resumed processing for document {document_id}"
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to resume task"
                }
                
        except Exception as e:
            logger.error(f"Error resuming document {document_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def cancel_document_processing(self, document_id: str) -> Dict[str, Any]:
        """
        Cancel document processing.
        
        Args:
            document_id: Document ID to cancel
            
        Returns:
            Dictionary with cancel status
        """
        try:
            if not self.task_state_manager:
                return {
                    "status": "error",
                    "error": "Task management not available"
                }
            
            success = self.task_state_manager.cancel_task(document_id)
            
            if success:
                # Update document repository
                self.doc_repository.mark_task_as_cancelled(document_id)
                return {
                    "status": "success",
                    "message": f"Cancelled processing for document {document_id}"
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to cancel task"
                }
                
        except Exception as e:
            logger.error(f"Error cancelling document {document_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _execute_processing_pipeline_with_control(
        self,
        state_manager: ProcessingStateManager,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the processing pipeline with pause/resume/cancel control.
        Enhanced version of the original pipeline execution.
        
        Args:
            state_manager: Processing state manager with task control
            context: Processing context
            
        Returns:
            Dictionary with processing results or control actions
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
        start_index = 0
        if current_stage in stage_order:
            start_index = stage_order.index(current_stage)
        elif current_stage == ProcessingStage.UPLOAD.value:
            # If current stage is upload and it's complete, start from extraction
            if state_manager.is_stage_complete(ProcessingStage.UPLOAD.value):
                start_index = 0  # Start from extraction
                # Advance to extraction stage
                state_manager.processing_state["current_stage"] = ProcessingStage.EXTRACTION.value
                state_manager._save_processing_state()
                logger.info(f"Advanced from upload to extraction for document {context['document_id']}")
            else:
                return {
                    "status": "error",
                    "error": "Upload stage not completed",
                    "failed_stage": ProcessingStage.UPLOAD.value
                }
        
        logger.info(f"Starting processing pipeline for document {context['document_id']} from stage {current_stage}")
        
        # Execute stages in order with control checks
        last_result = {}
        for i in range(start_index, len(stage_order)):
            stage = stage_order[i]
            
            # Check for cancellation before each stage
            if state_manager.check_cancel_requested():
                logger.info(f"Cancellation requested for document {context['document_id']} at stage {stage}")
                return {
                    "status": "cancelled",
                    "cancelled_stage": stage,
                    "message": "Processing cancelled by user request"
                }
            
            # Check for pause before each stage
            if state_manager.check_pause_requested():
                logger.info(f"Pause requested for document {context['document_id']} at stage {stage}")
                # Save checkpoint for current stage
                checkpoint_data = {
                    "stage": stage,
                    "timestamp": datetime.now().isoformat(),
                    "percent_complete": int((i / len(stage_order)) * 100)
                }
                state_manager.handle_pause_request(stage, checkpoint_data)
                return {
                    "status": "paused", 
                    "paused_stage": stage,
                    "message": "Processing paused by user request"
                }
            
            # Skip completed stages
            if stage in completed_stages:
                logger.info(f"Skipping completed stage: {stage}")
                continue
            
            # AUTO-ADVANCE: Try to advance to this stage if needed
            if not self._advance_to_next_stage_if_needed(state_manager, stage):
                # If we can't advance and stage isn't current, there's a problem
                if state_manager.get_current_stage() != stage:
                    error_msg = f"Cannot advance to stage {stage} from {state_manager.get_current_stage()}"
                    logger.error(error_msg)
                    return {
                        "status": "error",
                        "error": error_msg,
                        "failed_stage": stage
                    }
            
            processor = self.processors.get(stage)
            if not processor:
                error_msg = f"No processor found for stage: {stage}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "failed_stage": stage
                }
            
            # Now the stage should be ready - simpler check
            if not processor.can_execute(state_manager, context):
                error_msg = f"Stage {stage} processor reports it cannot execute"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "failed_stage": stage
                }
            
            # Validate dependencies
            if not processor.validate_dependencies(state_manager, context):
                error_msg = f"Dependencies not met for stage {stage}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "failed_stage": stage
                }
            
            # Execute stage with periodic control checks
            logger.info(f"Executing stage: {stage}")
            result = self._execute_stage_with_control_checks(processor, state_manager, context, stage)
            
            if result["status"] == "cancelled":
                return result
            elif result["status"] == "paused":
                return result
            elif result["status"] != "success":
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
    
    def _execute_stage_with_control_checks(
        self,
        processor,
        state_manager: ProcessingStateManager,
        context: Dict[str, Any],
        stage: str
    ) -> Dict[str, Any]:
        """
        Execute a stage with periodic pause/cancel checks.
        
        Args:
            processor: Stage processor instance
            state_manager: Processing state manager
            context: Processing context
            stage: Current stage name
            
        Returns:
            Stage execution result
        """
        try:
            # Register Celery task for this stage
            celery_task_id = context.get("celery_task_id")
            if celery_task_id:
                state_manager.register_celery_task(stage, celery_task_id)
            
            # Execute the stage processor
            result = processor.execute(state_manager, context)
            
            # Final check for control requests after execution
            if state_manager.check_cancel_requested():
                return {
                    "status": "cancelled",
                    "cancelled_stage": stage,
                    "message": "Processing cancelled during stage execution"
                }
            
            if state_manager.check_pause_requested():
                # If stage completed but pause was requested, we still pause
                checkpoint_data = {
                    "stage": stage,
                    "completed": result["status"] == "success",
                    "timestamp": datetime.now().isoformat()
                }
                state_manager.handle_pause_request(stage, checkpoint_data)
                return {
                    "status": "paused",
                    "paused_stage": stage,
                    "stage_result": result,
                    "message": "Processing paused after stage completion"
                }
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing stage {stage}: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "failed_stage": stage
            }
    
    def retry_document_processing(
        self,
        document_id: str,
        from_stage: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retry document processing from a specific stage or current failed stage.
        Enhanced with task state management.
        """
        try:
            # Check for concurrent operations
            existing_doc = self.doc_repository.get_document(document_id)
            if not existing_doc:
                return {
                    "status": "error",
                    "error": "Document not found"
                }
            
            if existing_doc.get("status") == "processing":
                doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
                lock_file = os.path.join(doc_dir, "stages", "processing.lock")
                
                if self.storage_adapter.file_exists(lock_file):
                    try:
                        lock_content = self.storage_adapter.read_file(lock_file)
                        if lock_content:
                            lock_data = json.loads(lock_content)
                            locked_at = datetime.fromisoformat(lock_data["locked_at"])
                            if (datetime.now() - locked_at).total_seconds() < 300:  # 5 minutes
                                return {
                                    "status": "error",
                                    "error": "Document is already being processed. Please wait."
                                }
                    except:
                        pass  # If we can't read lock, proceed
            
            # Create new workflow task for retry
            if self.task_state_manager:
                task_id = self.task_state_manager.create_workflow_task(
                    document_id=document_id,
                    initial_stage=from_stage or ProcessingStage.UPLOAD.value
                )
                logger.info(f"Created retry workflow task {task_id} for document {document_id}")
            
            # Get document metadata
            document = self.doc_repository.get_document(document_id)
            if not document:
                return {
                    "status": "error",
                    "error": "Document not found"
                }
            
            # Verify case_id if provided
            if case_id and document.get("case_id") != case_id:
                return {
                    "status": "error",
                    "error": "Document does not belong to specified case"
                }
            
            # Get document paths
            file_path = document.get("stored_file_path") or document.get("original_file_path")
            if not file_path or not os.path.exists(file_path):
                return {
                    "status": "error",
                    "error": "Original file not found"
                }
            
            doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
            
            # Initialize processing state manager with coordination AND task state manager
            state_manager = ProcessingStateManager(
                document_id=document_id,
                storage_adapter=self.storage_adapter,
                doc_dir=doc_dir,
                doc_repository=self.doc_repository,
                task_state_manager=self.task_state_manager  # NEW: Pass task state manager
            )
            
            # Reset to specified stage if provided
            if from_stage:
                if not self._is_valid_stage(from_stage):
                    return {
                        "status": "error",
                        "error": f"Invalid stage: {from_stage}"
                    }
                state_manager.reset_to_stage(from_stage)
                
                # If resetting to upload, mark it as complete since file is already stored
                if from_stage == ProcessingStage.UPLOAD.value:
                    state_manager.mark_stage_complete(ProcessingStage.UPLOAD.value)
                    
                logger.info(f"Reset document {document_id} to stage {from_stage}")
            
            # Create processing lock
            self._create_processing_lock(doc_dir)
            
            # Update document status to processing
            self.doc_repository.update_document(document_id, {
                "status": "processing",
                "retry_time": datetime.now().isoformat(),
                "retry_count": document.get("retry_count", 0) + 1
            })
            
            # Create processing context
            context = {
                "document_id": document_id,
                "case_id": document.get("case_id", "default"),
                "file_path": file_path,
                "stored_file_path": file_path,
                "doc_dir": doc_dir,
                "storage_adapter": self.storage_adapter,
                "metadata": document.get("user_metadata", {})
            }
            
            # Execute processing pipeline with control support
            pipeline_result = self._execute_processing_pipeline_with_control(state_manager, context)
            
            # Update document metadata based on result
            if pipeline_result["status"] == "success":
                self.doc_repository.update_document(document_id, {
                    "status": "processed",
                    "processing_date": datetime.now().isoformat(),
                    "processing_state": state_manager.get_stage_status()
                })
                state_manager.mark_stage_complete(ProcessingStage.COMPLETED.value)
                
                # Mark as completed in task manager
                if self.task_state_manager:
                    self.task_state_manager.complete_task(document_id, success=True)
                    
            elif pipeline_result["status"] in ["paused", "cancelled"]:
                self.doc_repository.update_document(document_id, {
                    "status": pipeline_result["status"],
                    f"{pipeline_result['status']}_stage": pipeline_result.get(f"{pipeline_result['status']}_stage"),
                    f"{pipeline_result['status']}_time": datetime.now().isoformat(),
                    "processing_state": state_manager.get_stage_status()
                })
            else:
                self.doc_repository.update_document(document_id, {
                    "status": "failed",
                    "failure_stage": pipeline_result.get("failed_stage"),
                    "error_message": pipeline_result.get("error"),
                    "failure_time": datetime.now().isoformat(),
                    "processing_state": state_manager.get_stage_status()
                })
                
                # Mark as failed in task manager
                if self.task_state_manager:
                    self.task_state_manager.complete_task(
                        document_id,
                        success=False,
                        error_details={"error": pipeline_result.get("error")}
                    )
            
            # Remove processing lock
            self._remove_processing_lock(doc_dir)
            
            return {
                "status": pipeline_result["status"],
                "document_id": document_id,
                "processing_state": state_manager.get_stage_status(),
                **pipeline_result
            }
            
        except Exception as e:
            logger.error(f"Error retrying document {document_id}: {str(e)}")
            
            # Remove processing lock on error
            try:
                doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
                self._remove_processing_lock(doc_dir)
            except:
                pass
                
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e)
            }
    
    def get_document_processing_status(self, document_id: str) -> Dict[str, Any]:
        """Get detailed processing status for a document with task control information"""
        try:
            # Get document metadata
            document = self.doc_repository.get_document(document_id)
            if not document:
                return {
                    "status": "error",
                    "error": "Document not found"
                }
            
            doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
            
            # Initialize processing state manager with task state manager
            state_manager = ProcessingStateManager(
                document_id=document_id,
                storage_adapter=self.storage_adapter,
                doc_dir=doc_dir,
                doc_repository=self.doc_repository,
                task_state_manager=self.task_state_manager
            )
            
            # Get comprehensive status including task control information
            processing_status = state_manager.get_stage_status()
            
            # Add task control status if available
            task_control_status = {}
            if self.task_state_manager:
                task = self.task_state_manager.get_task_status(document_id)
                if task:
                    task_control_status = {
                        "task_status": task.task_status,
                        "can_pause": task.can_pause,
                        "can_resume": task.can_resume,
                        "can_cancel": task.can_cancel,
                        "pause_requested": task.pause_requested,
                        "cancel_requested": task.cancel_requested,
                        "percent_complete": task.percent_complete,
                        "worker_info": {
                            "worker_id": task.worker_id,
                            "worker_hostname": task.worker_hostname
                        },
                        "celery_task_id": task.celery_task_id,
                        "retry_count": task.retry_count,
                        "max_retries": task.max_retries
                    }
            
            return {
                "status": "success",
                "document_metadata": document,
                "processing_status": processing_status,
                "task_control_status": task_control_status
            }
            
        except Exception as e:
            logger.error(f"Error getting processing status for {document_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    # Include all other existing methods from the original class...
    # (keeping the rest of the methods unchanged for brevity)
    
    def _initialize_document_processing(self, document_id: str, case_id: str, file_path: str, metadata: Optional[Dict[str, Any]], force_restart: bool) -> Dict[str, Any]:
        """Initialize document processing setup (unchanged)"""
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
                task_state_manager=self.task_state_manager
            )
            
            # Mark upload stage as complete if file exists (either stored or original)
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
                "original_filename": os.path.basename(file_path),
                "original_file_path": file_path,
                "stored_file_path": stored_file_path,
                "file_type": os.path.splitext(file_path)[1].lower()[1:],
                "processing_start_time": datetime.now().isoformat(),
                "status": "processing",
                "user_metadata": metadata or {},
                "processing_state": temp_state_manager.get_stage_status()
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
    
    def _is_valid_stage(self, stage: str) -> bool:
        """Check if a stage name is valid"""
        valid_stages = [s.value for s in ProcessingStage]
        return stage in valid_stages
    
    def _advance_to_next_stage_if_needed(self, state_manager: ProcessingStateManager, target_stage: str) -> bool:
        """Advance processing state to target stage if prerequisites are met (unchanged)"""
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
    
    def _create_processing_lock(self, doc_dir: str):
        """Create a processing lock file to prevent concurrent operations (unchanged)"""
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
        """Remove the processing lock file (unchanged)"""
        try:
            lock_file = os.path.join(doc_dir, "stages", "processing.lock")
            if self.storage_adapter.file_exists(lock_file):
                self.storage_adapter.delete_file(lock_file)
                logger.debug(f"Removed processing lock: {lock_file}")
        except Exception as e:
            logger.warning(f"Failed to remove processing lock: {str(e)}")
            
    def _cleanup_stale_locks_on_startup(self):
        """Clean up stale locks when service starts (unchanged)"""
        try:
            cleanup_result = self.cleanup_stale_locks()
            if cleanup_result["cleaned_documents"] > 0:
                logger.info(f"Startup cleanup: Reset {cleanup_result['cleaned_documents']} documents and removed {cleanup_result['locks_removed']} stale locks")
        except Exception as e:
            logger.error(f"Error during startup lock cleanup: {str(e)}")

    def cleanup_stale_locks(self, max_lock_age_minutes: int = 10) -> Dict[str, Any]:
        """Clean up stale processing locks across all documents (unchanged)"""
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
                                # If we can't read the lock, consider it stale
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
                            # If upload was completed, mark as failed so it can be retried from current stage
                            new_status = "failed"
                            error_message = f"Stale lock cleanup - can retry from {current_stage}"
                        else:
                            # If upload wasn't even completed, mark as pending
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