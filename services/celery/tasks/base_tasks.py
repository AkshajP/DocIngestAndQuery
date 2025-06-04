import logging
import time
from typing import Dict, Any, Optional
from celery import Task
from celery.exceptions import Ignore
from datetime import datetime

from core.celery_app import celery_app
from db.task_store.repository import TaskRepository, TaskStatus
from services.document.processing_state_manager import ProcessingStateManager
from services.document.storage import LocalStorageAdapter

logger = logging.getLogger(__name__)

class BaseDocumentTask(Task):
    """
    Base class for document processing tasks with pause/resume/cancel capabilities.
    """
    
    def __init__(self):
        self.task_repo = None
        self.state_manager = None
        self.current_stage = None
        self.document_id = None
        self.checkpoints = {}
        
    def before_start(self, task_id, args, kwargs):
        """Initialize task state before execution"""
        try:
            # Create fresh task repository for this task
            self.task_repo = TaskRepository()
            self.document_id = kwargs.get('document_id')
            self.current_stage = kwargs.get('stage')
            
            if self.document_id and self.current_stage:
                # Update task status to running (task should already be registered)
                try:
                    self.task_repo.update_task_status(task_id, TaskStatus.RUNNING, progress=0)
                    logger.info(f"Started task {task_id} for document {self.document_id}, stage {self.current_stage}")
                except Exception as e:
                    logger.warning(f"Could not update task status on start: {str(e)}")
                    # Don't fail the task if status update fails
                
        except Exception as e:
            logger.error(f"Error in task initialization: {str(e)}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle successful task completion"""
        try:
            if self.task_repo:
                self.task_repo.update_task_status(task_id, TaskStatus.SUCCESS, progress=100)
                logger.info(f"Task {task_id} completed successfully")
        except Exception as e:
            logger.error(f"Error marking task success: {str(e)}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        try:
            if self.task_repo:
                self.task_repo.update_task_status(
                    task_id, 
                    TaskStatus.FAILURE, 
                    error_details=str(exc)
                )
                logger.error(f"Task {task_id} failed: {str(exc)}")
        except Exception as e:
            logger.error(f"Error marking task failure: {str(e)}")
    
    def update_progress(self, progress: int, message: Optional[str] = None):
        """Update task progress"""
        try:
            if self.task_repo and self.request.id:
                metadata = {"progress_message": message} if message else None
                self.task_repo.update_task_status(
                    self.request.id, 
                    TaskStatus.RUNNING, 
                    progress=progress,
                    metadata=metadata
                )
                logger.debug(f"Updated progress to {progress}%: {message or ''}")
        except Exception as e:
            logger.debug(f"Could not update progress: {str(e)}")
    
    def check_for_cancellation(self):
        """Check if task has been cancelled"""
        try:
            if self.task_repo and self.request.id:
                task = self.task_repo.get_task_by_celery_id(self.request.id)
                if task and task['task_status'] == TaskStatus.CANCELLED.value:
                    logger.info(f"Task {self.request.id} was cancelled")
                    self.update_state(state='CANCELLED', meta={'cancelled': True})
                    raise Ignore()
        except Ignore:
            raise
        except Exception as e:
            logger.debug(f"Error checking cancellation: {str(e)}")
    
    def save_checkpoint(self, checkpoint_name: str, data: Dict[str, Any]):
        """Save checkpoint data for pause/resume"""
        try:
            if self.task_repo and self.request.id:
                checkpoint_data = {
                    checkpoint_name: {
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                self.task_repo.update_task_status(
                    self.request.id,
                    TaskStatus.RUNNING,
                    checkpoint_data=checkpoint_data
                )
                logger.debug(f"Saved checkpoint: {checkpoint_name}")
        except Exception as e:
            logger.debug(f"Could not save checkpoint: {str(e)}")
    
    def load_checkpoint(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data for resume"""
        try:
            if self.task_repo and self.request.id:
                task = self.task_repo.get_task_by_celery_id(self.request.id)
                if task and task.get('checkpoint_data'):
                    checkpoint = task['checkpoint_data'].get(checkpoint_name)
                    if checkpoint:
                        return checkpoint.get('data')
        except Exception as e:
            logger.debug(f"Could not load checkpoint: {str(e)}")
        return None
    
    def get_state_manager(self, document_id: str, case_id: str, user_id: str) -> ProcessingStateManager:
        """Get or create state manager for the document"""
        if not self.state_manager:
            storage_adapter = LocalStorageAdapter()
            doc_dir = f"document_store/{document_id}"
            
            self.state_manager = ProcessingStateManager(
                document_id=document_id,
                storage_adapter=storage_adapter,
                doc_dir=doc_dir,
                case_id=case_id,
                user_id=user_id
            )
        return self.state_manager
    
    def get_storage_adapter(self):
        """Get or create storage adapter for the task"""
        # Always create fresh storage adapter since it can't be serialized
        return LocalStorageAdapter()