import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

from db.document_store.document_tasks_repository import (
    DocumentTasksRepository, DocumentTask, TaskStatus, ProcessingStage
)

logger = logging.getLogger(__name__)

class TaskStateManager:
    """
    Manages Celery task state coordination with document processing pipeline.
    Handles task lifecycle, status transitions, and pause/resume/cancel operations.
    """
    
    def __init__(self, connection_string: Optional[str] = None, config=None):
        """Initialize task state manager with database repository"""
        if connection_string is None:
            # Try to get connection string from config
            try:
                if config:
                    config = config
                    connection_string = config.database.connection_string
            except Exception as e:
                logger.warning(f"Could not get connection string from config: {str(e)}")
                # Fall back to environment or default
                pass
        
        self.tasks_repo = DocumentTasksRepository(connection_string)
    
    def create_workflow_task(
        self, 
        document_id: str, 
        initial_stage: str = ProcessingStage.UPLOAD.value,
        celery_task_id: Optional[str] = None,
        max_retries: int = 3
    ) -> int:
        """
        Create a new document processing workflow task.
        
        Args:
            document_id: Document ID
            initial_stage: Initial processing stage
            celery_task_id: Optional Celery task ID
            max_retries: Maximum retry attempts
            
        Returns:
            Task database ID
        """
        try:
            # Check if task already exists for this document
            existing_task = self.tasks_repo.get_task_by_document_id(document_id)
            if existing_task:
                logger.warning(f"Task already exists for document {document_id}, updating instead")
                # Update existing task
                self.tasks_repo.update_task_status(
                    document_id=document_id,
                    task_status=TaskStatus.PENDING.value,
                    celery_task_id=celery_task_id
                )
                return existing_task.id
            
            # Create new task
            task = DocumentTask(
                document_id=document_id,
                current_stage=initial_stage,
                celery_task_id=celery_task_id,
                task_status=TaskStatus.PENDING.value,
                can_pause=True,
                can_resume=False,
                can_cancel=True,
                max_retries=max_retries
            )
            
            task_id = self.tasks_repo.create_task(task)
            logger.info(f"Created workflow task {task_id} for document {document_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error creating workflow task for document {document_id}: {str(e)}")
            raise
    
    def start_task(
        self, 
        document_id: str, 
        celery_task_id: str,
        worker_info: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Mark a task as started.
        
        Args:
            document_id: Document ID
            celery_task_id: Celery task ID
            worker_info: Optional worker information
            
        Returns:
            True if successful
        """
        return self.tasks_repo.update_task_status(
            document_id=document_id,
            task_status=TaskStatus.STARTED.value,
            celery_task_id=celery_task_id,
            worker_info=worker_info
        )
    
    def complete_task(
        self, 
        document_id: str, 
        success: bool = True,
        error_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a task as completed (success or failure).
        
        Args:
            document_id: Document ID
            success: Whether task completed successfully
            error_details: Optional error details for failed tasks
            
        Returns:
            True if successful
        """
        status = TaskStatus.SUCCESS.value if success else TaskStatus.FAILURE.value
        
        return self.tasks_repo.update_task_status(
            document_id=document_id,
            task_status=status,
            error_details=error_details
        )
    
    def pause_task(self, document_id: str) -> bool:
        """
        Request pause for a running task.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if pause request was set
        """
        try:
            # Check if task can be paused
            task = self.tasks_repo.get_task_by_document_id(document_id)
            if not task or not task.can_pause or task.task_status != TaskStatus.STARTED.value:
                logger.warning(f"Task {document_id} cannot be paused (status: {task.task_status if task else 'not found'})")
                return False
            
            # Set pause request flag
            success = self.tasks_repo.set_pause_request(document_id, pause=True)
            if success:
                logger.info(f"Pause requested for task {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error pausing task {document_id}: {str(e)}")
            return False
    
    def mark_task_paused(self, document_id: str) -> bool:
        """
        Mark a task as actually paused (called by the task itself).
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful
        """
        return self.tasks_repo.update_task_status(
            document_id=document_id,
            task_status=TaskStatus.PAUSED.value
        )
    
    def resume_task(self, document_id: str, celery_task_id: str) -> bool:
        """
        Resume a paused task.
        
        Args:
            document_id: Document ID
            celery_task_id: New Celery task ID for resumed task
            
        Returns:
            True if successful
        """
        try:
            # Check if task can be resumed
            task = self.tasks_repo.get_task_by_document_id(document_id)
            if not task or not task.can_resume or task.task_status != TaskStatus.PAUSED.value:
                logger.warning(f"Task {document_id} cannot be resumed (status: {task.task_status if task else 'not found'})")
                return False
            
            # Clear pause request and mark as started with new task ID
            success = self.tasks_repo.set_pause_request(document_id, pause=False)
            if success:
                success = self.tasks_repo.update_task_status(
                    document_id=document_id,
                    task_status=TaskStatus.STARTED.value,
                    celery_task_id=celery_task_id
                )
            
            if success:
                logger.info(f"Resumed task {document_id} with new Celery task {celery_task_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error resuming task {document_id}: {str(e)}")
            return False
    
    def cancel_task(self, document_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful
        """
        try:
            # Check if task can be cancelled
            task = self.tasks_repo.get_task_by_document_id(document_id)
            if not task or not task.can_cancel:
                logger.warning(f"Task {document_id} cannot be cancelled")
                return False
            
            # Set cancel request flag
            success = self.tasks_repo.set_cancel_request(document_id, cancel=True)
            if success:
                # Also update status to cancelled
                success = self.tasks_repo.update_task_status(
                    document_id=document_id,
                    task_status=TaskStatus.CANCELLED.value
                )
            
            if success:
                logger.info(f"Cancelled task {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error cancelling task {document_id}: {str(e)}")
            return False
    
    def update_progress(
        self, 
        document_id: str, 
        percent_complete: int,
        checkpoint_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update task progress and checkpoint data.
        
        Args:
            document_id: Document ID
            percent_complete: Progress percentage (0-100)
            checkpoint_data: Optional checkpoint data for resume
            
        Returns:
            True if successful
        """
        return self.tasks_repo.update_progress(
            document_id=document_id,
            percent_complete=percent_complete,
            checkpoint_data=checkpoint_data
        )
    
    def advance_stage(self, document_id: str, new_stage: str) -> bool:
        """
        Advance task to the next processing stage.
        
        Args:
            document_id: Document ID
            new_stage: New processing stage
            
        Returns:
            True if successful
        """
        try:
            task = self.tasks_repo.get_task_by_document_id(document_id)
            if not task:
                logger.error(f"Task not found for document {document_id}")
                return False
            
            # Update the current stage in the task metadata
            stage_metadata = task.stage_metadata.copy()
            stage_metadata['current_stage'] = new_stage
            stage_metadata['stage_advanced_at'] = datetime.now().isoformat()
            
            # Update task with new stage - we'll need to add this method to the repository
            # For now, use the checkpoint data field to track this
            return self.update_progress(
                document_id=document_id,
                percent_complete=task.percent_complete,
                checkpoint_data={
                    **task.checkpoint_data,
                    'current_stage': new_stage,
                    'stage_metadata': stage_metadata
                }
            )
            
        except Exception as e:
            logger.error(f"Error advancing stage for document {document_id}: {str(e)}")
            return False
    
    def get_task_status(self, document_id: str) -> Optional[DocumentTask]:
        """
        Get current task status.
        
        Args:
            document_id: Document ID
            
        Returns:
            DocumentTask instance or None
        """
        return self.tasks_repo.get_task_by_document_id(document_id)
    
    def check_pause_requested(self, document_id: str) -> bool:
        """
        Check if pause has been requested for a task.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if pause requested
        """
        task = self.tasks_repo.get_task_by_document_id(document_id)
        return task.pause_requested if task else False
    
    def check_cancel_requested(self, document_id: str) -> bool:
        """
        Check if cancellation has been requested for a task.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if cancel requested
        """
        task = self.tasks_repo.get_task_by_document_id(document_id)
        return task.cancel_requested if task else False
    
    def get_checkpoint_data(self, document_id: str) -> Dict[str, Any]:
        """
        Get checkpoint data for a task.
        
        Args:
            document_id: Document ID
            
        Returns:
            Checkpoint data dictionary
        """
        task = self.tasks_repo.get_task_by_document_id(document_id)
        return task.checkpoint_data if task else {}
    
    def cleanup_old_tasks(self, days_old: int = 7) -> int:
        """
        Clean up old completed/failed tasks.
        
        Args:
            days_old: Remove tasks older than this many days
            
        Returns:
            Number of tasks cleaned up
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Get old completed/failed tasks
            completed_tasks = self.tasks_repo.get_tasks_by_status(TaskStatus.SUCCESS.value, limit=1000)
            failed_tasks = self.tasks_repo.get_tasks_by_status(TaskStatus.FAILURE.value, limit=1000)
            cancelled_tasks = self.tasks_repo.get_tasks_by_status(TaskStatus.CANCELLED.value, limit=1000)
            
            cleaned_count = 0
            for task_list in [completed_tasks, failed_tasks, cancelled_tasks]:
                for task in task_list:
                    if task.completed_at and task.completed_at < cutoff_date:
                        if self.tasks_repo.delete_task(task.document_id):
                            cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old tasks")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old tasks: {str(e)}")
            return 0
    
    def get_active_tasks(self) -> List[DocumentTask]:
        """
        Get all currently active tasks.
        
        Returns:
            List of active DocumentTask instances
        """
        try:
            active_statuses = [
                TaskStatus.PENDING.value,
                TaskStatus.STARTED.value,
                TaskStatus.PAUSED.value
            ]
            
            active_tasks = []
            for status in active_statuses:
                tasks = self.tasks_repo.get_tasks_by_status(status)
                active_tasks.extend(tasks)
            
            return active_tasks
            
        except Exception as e:
            logger.error(f"Error getting active tasks: {str(e)}")
            return []