import logging
from typing import Dict, Any, List, Optional
from celery import chain, chord, group
from celery.result import AsyncResult

from core.celery_app import celery_app
from .tasks.document_tasks import (
    extract_document_task, chunk_document_task, embed_document_task,
    build_tree_task, store_vectors_task
)
from services.document.processing_state_manager import ProcessingStage
from db.task_store.repository import TaskRepository, TaskStatus

logger = logging.getLogger(__name__)

class DocumentTaskOrchestrator:
    """
    Orchestrates document processing tasks with Celery chains.
    """
    
    def __init__(self):
        self.task_repo = TaskRepository()
        
        # Task mapping for each stage
        self.stage_tasks = {
            ProcessingStage.EXTRACTION.value: extract_document_task,
            ProcessingStage.CHUNKING.value: chunk_document_task,
            ProcessingStage.EMBEDDING.value: embed_document_task,
            ProcessingStage.TREE_BUILDING.value: build_tree_task,
            ProcessingStage.VECTOR_STORAGE.value: store_vectors_task
        }
    
    def create_processing_chain(
        self,
        document_id: str,
        case_id: str,
        user_id: str,
        context: Dict[str, Any],
        start_from_stage: Optional[str] = None
    ) -> AsyncResult:
        """
        Create a Celery chain for document processing.
        
        Args:
            document_id: Document ID
            case_id: Case ID
            user_id: User ID
            context: Processing context
            start_from_stage: Optional stage to start from
            
        Returns:
            AsyncResult for the chain
        """
        
        # Define stage order
        stage_order = [
            ProcessingStage.EXTRACTION.value,
            ProcessingStage.CHUNKING.value,
            ProcessingStage.EMBEDDING.value,
            ProcessingStage.TREE_BUILDING.value,
            ProcessingStage.VECTOR_STORAGE.value
        ]
        
        # Find starting point
        start_index = 0
        if start_from_stage and start_from_stage in stage_order:
            start_index = stage_order.index(start_from_stage)
        
        # Clean context for Celery serialization (remove non-serializable objects)
        celery_context = self._prepare_context_for_celery(context)
        
        # Create task signatures for the chain
        task_signatures = []
        
        for stage in stage_order[start_index:]:
            task_func = self.stage_tasks.get(stage)
            if task_func:
                signature = task_func.s(
                    document_id=document_id,
                    case_id=case_id,
                    user_id=user_id,
                    stage=stage,
                    context=celery_context
                )
                task_signatures.append(signature)
        
        if not task_signatures:
            raise ValueError("No valid tasks found for processing")
        
        # Create and start the chain
        processing_chain = chain(*task_signatures)
        result = processing_chain.apply_async()
        
        logger.info(f"Created processing chain for document {document_id} with {len(task_signatures)} tasks")
        return result
    
    def _prepare_context_for_celery(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context for Celery by removing non-serializable objects.
        
        Args:
            context: Original context
            
        Returns:
            Cleaned context safe for JSON serialization
        """
        # List of keys to exclude (non-serializable objects)
        exclude_keys = {
            'storage_adapter',  # Will be recreated in task
            'task_instance',    # Task-specific
            'celery_task_id'    # Will be set by task
        }
        
        # Create clean context
        celery_context = {}
        excluded_keys = []
        
        for key, value in context.items():
            if key in exclude_keys:
                excluded_keys.append(key)
                continue
                
            # Only include JSON-serializable values
            try:
                import json
                json.dumps(value)  # Test if value is serializable
                celery_context[key] = value
            except (TypeError, ValueError) as e:
                excluded_keys.append(f"{key} ({type(value).__name__})")
                logger.debug(f"Excluding non-serializable context key: {key} of type {type(value).__name__}: {str(e)}")
        
        if excluded_keys:
            logger.info(f"Excluded non-serializable keys from Celery context: {excluded_keys}")
        
        logger.debug(f"Celery context prepared with keys: {list(celery_context.keys())}")
        return celery_context
    
    def submit_single_task(
        self,
        stage: str,
        document_id: str,
        case_id: str,
        user_id: str,
        context: Dict[str, Any]
    ) -> Optional[AsyncResult]:
        """
        Submit a single task for a specific stage.
        
        Args:
            stage: Processing stage
            document_id: Document ID
            case_id: Case ID
            user_id: User ID
            context: Processing context
            
        Returns:
            AsyncResult or None if stage not found
        """
        
        task_func = self.stage_tasks.get(stage)
        if not task_func:
            logger.error(f"No task function found for stage: {stage}")
            return None
        
        # Clean context for Celery serialization
        celery_context = self._prepare_context_for_celery(context)
        
        result = task_func.apply_async(
            kwargs={
                'document_id': document_id,
                'case_id': case_id,
                'user_id': user_id,
                'stage': stage,
                'context': celery_context
            }
        )
        
        logger.info(f"Submitted task for document {document_id}, stage {stage}")
        return result
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a Celery task"""
        try:
            result = AsyncResult(task_id, app=celery_app)
            
            return {
                "task_id": task_id,
                "status": result.status,
                "result": result.result,
                "info": result.info,
                "successful": result.successful(),
                "failed": result.failed()
            }
        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {str(e)}")
            return {
                "task_id": task_id,
                "status": "UNKNOWN",
                "error": str(e)
            }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a Celery task"""
        try:
            result = AsyncResult(task_id, app=celery_app)
            result.revoke(terminate=True)
            
            # Update in database
            self.task_repo.update_task_status(
                task_id,
                TaskStatus.CANCELLED,
                error_details="Task cancelled by user request"
            )
            
            logger.info(f"Cancelled task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {str(e)}")
            return False
    
    def get_document_tasks_status(self, document_id: str) -> Dict[str, Any]:
        """Get status of all tasks for a document"""
        try:
            # Get tasks from database
            db_tasks = self.task_repo.get_tasks_by_document(document_id)
            
            task_statuses = {}
            
            for db_task in db_tasks:
                celery_task_id = db_task['celery_task_id']
                stage = db_task['processing_stage']
                
                # Get Celery task status
                celery_status = self.get_task_status(celery_task_id)
                
                # Combine database and Celery information
                task_statuses[stage] = {
                    **db_task,
                    "celery_status": celery_status
                }
            
            return {
                "document_id": document_id,
                "tasks": task_statuses,
                "total_tasks": len(task_statuses)
            }
            
        except Exception as e:
            logger.error(f"Error getting document tasks status: {str(e)}")
            return {
                "document_id": document_id,
                "tasks": {},
                "error": str(e)
            }

class TaskControlManager:
    """
    Manages task control operations (pause/resume/cancel).
    """
    
    def __init__(self):
        self.task_repo = TaskRepository()
        self.orchestrator = DocumentTaskOrchestrator()
    
    def pause_document_tasks(self, document_id: str, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Pause document processing tasks.
        
        Note: Actual pause implementation requires worker cooperation.
        This marks tasks as paused in the database.
        """
        try:
            active_tasks = self.task_repo.get_tasks_by_document(document_id)
            
            if stage:
                active_tasks = [t for t in active_tasks if t['processing_stage'] == stage]
            
            # Filter to only pausable tasks
            pausable_tasks = [t for t in active_tasks if t.get('can_pause', False)]
            
            paused_count = 0
            for task in pausable_tasks:
                success = self.task_repo.update_task_status(
                    task['celery_task_id'],
                    TaskStatus.PAUSED,
                    checkpoint_data={"paused_by_user": True}
                )
                if success:
                    paused_count += 1
            
            return {
                "status": "success" if paused_count > 0 else "no_action",
                "paused_tasks": paused_count,
                "message": f"Paused {paused_count} tasks"
            }
            
        except Exception as e:
            logger.error(f"Error pausing tasks: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def resume_document_tasks(self, document_id: str, stage: Optional[str] = None) -> Dict[str, Any]:
        """Resume paused document processing tasks."""
        try:
            paused_tasks = self.task_repo.get_tasks_by_document(document_id)
            paused_tasks = [t for t in paused_tasks if t['task_status'] == 'paused']
            
            if stage:
                paused_tasks = [t for t in paused_tasks if t['processing_stage'] == stage]
            
            resumed_count = 0
            for task in paused_tasks:
                if task.get('can_resume', False):
                    success = self.task_repo.update_task_status(
                        task['celery_task_id'],
                        TaskStatus.RUNNING
                    )
                    if success:
                        resumed_count += 1
            
            return {
                "status": "success" if resumed_count > 0 else "no_action",
                "resumed_tasks": resumed_count,
                "message": f"Resumed {resumed_count} tasks"
            }
            
        except Exception as e:
            logger.error(f"Error resuming tasks: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def cancel_document_tasks(self, document_id: str, stage: Optional[str] = None) -> Dict[str, Any]:
        """Cancel document processing tasks."""
        try:
            active_tasks = self.task_repo.get_tasks_by_document(document_id)
            active_tasks = [t for t in active_tasks if t['task_status'] in ['pending', 'running', 'paused']]
            
            if stage:
                active_tasks = [t for t in active_tasks if t['processing_stage'] == stage]
            
            cancelled_count = 0
            for task in active_tasks:
                if task.get('can_cancel', False):
                    # Cancel in Celery
                    celery_cancelled = self.orchestrator.cancel_task(task['celery_task_id'])
                    if celery_cancelled:
                        cancelled_count += 1
            
            return {
                "status": "success" if cancelled_count > 0 else "no_action",
                "cancelled_tasks": cancelled_count,
                "message": f"Cancelled {cancelled_count} tasks"
            }
            
        except Exception as e:
            logger.error(f"Error cancelling tasks: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }