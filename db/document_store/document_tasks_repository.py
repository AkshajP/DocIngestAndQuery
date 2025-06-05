import psycopg2
import psycopg2.extras
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "PENDING"
    STARTED = "STARTED" 
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PAUSED = "PAUSED"
    RESUMED = "RESUMED"
    CANCELLED = "CANCELLED"
    RETRY = "RETRY"

class ProcessingStage(Enum):
    """Processing stage enumeration"""
    UPLOAD = "upload"
    EXTRACTION = "extraction"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    TREE_BUILDING = "tree_building"
    VECTOR_STORAGE = "vector_storage"
    COMPLETED = "completed"

@dataclass
class DocumentTask:
    """Document task data structure"""
    id: Optional[int] = None
    document_id: str = ""
    current_stage: str = ProcessingStage.UPLOAD.value
    celery_task_id: Optional[str] = None
    task_status: str = TaskStatus.PENDING.value
    can_pause: bool = True
    can_resume: bool = False
    can_cancel: bool = True
    pause_requested: bool = False
    cancel_requested: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    worker_hostname: Optional[str] = None
    percent_complete: int = 0
    checkpoint_data: Dict[str, Any] = None
    stage_metadata: Dict[str, Any] = None
    error_details: Dict[str, Any] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.checkpoint_data is None:
            self.checkpoint_data = {}
        if self.stage_metadata is None:
            self.stage_metadata = {}
        if self.error_details is None:
            self.error_details = {}

class DocumentTasksRepository:
    """Repository for managing document tasks in PostgreSQL"""
    
    def __init__(self, connection_string: str = None):
        """
        Initialize the repository with database connection
        
        Args:
            connection_string: PostgreSQL connection string (if None, will use environment)
        """
        self.connection_string = connection_string or self._get_connection_string()
    
    def _get_connection_string(self) -> str:
        """Get database connection string from environment"""
        import os
        return os.getenv(
            'DATABASE_URL', 
            'postgresql://youruser:yourpassword@localhost:5433/yourdb'
        )
    
    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            self.connection_string,
            cursor_factory=psycopg2.extras.RealDictCursor
        )
    
    def create_task(self, document_task: DocumentTask) -> int:
        """
        Create a new document task
        
        Args:
            document_task: DocumentTask instance
            
        Returns:
            Task ID
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO document_tasks (
                        document_id, current_stage, celery_task_id, task_status,
                        can_pause, can_resume, can_cancel, pause_requested, cancel_requested,
                        worker_id, worker_hostname, percent_complete,
                        checkpoint_data, stage_metadata, error_details,
                        retry_count, max_retries
                    ) VALUES (
                        %(document_id)s, %(current_stage)s, %(celery_task_id)s, %(task_status)s,
                        %(can_pause)s, %(can_resume)s, %(can_cancel)s, %(pause_requested)s, %(cancel_requested)s,
                        %(worker_id)s, %(worker_hostname)s, %(percent_complete)s,
                        %(checkpoint_data)s, %(stage_metadata)s, %(error_details)s,
                        %(retry_count)s, %(max_retries)s
                    ) RETURNING id
                """, {
                    'document_id': document_task.document_id,
                    'current_stage': document_task.current_stage,
                    'celery_task_id': document_task.celery_task_id,
                    'task_status': document_task.task_status,
                    'can_pause': document_task.can_pause,
                    'can_resume': document_task.can_resume,
                    'can_cancel': document_task.can_cancel,
                    'pause_requested': document_task.pause_requested,
                    'cancel_requested': document_task.cancel_requested,
                    'worker_id': document_task.worker_id,
                    'worker_hostname': document_task.worker_hostname,
                    'percent_complete': document_task.percent_complete,
                    'checkpoint_data': json.dumps(document_task.checkpoint_data),
                    'stage_metadata': json.dumps(document_task.stage_metadata),
                    'error_details': json.dumps(document_task.error_details),
                    'retry_count': document_task.retry_count,
                    'max_retries': document_task.max_retries,
                })
                
                task_id = cursor.fetchone()['id']
                conn.commit()
                logger.info(f"Created document task {task_id} for document {document_task.document_id}")
                return task_id
    
    def get_task_by_document_id(self, document_id: str) -> Optional[DocumentTask]:
        """
        Get task by document ID
        
        Args:
            document_id: Document ID
            
        Returns:
            DocumentTask instance or None
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM document_tasks WHERE document_id = %s
                """, (document_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_document_task(row)
                return None
    
    def get_task_by_celery_id(self, celery_task_id: str) -> Optional[DocumentTask]:
        """
        Get task by Celery task ID
        
        Args:
            celery_task_id: Celery task ID
            
        Returns:
            DocumentTask instance or None
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM document_tasks WHERE celery_task_id = %s
                """, (celery_task_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_document_task(row)
                return None
    
    def update_task_status(
        self, 
        document_id: str, 
        task_status: str,
        celery_task_id: Optional[str] = None,
        worker_info: Optional[Dict[str, str]] = None,
        error_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update task status
        
        Args:
            document_id: Document ID
            task_status: New task status
            celery_task_id: Optional Celery task ID
            worker_info: Optional worker information dict
            error_details: Optional error details
            
        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                update_fields = ['task_status = %s']
                update_values = [task_status]
                
                # Set timestamps based on status
                if task_status == TaskStatus.STARTED.value:
                    update_fields.append('started_at = CURRENT_TIMESTAMP')
                elif task_status in [TaskStatus.SUCCESS.value, TaskStatus.FAILURE.value, TaskStatus.CANCELLED.value]:
                    update_fields.append('completed_at = CURRENT_TIMESTAMP')
                
                # Add optional fields
                if celery_task_id:
                    update_fields.append('celery_task_id = %s')
                    update_values.append(celery_task_id)
                
                if worker_info:
                    if 'worker_id' in worker_info:
                        update_fields.append('worker_id = %s')
                        update_values.append(worker_info['worker_id'])
                    if 'worker_hostname' in worker_info:
                        update_fields.append('worker_hostname = %s')
                        update_values.append(worker_info['worker_hostname'])
                
                if error_details:
                    update_fields.append('error_details = %s')
                    update_values.append(json.dumps(error_details))
                
                # Update control flags based on status
                if task_status == TaskStatus.PAUSED.value:
                    update_fields.extend(['can_pause = false', 'can_resume = true', 'pause_requested = false'])
                elif task_status == TaskStatus.STARTED.value:
                    update_fields.extend(['can_pause = true', 'can_resume = false'])
                elif task_status in [TaskStatus.SUCCESS.value, TaskStatus.FAILURE.value, TaskStatus.CANCELLED.value]:
                    update_fields.extend(['can_pause = false', 'can_resume = false', 'can_cancel = false'])
                
                update_values.append(document_id)
                
                cursor.execute(f"""
                    UPDATE document_tasks 
                    SET {', '.join(update_fields)}
                    WHERE document_id = %s
                """, update_values)
                
                rows_affected = cursor.rowcount
                conn.commit()
                
                if rows_affected > 0:
                    logger.info(f"Updated task status to {task_status} for document {document_id}")
                    return True
                else:
                    logger.warning(f"No task found for document {document_id}")
                    return False
    
    def update_progress(self, document_id: str, percent_complete: int, checkpoint_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update task progress
        
        Args:
            document_id: Document ID
            percent_complete: Progress percentage (0-100)
            checkpoint_data: Optional checkpoint data for resume
            
        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                update_fields = ['percent_complete = %s']
                update_values = [max(0, min(100, percent_complete))]  # Clamp to 0-100
                
                if checkpoint_data:
                    update_fields.append('checkpoint_data = %s')
                    update_values.append(json.dumps(checkpoint_data))
                
                update_values.append(document_id)
                
                cursor.execute(f"""
                    UPDATE document_tasks 
                    SET {', '.join(update_fields)}
                    WHERE document_id = %s
                """, update_values)
                
                rows_affected = cursor.rowcount
                conn.commit()
                
                return rows_affected > 0
    
    def set_pause_request(self, document_id: str, pause: bool = True) -> bool:
        """
        Set pause request flag
        
        Args:
            document_id: Document ID
            pause: True to request pause, False to clear
            
        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE document_tasks 
                    SET pause_requested = %s
                    WHERE document_id = %s AND task_status = %s
                """, (pause, document_id, TaskStatus.STARTED.value))
                
                rows_affected = cursor.rowcount
                conn.commit()
                
                return rows_affected > 0
    
    def set_cancel_request(self, document_id: str, cancel: bool = True) -> bool:
        """
        Set cancel request flag
        
        Args:
            document_id: Document ID
            cancel: True to request cancel, False to clear
            
        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE document_tasks 
                    SET cancel_requested = %s
                    WHERE document_id = %s AND can_cancel = true
                """, (cancel, document_id))
                
                rows_affected = cursor.rowcount
                conn.commit()
                
                return rows_affected > 0
    
    def get_tasks_by_status(self, task_status: str, limit: int = 100) -> List[DocumentTask]:
        """
        Get tasks by status
        
        Args:
            task_status: Task status to filter by
            limit: Maximum number of tasks to return
            
        Returns:
            List of DocumentTask instances
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM document_tasks 
                    WHERE task_status = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (task_status, limit))
                
                return [self._row_to_document_task(row) for row in cursor.fetchall()]
    
    def delete_task(self, document_id: str) -> bool:
        """
        Delete a document task
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM document_tasks WHERE document_id = %s
                """, (document_id,))
                
                rows_affected = cursor.rowcount
                conn.commit()
                
                if rows_affected > 0:
                    logger.info(f"Deleted task for document {document_id}")
                    return True
                return False
    
    def _row_to_document_task(self, row: Dict[str, Any]) -> DocumentTask:
        """Convert database row to DocumentTask instance"""
        return DocumentTask(
            id=row['id'],
            document_id=row['document_id'],
            current_stage=row['current_stage'],
            celery_task_id=row['celery_task_id'],
            task_status=row['task_status'],
            can_pause=row['can_pause'],
            can_resume=row['can_resume'],
            can_cancel=row['can_cancel'],
            pause_requested=row['pause_requested'],
            cancel_requested=row['cancel_requested'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            worker_id=row['worker_id'],
            worker_hostname=row['worker_hostname'],
            percent_complete=row['percent_complete'],
            checkpoint_data=json.loads(row['checkpoint_data']) if row['checkpoint_data'] else {},
            stage_metadata=json.loads(row['stage_metadata']) if row['stage_metadata'] else {},
            error_details=json.loads(row['error_details']) if row['error_details'] else {},
            retry_count=row['retry_count'],
            max_retries=row['max_retries'],
        )