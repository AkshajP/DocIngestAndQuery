import psycopg2
import psycopg2.extras
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from core.config import get_config

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running" 
    PAUSED = "paused"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    RETRY = "retry"

class TaskRepository:
    """Repository for managing Celery task states in PostgreSQL"""
    
    def __init__(self, db_config=None):
        """Initialize task repository with database configuration"""
        self.config = db_config or get_config().database
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Connect to the database"""
        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                dbname=self.config.dbname,
                user=self.config.user,
                password=self.config.password,
                connect_timeout=self.config.connection_timeout
            )
            logger.info("Connected to task database")
        except Exception as e:
            logger.error(f"Error connecting to task database: {str(e)}")
            raise
    
    def _ensure_connection(self):
        """Ensure database connection is active"""
        try:
            if not self.conn or self.conn.closed:
                self._connect()
        except Exception as e:
            logger.error(f"Failed to reconnect to database: {str(e)}")
            raise
    
    def register_task(
        self,
        document_id: str,
        case_id: str,
        user_id: str,
        processing_stage: str,
        celery_task_id: str,
        task_name: str,
        worker_hostname: Optional[str] = None,
        worker_pid: Optional[int] = None
    ) -> int:
        """
        Register a new Celery task.
        
        Returns:
            Task database ID (0 if already exists)
        """
        self._ensure_connection()
        
        try:
            with self.conn.cursor() as cursor:
                # First check if task already exists
                cursor.execute(
                    "SELECT id FROM document_tasks WHERE celery_task_id = %s",
                    (celery_task_id,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    logger.info(f"Task {celery_task_id} already registered with id {existing[0]}")
                    self.conn.commit()  # Important: commit to clear transaction
                    return existing[0]
                
                # If not exists, insert new task
                cursor.execute("""
                    INSERT INTO document_tasks (
                        document_id, case_id, user_id, processing_stage, 
                        celery_task_id, task_name, worker_hostname, worker_pid,
                        started_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING id
                """, (
                    document_id, case_id, user_id, processing_stage,
                    celery_task_id, task_name, worker_hostname, worker_pid
                ))
                
                task_id = cursor.fetchone()[0]
                self.conn.commit()
                
                logger.info(f"Registered task {celery_task_id} for document {document_id}, stage {processing_stage}")
                return task_id
                
        except psycopg2.errors.UniqueViolation:
            # Handle race condition where another worker registered the task
            logger.info(f"Task {celery_task_id} was registered by another worker")
            self.conn.rollback()  # Clear the aborted transaction
            
            # Get the existing task ID
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM document_tasks WHERE celery_task_id = %s",
                    (celery_task_id,)
                )
                existing = cursor.fetchone()
                return existing[0] if existing else 0
                
        except Exception as e:
            logger.error(f"Error registering task: {str(e)}")
            self.conn.rollback()  # Always rollback on error
            raise
    
    def update_task_status(
        self,
        celery_task_id: str,
        status: TaskStatus,
        progress: Optional[int] = None,
        error_details: Optional[str] = None,
        checkpoint_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update task status and related fields"""
        self._recover_connection()
        
        try:
            with self.conn.cursor() as cursor:
                # Check if we're in a failed transaction state
                if self.conn.status != psycopg2.extensions.STATUS_READY:
                    logger.warning("Database connection not ready, rolling back and reconnecting")
                    self.conn.rollback()
                    self._ensure_connection()
                
                # Build dynamic update query
                updates = ["task_status = %s"]
                params = [status.value]
                
                if progress is not None:
                    updates.append("progress = %s")
                    params.append(max(0, min(100, progress)))
                
                if error_details is not None:
                    updates.append("error_details = %s")
                    params.append(error_details)
                
                if checkpoint_data is not None:
                    updates.append("checkpoint_data = %s")
                    params.append(json.dumps(checkpoint_data))
                
                if metadata is not None:
                    # Merge with existing metadata
                    cursor.execute(
                        "SELECT task_metadata FROM document_tasks WHERE celery_task_id = %s",
                        (celery_task_id,)
                    )
                    result = cursor.fetchone()
                    existing_metadata = result[0] if result and result[0] else {}
                    existing_metadata.update(metadata)
                    
                    updates.append("task_metadata = %s")
                    params.append(json.dumps(existing_metadata))
                
                # Update control flags based on status
                if status == TaskStatus.RUNNING:
                    updates.extend(["can_pause = true", "can_resume = false", "can_cancel = true"])
                elif status == TaskStatus.PAUSED:
                    updates.extend(["can_pause = false", "can_resume = true", "can_cancel = true"])
                elif status in [TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.CANCELLED]:
                    updates.extend(["can_pause = false", "can_resume = false", "can_cancel = false", "completed_at = NOW()"])
                
                # Add celery_task_id to params for WHERE clause
                params.append(celery_task_id)
                
                cursor.execute(f"""
                    UPDATE document_tasks 
                    SET {', '.join(updates)}
                    WHERE celery_task_id = %s
                """, params)
                
                rows_updated = cursor.rowcount
                self.conn.commit()
                
                if rows_updated > 0:
                    logger.info(f"Updated task {celery_task_id} status to {status.value}")
                return rows_updated > 0
                
        except psycopg2.Error as e:
            logger.error(f"Database error updating task status: {str(e)}")
            try:
                self.conn.rollback()
            except:
                pass
            # Try to reconnect for next operation
            try:
                self._connect()
            except:
                pass
            return False
        except Exception as e:
            logger.error(f"Error updating task status: {str(e)}")
            try:
                self.conn.rollback()
            except:
                pass
            return False
    
    def get_task_by_celery_id(self, celery_task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by Celery task ID"""
        self._ensure_connection()
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM document_tasks WHERE celery_task_id = %s",
                    (celery_task_id,)
                )
                
                result = cursor.fetchone()
                self.conn.commit()  # Ensure clean transaction state
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error getting task by celery_id: {str(e)}")
            self.conn.rollback()
            return None
    
    def get_tasks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all tasks for a document"""
        self._ensure_connection()
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM document_tasks WHERE document_id = %s ORDER BY created_at",
                (document_id,)
            )
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_task_by_stage(self, document_id: str, processing_stage: str) -> Optional[Dict[str, Any]]:
        """Get task for a specific document and stage"""
        self._ensure_connection()
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM document_tasks WHERE document_id = %s AND processing_stage = %s",
                (document_id, processing_stage)
            )
            
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks for this document"""
        try:
            return self.task_repository.get_tasks_by_document(self.document_id)
        except Exception as e:
            logger.error(f"Error getting tasks for document {self.document_id}: {str(e)}")
            return []
    
    def get_active_tasks(self, case_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all active (running/paused/pending) tasks"""
        self._ensure_connection()
        
        active_statuses = [TaskStatus.PENDING.value, TaskStatus.RUNNING.value, TaskStatus.PAUSED.value]
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            if case_id:
                cursor.execute(
                    "SELECT * FROM document_tasks WHERE task_status = ANY(%s) AND case_id = %s ORDER BY created_at",
                    (active_statuses, case_id)
                )
            else:
                cursor.execute(
                    "SELECT * FROM document_tasks WHERE task_status = ANY(%s) ORDER BY created_at",
                    (active_statuses,)
                )
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_controllable_tasks(self, document_id: str) -> Dict[str, List[str]]:
        """
        Get tasks that can be paused/resumed/cancelled for a document.
        
        Returns:
            Dict with keys: 'pausable', 'resumable', 'cancellable'
        """
        self._ensure_connection()
        
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT processing_stage, can_pause, can_resume, can_cancel 
                FROM document_tasks 
                WHERE document_id = %s
            """, (document_id,))
            
            results = cursor.fetchall()
            
            controllable = {
                'pausable': [row[0] for row in results if row[1]],  # can_pause
                'resumable': [row[0] for row in results if row[2]],  # can_resume  
                'cancellable': [row[0] for row in results if row[3]]  # can_cancel
            }
            
            return controllable
    
    def delete_task(self, celery_task_id: str) -> bool:
        """Delete a task record"""
        self._ensure_connection()
        
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM document_tasks WHERE celery_task_id = %s",
                    (celery_task_id,)
                )
                
                rows_deleted = cursor.rowcount
                self.conn.commit()
                
                return rows_deleted > 0
        except Exception as e:
            logger.error(f"Error deleting task: {str(e)}")
            self.conn.rollback()
            return False
    
    def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clean up completed tasks older than specified hours"""
        self._ensure_connection()
        
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM document_tasks 
                    WHERE task_status IN ('success', 'failure', 'cancelled')
                    AND completed_at < NOW() - INTERVAL '%s hours'
                """, (older_than_hours,))
                
                rows_deleted = cursor.rowcount
                self.conn.commit()
                
                logger.info(f"Cleaned up {rows_deleted} old completed tasks")
                return rows_deleted
        except Exception as e:
            logger.error(f"Error cleaning up tasks: {str(e)}")
            self.conn.rollback()
            return 0
    
    def get_task_stats(self, case_id: Optional[str] = None) -> Dict[str, Any]:
        """Get task statistics"""
        self._ensure_connection()
        
        with self.conn.cursor() as cursor:
            base_query = "SELECT task_status, COUNT(*) FROM document_tasks"
            params = []
            
            if case_id:
                base_query += " WHERE case_id = %s"
                params.append(case_id)
            
            base_query += " GROUP BY task_status"
            
            cursor.execute(base_query, params)
            status_counts = dict(cursor.fetchall())
            
            # Get total count
            count_query = "SELECT COUNT(*) FROM document_tasks"
            if case_id:
                count_query += " WHERE case_id = %s"
                cursor.execute(count_query, (case_id,))
            else:
                cursor.execute(count_query)
            
            total_tasks = cursor.fetchone()[0]
            
            return {
                "total_tasks": total_tasks,
                "status_counts": status_counts,
                "active_tasks": status_counts.get('running', 0) + status_counts.get('pending', 0) + status_counts.get('paused', 0)
            }
            
    def _recover_connection(self):
        """Recover from aborted transaction state"""
        try:
            if self.conn and self.conn.status == psycopg2.extensions.STATUS_IN_TRANSACTION:
                self.conn.rollback()
                logger.info("Rolled back aborted transaction")
        except:
            pass
        
        # Ensure connection is valid
        self._ensure_connection()