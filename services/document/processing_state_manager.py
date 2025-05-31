import os
import json
import pickle
import logging
import time
from typing import Any, Optional, Dict, List
from datetime import datetime
from enum import Enum
from db.task_store.repository import TaskRepository, TaskStatus
from services.document.storage import StorageAdapter

logger = logging.getLogger(__name__)

class ProcessingStage(Enum):
    """Enumeration of all processing stages"""
    UPLOAD = "upload"
    EXTRACTION = "extraction"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    TREE_BUILDING = "tree_building"
    VECTOR_STORAGE = "vector_storage"
    COMPLETED = "completed"

class ProcessingStateManager:
    """
    Enhanced ProcessingStateManager with Celery task integration.
    Manages both file-based stage data AND database-backed task states.
    """
    
    def __init__(self, document_id: str, storage_adapter: StorageAdapter, doc_dir: str, doc_repository=None, case_id: str = None, user_id: str = None):
        """
        Initialize the processing state manager with task repository integration.
        
        Args:
            document_id: Document ID
            storage_adapter: Storage adapter for file operations
            doc_dir: Document storage directory
            doc_repository: Optional document repository for coordination
            case_id: Case ID (required for task management)
            user_id: User ID (required for task management)
        """
        # ... existing initialization code ...
        self.document_id = document_id
        self.storage_adapter = storage_adapter
        self.doc_dir = doc_dir
        self.doc_repository = doc_repository
        self.stages_dir = os.path.join(doc_dir, "stages")
        self.case_id = case_id
        self.user_id = user_id
        
        # Ensure stages directory exists
        self.storage_adapter.create_directory(self.stages_dir)
        
        # Load or initialize processing state (existing logic)
        self.processing_state = self._load_processing_state()
        
        # Initialize task repository for Celery integration
        self.task_repository = TaskRepository()
    
    def register_celery_task(
        self, 
        stage: str, 
        celery_task_id: str, 
        task_name: str,
        worker_hostname: Optional[str] = None,
        worker_pid: Optional[int] = None
    ) -> bool:
        """
        Register a Celery task for a processing stage.
        
        Args:
            stage: Processing stage name
            celery_task_id: Celery task ID
            task_name: Task name/type
            worker_hostname: Optional worker hostname
            worker_pid: Optional worker PID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.case_id or not self.user_id:
                logger.error(f"Cannot register task without case_id and user_id for document {self.document_id}")
                return False
            
            # Register task in database
            task_db_id = self.task_repository.register_task(
                document_id=self.document_id,
                case_id=self.case_id,
                user_id=self.user_id,
                processing_stage=stage,
                celery_task_id=celery_task_id,
                task_name=task_name,
                worker_hostname=worker_hostname,
                worker_pid=worker_pid
            )
            
            logger.info(f"Registered Celery task {celery_task_id} for document {self.document_id}, stage {stage}")
            return task_db_id > 0
            
        except Exception as e:
            logger.error(f"Error registering Celery task for stage {stage}: {str(e)}")
            return False
    
    def update_task_status(
        self,
        stage: str,
        status: TaskStatus,
        progress: Optional[int] = None,
        error_details: Optional[str] = None,
        checkpoint_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update Celery task status for a stage.
        
        Args:
            stage: Processing stage name
            status: Task status
            progress: Progress percentage (0-100)
            error_details: Error details for failed tasks
            checkpoint_data: Checkpoint data for pause/resume
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the task for this stage
            task = self.task_repository.get_task_by_stage(self.document_id, stage)
            if not task:
                logger.info(f"No task found for document {self.document_id}, stage {stage} - this is normal if not using Celery")
                return False
            
            # Update task status in database
            success = self.task_repository.update_task_status(
                celery_task_id=task['celery_task_id'],
                status=status,
                progress=progress,
                error_details=error_details,
                checkpoint_data=checkpoint_data,
                metadata=metadata
            )
            
            if success:
                logger.info(f"Updated task status to {status.value} for document {self.document_id}, stage {stage}")
            
            return success
            
        except Exception as e:
            logger.info(f"Could not update task status for stage {stage} (this is normal if not using Celery): {str(e)}")
            return False
        
    def _load_processing_state(self) -> Dict[str, Any]:
        """Load processing state from storage"""
        state_file = os.path.join(self.stages_dir, "processing_state.json")
        
        if self.storage_adapter.file_exists(state_file):
            try:
                state_data = self.storage_adapter.read_file(state_file)
                if state_data:
                    return json.loads(state_data)
            except Exception as e:
                logger.error(f"Error loading processing state for {self.document_id}: {str(e)}")
        
        # Return default state
        return {
            "current_stage": ProcessingStage.UPLOAD.value,
            "completed_stages": [],
            "stage_data_paths": {},
            "stage_completion_times": {},
            "stage_error_details": {},
            "retry_counts": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_processing_state(self) -> bool:
        """Save current processing state to storage"""
        try:
            self.processing_state["last_updated"] = datetime.now().isoformat()
            state_file = os.path.join(self.stages_dir, "processing_state.json")
            
            state_json = json.dumps(self.processing_state, indent=2)
            success = self.storage_adapter.write_file(state_json, state_file)
            
            if success:
                logger.info(f"Saved processing state for document {self.document_id}")
            return success
        except Exception as e:
            logger.error(f"Error saving processing state for {self.document_id}: {str(e)}")
            return False
    
    def _acquire_state_lock(self, timeout_seconds: int = 30) -> bool:
        """
        Acquire a lock for state modification operations.
        
        Args:
            timeout_seconds: Maximum time to wait for lock acquisition
            
        Returns:
            True if lock acquired, False otherwise
        """
        lock_file = os.path.join(self.stages_dir, "state.lock")
        start_time = time.time()
        
        while (time.time() - start_time) < timeout_seconds:
            try:
                # Check if lock file exists
                if not self.storage_adapter.file_exists(lock_file):
                    # Create lock file
                    lock_data = {
                        "locked_at": datetime.now().isoformat(),
                        "locked_by": f"process_{os.getpid()}",
                        "document_id": self.document_id
                    }
                    
                    success = self.storage_adapter.write_file(
                        json.dumps(lock_data), 
                        lock_file
                    )
                    
                    if success:
                        logger.debug(f"Acquired state lock for document {self.document_id}")
                        return True
                else:
                    # Check if existing lock is stale (older than 5 minutes)
                    try:
                        lock_content = self.storage_adapter.read_file(lock_file)
                        if lock_content:
                            lock_data = json.loads(lock_content)
                            locked_at = datetime.fromisoformat(lock_data["locked_at"])
                            
                            # If lock is older than 5 minutes, consider it stale
                            if (datetime.now() - locked_at).total_seconds() > 300:
                                logger.warning(f"Removing stale lock for document {self.document_id}")
                                self.storage_adapter.delete_file(lock_file)
                                continue  # Try to acquire lock again
                    except Exception as e:
                        logger.warning(f"Error checking lock staleness: {str(e)}")
                        # If we can't read the lock, try to remove it
                        self.storage_adapter.delete_file(lock_file)
                        continue
                
                # Wait a bit before retrying
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error acquiring lock for document {self.document_id}: {str(e)}")
                time.sleep(0.1)
        
        logger.error(f"Could not acquire lock for document {self.document_id} within {timeout_seconds} seconds")
        return False
    
    def _release_state_lock(self):
        """Release the state modification lock"""
        try:
            lock_file = os.path.join(self.stages_dir, "state.lock")
            if self.storage_adapter.file_exists(lock_file):
                self.storage_adapter.delete_file(lock_file)
                logger.debug(f"Released state lock for document {self.document_id}")
        except Exception as e:
            logger.warning(f"Error releasing lock for document {self.document_id}: {str(e)}")
    
    def get_current_stage(self) -> str:
        """Get the current processing stage"""
        return self.processing_state.get("current_stage", ProcessingStage.UPLOAD.value)
    
    def is_stage_complete(self, stage: str) -> bool:
        """Check if a stage is completed"""
        return stage in self.processing_state.get("completed_stages", [])
    
    def get_completed_stages(self) -> List[str]:
        """Get list of completed stages"""
        return self.processing_state.get("completed_stages", [])
    
    def get_stage_file_path(self, stage: str, filename: str) -> str:
        """Get the file path for stage data"""
        return os.path.join(self.stages_dir, f"{stage}_{filename}")
    
    def save_stage_data(self, stage: str, data: Any, filename: str = "data") -> bool:
        """
        Save data for a specific stage.
        
        Args:
            stage: Processing stage name
            data: Data to save
            filename: Base filename (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine file extension and serialization method based on data type
            if isinstance(data, str):
                file_path = self.get_stage_file_path(stage, f"{filename}.txt")
                content = data
            elif self._is_json_serializable(data):
                file_path = self.get_stage_file_path(stage, f"{filename}.json")
                content = json.dumps(data, indent=2, default=self._json_serializer)
            else:
                # Use pickle for complex objects (embeddings, DataFrames, etc.)
                file_path = self.get_stage_file_path(stage, f"{filename}.pkl")
                content = pickle.dumps(data)
            
            # Save the data
            success = self.storage_adapter.write_file(content, file_path)
            
            if success:
                # Update stage data paths without locking (this is just metadata)
                if "stage_data_paths" not in self.processing_state:
                    self.processing_state["stage_data_paths"] = {}
                if stage not in self.processing_state["stage_data_paths"]:
                    self.processing_state["stage_data_paths"][stage] = {}
                
                self.processing_state["stage_data_paths"][stage][filename] = file_path
                self._save_processing_state()  # Save state updates
                
                logger.info(f"Saved {stage} data ({filename}) for document {self.document_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error saving {stage} data for {self.document_id}: {str(e)}")
            return False
    
    def _is_json_serializable(self, obj: Any) -> bool:
        """
        Check if an object is JSON serializable.
        
        Args:
            obj: Object to check
            
        Returns:
            True if JSON serializable, False otherwise
        """
        try:
            json.dumps(obj, default=self._json_serializer)
            return True
        except (TypeError, ValueError):
            return False
    
    def _json_serializer(self, obj: Any) -> Any:
        """
        Custom JSON serializer for handling special data types.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation
        """
        # Handle pandas DataFrames
        if hasattr(obj, 'to_dict'):
            return obj.to_dict('records')
        
        # Handle numpy arrays
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        
        # Handle datetime objects
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        
        # For other objects, convert to string representation
        return str(obj)
    
    def load_stage_data(self, stage: str, filename: str = "data") -> Optional[Any]:
        """
        Load data for a specific stage.
        
        Args:
            stage: Processing stage name
            filename: Base filename (without extension)
            
        Returns:
            Loaded data or None if not found
        """
        try:
            # Get file path from processing state
            stage_paths = self.processing_state.get("stage_data_paths", {}).get(stage, {})
            
            if filename not in stage_paths:
                logger.warning(f"No data file '{filename}' found for stage '{stage}' in document {self.document_id}")
                return None
            
            file_path = stage_paths[filename]
            
            if not self.storage_adapter.file_exists(file_path):
                logger.warning(f"Stage data file does not exist: {file_path}")
                return None
            
            # Load the data
            content = self.storage_adapter.read_file(file_path)
            
            if content is None:
                return None
            
            # Determine deserialization method based on file extension
            if file_path.endswith('.json'):
                return json.loads(content)
            elif file_path.endswith('.txt'):
                return content
            elif file_path.endswith('.pkl'):
                return pickle.loads(content)
            else:
                # Default to returning raw content
                return content
                
        except Exception as e:
            logger.error(f"Error loading {stage} data ({filename}) for {self.document_id}: {str(e)}")
            return None
    
    def mark_stage_complete(self, stage: str) -> bool:
        """
        Mark a stage as completed and advance to next stage.
        Enhanced to work with task system.
        """
        # ... existing logic ...
        
        # Try to acquire lock with longer timeout for critical operations
        if not self._acquire_state_lock(timeout_seconds=60):
            logger.error(f"Could not acquire lock for document {self.document_id}")
            logger.warning(f"Proceeding without lock for stage completion of {stage} in document {self.document_id}")
        
        try:
            # Add to completed stages if not already there
            completed_stages = self.processing_state.get("completed_stages", [])
            if stage not in completed_stages:
                completed_stages.append(stage)
                self.processing_state["completed_stages"] = completed_stages
            
            # Update completion time
            completion_times = self.processing_state.get("stage_completion_times", {})
            completion_times[stage] = datetime.now().isoformat()
            self.processing_state["stage_completion_times"] = completion_times
            
            # Advance to next stage
            next_stage = self._get_next_stage(stage)
            if next_stage:
                self.processing_state["current_stage"] = next_stage
            
            # Clear any error details for this stage
            if "stage_error_details" in self.processing_state and stage in self.processing_state["stage_error_details"]:
                del self.processing_state["stage_error_details"][stage]
            
            success = self._save_processing_state()
            
            # Sync with document repository if available
            if success and self.doc_repository:
                try:
                    self.doc_repository.update_processing_state(
                        self.document_id, 
                        self.processing_state
                    )
                except Exception as e:
                    logger.warning(f"Failed to sync with document repository: {str(e)}")
            
            if success:
                logger.info(f"Marked stage '{stage}' as complete for document {self.document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error marking stage complete for {self.document_id}: {str(e)}")
            return False
        finally:
            self._release_state_lock()
    
    def mark_stage_failed(self, stage: str, error_message: str) -> bool:
        """
        Mark a stage as failed with error details.
        
        Args:
            stage: Stage that failed
            error_message: Error message
            
        Returns:
            True if successful, False otherwise
        """
        # Try to acquire lock
        if not self._acquire_state_lock(timeout_seconds=30):
            logger.warning(f"Could not acquire lock for marking stage failed, proceeding anyway for document {self.document_id}")
        
        try:
            # Update error details
            error_details = self.processing_state.get("stage_error_details", {})
            error_details[stage] = {
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
            self.processing_state["stage_error_details"] = error_details
            
            # Increment retry count
            retry_counts = self.processing_state.get("retry_counts", {})
            retry_counts[stage] = retry_counts.get(stage, 0) + 1
            self.processing_state["retry_counts"] = retry_counts
            
            success = self._save_processing_state()
            
            # Sync with document repository if available
            if success and self.doc_repository:
                try:
                    self.doc_repository.update_processing_state(
                        self.document_id, 
                        self.processing_state
                    )
                except Exception as e:
                    logger.warning(f"Failed to sync failed stage with document repository: {str(e)}")
            
            if success:
                logger.error(f"Marked stage '{stage}' as failed for document {self.document_id}: {error_message}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error marking stage failed for {self.document_id}: {str(e)}")
            return False
        finally:
            self._release_state_lock()
    
    def reset_to_stage(self, target_stage: str) -> bool:
        """
        Reset processing to a specific stage, clearing subsequent stages.
        
        Args:
            target_stage: Stage to reset to
            
        Returns:
            True if successful, False otherwise
        """
        # Try to acquire lock
        if not self._acquire_state_lock(timeout_seconds=30):
            logger.warning(f"Could not acquire lock for stage reset, proceeding anyway for document {self.document_id}")
        
        try:
            stage_order = [stage.value for stage in ProcessingStage]
            target_index = stage_order.index(target_stage)
            
            # Remove stages after target from completed list
            completed_stages = [
                stage for stage in self.processing_state.get("completed_stages", [])
                if stage_order.index(stage) <= target_index
            ]
            
            self.processing_state["completed_stages"] = completed_stages
            self.processing_state["current_stage"] = target_stage
            
            # Clear completion times for reset stages
            completion_times = self.processing_state.get("stage_completion_times", {})
            for stage in stage_order[target_index + 1:]:
                if stage in completion_times:
                    del completion_times[stage]
            
            # Clear error details for reset stages
            error_details = self.processing_state.get("stage_error_details", {})
            for stage in stage_order[target_index:]:
                if stage in error_details:
                    del error_details[stage]
            
            success = self._save_processing_state()
            
            # Sync with document repository if available
            if success and self.doc_repository:
                try:
                    self.doc_repository.update_processing_state(
                        self.document_id, 
                        self.processing_state
                    )
                except Exception as e:
                    logger.warning(f"Failed to sync stage reset with document repository: {str(e)}")
            
            if success:
                logger.info(f"Reset document {self.document_id} to stage '{target_stage}'")
            
            return success
            
        except Exception as e:
            logger.error(f"Error resetting to stage for {self.document_id}: {str(e)}")
            return False
        finally:
            self._release_state_lock()
    
    def get_stage_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all stages"""
        return {
            "document_id": self.document_id,
            "current_stage": self.get_current_stage(),
            "completed_stages": self.get_completed_stages(),
            "stage_completion_times": self.processing_state.get("stage_completion_times", {}),
            "stage_error_details": self.processing_state.get("stage_error_details", {}),
            "retry_counts": self.processing_state.get("retry_counts", {}),
            "last_updated": self.processing_state.get("last_updated")
        }
    
    def _get_next_stage(self, current_stage: str) -> Optional[str]:
        """Get the next stage in the processing pipeline"""
        stage_order = [stage.value for stage in ProcessingStage]
        
        try:
            current_index = stage_order.index(current_stage)
            if current_index < len(stage_order) - 1:
                return stage_order[current_index + 1]
            return None
        except ValueError:
            logger.error(f"Unknown stage: {current_stage}")
            return None
    
    def cleanup_stage_data(self, stage: str) -> bool:
        """
        Clean up intermediate data for a specific stage.
        
        Args:
            stage: Stage to clean up
            
        Returns:
            True if successful, False otherwise
        """
        try:
            stage_paths = self.processing_state.get("stage_data_paths", {}).get(stage, {})
            
            for filename, file_path in stage_paths.items():
                if self.storage_adapter.file_exists(file_path):
                    self.storage_adapter.delete_file(file_path)
                    logger.info(f"Cleaned up stage data: {file_path}")
            
            # Remove from processing state
            if "stage_data_paths" in self.processing_state and stage in self.processing_state["stage_data_paths"]:
                del self.processing_state["stage_data_paths"][stage]
            
            return self._save_processing_state()
        except Exception as e:
            logger.error(f"Error cleaning up stage data for {self.document_id}: {str(e)}")
            return False
        
    def mark_task_started(self, stage: str, progress: int = 0) -> bool:
        """Mark a task as started"""
        try:
            return self.update_task_status(stage, TaskStatus.RUNNING, progress=progress)
        except Exception as e:
            logger.info(f"Could not mark task started for stage {stage} (normal if not using Celery): {str(e)}")
            return False
    
    def mark_task_completed(self, stage: str) -> bool:
        """Mark a task as successfully completed"""
        # Try to update task status if task exists
        task_updated = False
        try:
            task_updated = self.update_task_status(stage, TaskStatus.SUCCESS, progress=100)
        except Exception as e:
            logger.info(f"No task to update for stage {stage}: {str(e)}")
        
        # Always mark stage as complete
        stage_completed = self.mark_stage_complete(stage)
        
        return task_updated or stage_completed
    
    def mark_task_failed(self, stage: str, error_message: str) -> bool:
        """Mark a task as failed"""
        # Try to update task status if task exists
        task_updated = False
        try:
            task_updated = self.update_task_status(stage, TaskStatus.FAILURE, error_details=error_message)
        except Exception as e:
            logger.info(f"No task to update for stage {stage}: {str(e)}")
        
        # Always mark stage as failed
        stage_failed = self.mark_stage_failed(stage, error_message)
        
        return task_updated or stage_failed
    
    def mark_task_paused(self, stage: str, checkpoint_data: Optional[Dict[str, Any]] = None) -> bool:
        """Mark a task as paused with checkpoint data"""
        return self.update_task_status(stage, TaskStatus.PAUSED, checkpoint_data=checkpoint_data)
    
    def resume_task(self, stage: str) -> bool:
        """Resume a paused task"""
        return self.update_task_status(stage, TaskStatus.RUNNING)
    
    def cancel_task(self, stage: str) -> bool:
        """Cancel a running task"""
        return self.update_task_status(stage, TaskStatus.CANCELLED)
    
    def get_task_info(self, stage: str) -> Optional[Dict[str, Any]]:
        """Get task information for a stage"""
        return self.task_repository.get_task_by_stage(self.document_id, stage)
    
    def get_celery_task_id(self, stage: str) -> Optional[str]:
        """Get Celery task ID for a stage"""
        task = self.get_task_info(stage)
        return task['celery_task_id'] if task else None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks for this document"""
        return self.task_repository.get_tasks_by_document(self.document_id)
    
    def get_task_controllability(self) -> Dict[str, List[str]]:
        """Get which tasks can be paused/resumed/cancelled"""
        return self.task_repository.get_controllable_tasks(self.document_id)
    
    def get_task_progress(self, stage: str) -> int:
        """Get task progress for a stage (0-100)"""
        task = self.get_task_info(stage)
        return task['progress'] if task else 0
    
    def update_task_progress(self, stage: str, progress: int, message: Optional[str] = None) -> bool:
        """Update task progress"""
        try:
            metadata = {"progress_message": message} if message else None
            return self.update_task_status(stage, TaskStatus.RUNNING, progress=progress, metadata=metadata)
        except Exception as e:
            logger.debug(f"Could not update task progress for stage {stage} (normal if not using Celery): {str(e)}")
            return False
    
    # === ENHANCED STATUS METHODS ===
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status including both file-based stages and Celery tasks.
        
        Returns:
            Complete status information
        """
        # Get existing file-based status
        file_status = self.get_stage_status()
        
        # Get task information
        tasks = self.get_all_tasks()
        task_summary = {}
        
        for task in tasks:
            stage = task['processing_stage']
            task_summary[stage] = {
                "celery_task_id": task['celery_task_id'],
                "task_status": task['task_status'],
                "progress": task['progress'],
                "can_pause": task['can_pause'],
                "can_resume": task['can_resume'],
                "can_cancel": task['can_cancel'],
                "started_at": task['started_at'].isoformat() if task['started_at'] else None,
                "updated_at": task['updated_at'].isoformat() if task['updated_at'] else None,
                "completed_at": task['completed_at'].isoformat() if task['completed_at'] else None,
                "worker_hostname": task['worker_hostname'],
                "error_details": task['error_details']
            }
        
        return {
            **file_status,
            "tasks": task_summary,
            "task_controllability": self.get_task_controllability()
        }
    # === EXISTING METHODS (keep as-is, but can be enhanced) ===
    
    def mark_stage_complete(self, stage: str) -> bool:
        """
        Mark a stage as completed and advance to next stage.
        Enhanced to work with task system.
        """
        
        # Try to acquire lock with longer timeout for critical operations
        if not self._acquire_state_lock(timeout_seconds=60):
            logger.error(f"Could not acquire lock for document {self.document_id}")
            logger.warning(f"Proceeding without lock for stage completion of {stage} in document {self.document_id}")
        
        try:
            # Add to completed stages if not already there
            completed_stages = self.processing_state.get("completed_stages", [])
            if stage not in completed_stages:
                completed_stages.append(stage)
                self.processing_state["completed_stages"] = completed_stages
            
            # Update completion time
            completion_times = self.processing_state.get("stage_completion_times", {})
            completion_times[stage] = datetime.now().isoformat()
            self.processing_state["stage_completion_times"] = completion_times
            
            # Advance to next stage
            next_stage = self._get_next_stage(stage)
            if next_stage:
                self.processing_state["current_stage"] = next_stage
            
            # Clear any error details for this stage
            if "stage_error_details" in self.processing_state and stage in self.processing_state["stage_error_details"]:
                del self.processing_state["stage_error_details"][stage]
            
            success = self._save_processing_state()
            
            # Sync with document repository if available
            if success and self.doc_repository:
                try:
                    self.doc_repository.update_processing_state(
                        self.document_id, 
                        self.processing_state
                    )
                except Exception as e:
                    logger.warning(f"Failed to sync with document repository: {str(e)}")
            
            if success:
                logger.info(f"Marked stage '{stage}' as complete for document {self.document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error marking stage complete for {self.document_id}: {str(e)}")
            return False
        finally:
            self._release_state_lock()