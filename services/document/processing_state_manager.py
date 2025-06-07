import os
import json
import pickle
import logging
import time
from typing import Any, Optional, Dict, List
from datetime import datetime
from enum import Enum

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
    Manages the processing state of documents through different stages.
    Handles persistence of intermediate results and stage tracking.
    Now integrates with Celery task management for pause/resume/cancel support.
    """
    
    def __init__(self, document_id: str, storage_adapter: StorageAdapter, doc_dir: str, doc_repository=None, task_state_manager=None):
        """
        Initialize the processing state manager.
        
        Args:
            document_id: Document ID
            storage_adapter: Storage adapter for file operations
            doc_dir: Document storage directory
            doc_repository: Optional document repository for coordination
            task_state_manager: Optional Celery task state manager
        """
        self.document_id = document_id
        self.storage_adapter = storage_adapter
        self.doc_dir = doc_dir
        self.doc_repository = doc_repository
        self.task_state_manager = task_state_manager
        self.stages_dir = os.path.join(doc_dir, "stages")
        
        # Ensure stages directory exists
        self.storage_adapter.create_directory(self.stages_dir)
        
        # Load or initialize processing state
        self.processing_state = self._load_processing_state()
    
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
            "last_updated": datetime.now().isoformat(),
            "celery_integration": {
                "task_ids": {},  # Map stage to celery task IDs
                "checkpoints": {},  # Stage checkpoint data
                "pause_points": {}  # Valid pause points per stage
            }
        }
    
    def _save_processing_state(self) -> bool:
        """Save current processing state to storage"""
        try:
            self.processing_state["last_updated"] = datetime.now().isoformat()
            state_file = os.path.join(self.stages_dir, "processing_state.json")
            
            state_json = json.dumps(self.processing_state, indent=2)
            success = self.storage_adapter.write_file(state_json, state_file)
            
            if success:
                logger.debug(f"Saved processing state for document {self.document_id}")
            return success
        except Exception as e:
            logger.error(f"Error saving processing state for {self.document_id}: {str(e)}")
            return False
    
    def check_pause_requested(self) -> bool:
        """Check if pause has been requested via Celery task manager"""
        if self.task_state_manager:
            return self.task_state_manager.check_pause_requested(self.document_id)
        return False
    
    def check_cancel_requested(self) -> bool:
        """Check if cancellation has been requested via Celery task manager"""
        if self.task_state_manager:
            return self.task_state_manager.check_cancel_requested(self.document_id)
        return False
    
    def handle_pause_request(self, stage: str, checkpoint_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Handle a pause request by saving checkpoint and marking task as paused.
        
        Args:
            stage: Current processing stage
            checkpoint_data: Optional checkpoint data for resuming
            
        Returns:
            True if pause was handled successfully
        """
        try:
            # Save checkpoint data
            if checkpoint_data:
                celery_integration = self.processing_state.get("celery_integration", {})
                celery_integration.setdefault("checkpoints", {})[stage] = checkpoint_data
                self.processing_state["celery_integration"] = celery_integration
                self._save_processing_state()
            
            # Update Celery task manager
            if self.task_state_manager:
                success = self.task_state_manager.mark_task_paused(self.document_id)
                if success:
                    logger.info(f"Document {self.document_id} paused at stage {stage}")
                return success
                
            return True
            
        except Exception as e:
            logger.error(f"Error handling pause request for {self.document_id}: {str(e)}")
            return False
    
    def get_checkpoint_data(self, stage: str) -> Dict[str, Any]:
        """
        Get checkpoint data for resuming a stage.
        
        Args:
            stage: Processing stage
            
        Returns:
            Checkpoint data dictionary
        """
        # Try Celery task manager first
        if self.task_state_manager:
            celery_checkpoint = self.task_state_manager.get_checkpoint_data(self.document_id)
            if celery_checkpoint:
                return celery_checkpoint
        
        # Fall back to local processing state
        celery_integration = self.processing_state.get("celery_integration", {})
        checkpoints = celery_integration.get("checkpoints", {})
        return checkpoints.get(stage, {})
    
    def save_checkpoint(self, stage: str, checkpoint_data: Dict[str, Any]) -> bool:
        """
        Save checkpoint data for a stage.
        
        Args:
            stage: Processing stage
            checkpoint_data: Checkpoint data
            
        Returns:
            True if successful
        """
        try:
            # Save to Celery task manager
            if self.task_state_manager:
                self.task_state_manager.update_progress(
                    self.document_id,
                    percent_complete=checkpoint_data.get("percent_complete", 0),
                    checkpoint_data=checkpoint_data
                )
            
            # Also save locally
            celery_integration = self.processing_state.get("celery_integration", {})
            celery_integration.setdefault("checkpoints", {})[stage] = checkpoint_data
            self.processing_state["celery_integration"] = celery_integration
            
            return self._save_processing_state()
            
        except Exception as e:
            logger.error(f"Error saving checkpoint for {self.document_id} stage {stage}: {str(e)}")
            return False
    
    def register_celery_task(self, stage: str, celery_task_id: str) -> bool:
        """
        Register a Celery task ID for a processing stage.
        
        Args:
            stage: Processing stage
            celery_task_id: Celery task ID
            
        Returns:
            True if successful
        """
        try:
            celery_integration = self.processing_state.get("celery_integration", {})
            celery_integration.setdefault("task_ids", {})[stage] = celery_task_id
            self.processing_state["celery_integration"] = celery_integration
            
            return self._save_processing_state()
            
        except Exception as e:
            logger.error(f"Error registering Celery task for {self.document_id}: {str(e)}")
            return False
    
    def _acquire_state_lock(self, timeout_seconds: int = 30) -> bool:
        """
        Acquire a lock for state modification operations.
        Enhanced with cancellation checking.
        
        Args:
            timeout_seconds: Maximum time to wait for lock acquisition
            
        Returns:
            True if lock acquired, False otherwise
        """
        lock_file = os.path.join(self.stages_dir, "state.lock")
        start_time = time.time()
        
        while (time.time() - start_time) < timeout_seconds:
            # Check for cancellation before trying to acquire lock
            if self.check_cancel_requested():
                logger.info(f"Cancellation requested, aborting lock acquisition for {self.document_id}")
                return False
            
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
        Save data for a specific stage with cancellation checking.
        
        Args:
            stage: Processing stage name
            data: Data to save
            filename: Base filename (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        # Check for cancellation before saving
        if self.check_cancel_requested():
            logger.info(f"Cancellation requested, aborting data save for {self.document_id}")
            return False
            
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
    
    def mark_stage_complete(self, stage: str, celery_task_id: Optional[str] = None) -> bool:
        """
        Mark a stage as completed and advance to next stage.
        Enhanced with better Celery task integration and stage advancement.
        
        Args:
            stage: Stage to mark as complete
            celery_task_id: Optional Celery task ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check for cancellation before marking complete
        if self.check_cancel_requested():
            logger.info(f"Cancellation requested, aborting stage completion for {self.document_id}")
            return False
            
        # Try to acquire lock with longer timeout for critical operations
        if not self._acquire_state_lock(timeout_seconds=60):
            logger.error(f"Could not acquire lock for document {self.document_id}")
            logger.warning(f"Proceeding without lock for stage completion of {stage} in document {self.document_id}")
        
        try:
            # Register Celery task ID if provided
            if celery_task_id:
                self.register_celery_task(stage, celery_task_id)
            
            # Add to completed stages if not already there
            completed_stages = self.processing_state.get("completed_stages", [])
            if stage not in completed_stages:
                completed_stages.append(stage)
                self.processing_state["completed_stages"] = completed_stages
            
            # Update completion time
            completion_times = self.processing_state.get("stage_completion_times", {})
            completion_times[stage] = datetime.now().isoformat()
            self.processing_state["stage_completion_times"] = completion_times
            
            # Get next stage
            next_stage = self._get_next_stage(stage)
            
            # Update current stage in local state
            if next_stage:
                self.processing_state["current_stage"] = next_stage
            else:
                # If no next stage, mark as completed
                self.processing_state["current_stage"] = "completed"
            
            # Clear any error details for this stage
            if "stage_error_details" in self.processing_state and stage in self.processing_state["stage_error_details"]:
                del self.processing_state["stage_error_details"][stage]
            
            success = self._save_processing_state()
            
            # IMPORTANT: Update Celery task manager with stage advancement
            if success and self.task_state_manager:
                target_stage = next_stage if next_stage else "completed"
                try:
                    # Mark the completed stage in the database
                    self.task_state_manager.tasks_repo.mark_stage_completed(
                        document_id=self.document_id,
                        completed_stage=stage,
                        next_stage=target_stage
                    )
                    
                    # Advance to next stage
                    self.task_state_manager.advance_stage(self.document_id, target_stage)
                    
                    logger.info(f"Updated task manager: completed {stage}, advanced to {target_stage}")
                    
                except Exception as e:
                    logger.error(f"Failed to update task manager for stage completion: {str(e)}")
                    # Don't fail the entire operation just because task manager update failed
            
            # Sync with document repository if available (without requiring lock)
            if success and self.doc_repository:
                try:
                    self.doc_repository.update_processing_state(
                        self.document_id, 
                        self.processing_state
                    )
                except Exception as e:
                    logger.warning(f"Failed to sync with document repository: {str(e)}")
                    # Don't fail the stage completion just because repo sync failed
            
            if success:
                target_stage = next_stage if next_stage else "completed"
                logger.info(f"Marked stage '{stage}' as complete for document {self.document_id}, advanced to {target_stage}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error marking stage complete for {self.document_id}: {str(e)}")
            return False
        finally:
            # Always try to release the lock
            self._release_state_lock()
    
    def mark_stage_failed(self, stage: str, error_message: str, celery_task_id: Optional[str] = None) -> bool:
        """
        Mark a stage as failed with error details.
        Enhanced with Celery task integration.
        
        Args:
            stage: Stage that failed
            error_message: Error message
            celery_task_id: Optional Celery task ID
            
        Returns:
            True if successful, False otherwise
        """
        # Try to acquire lock
        if not self._acquire_state_lock(timeout_seconds=30):
            logger.warning(f"Could not acquire lock for marking stage failed, proceeding anyway for document {self.document_id}")
        
        try:
            # Register Celery task ID if provided
            if celery_task_id:
                self.register_celery_task(stage, celery_task_id)
            
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
            
            # Update Celery task manager
            if self.task_state_manager:
                try:
                    self.task_state_manager.complete_task(
                        self.document_id,
                        success=False,
                        error_details={"stage": stage, "error": error_message}
                    )
                except Exception as e:
                    logger.warning(f"Failed to update Celery task manager: {str(e)}")
            
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
        Enhanced with Celery task coordination.
        
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
            
            # Clear Celery-related data for reset stages
            celery_integration = self.processing_state.get("celery_integration", {})
            for data_type in ["task_ids", "checkpoints", "pause_points"]:
                if data_type in celery_integration:
                    for stage in stage_order[target_index:]:
                        if stage in celery_integration[data_type]:
                            del celery_integration[data_type][stage]
            
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
        """Get comprehensive status of all stages with Celery integration"""
        status = {
            "document_id": self.document_id,
            "current_stage": self.get_current_stage(),
            "completed_stages": self.get_completed_stages(),
            "stage_completion_times": self.processing_state.get("stage_completion_times", {}),
            "stage_error_details": self.processing_state.get("stage_error_details", {}),
            "retry_counts": self.processing_state.get("retry_counts", {}),
            "last_updated": self.processing_state.get("last_updated"),
            "celery_integration": self.processing_state.get("celery_integration", {})
        }
        
        # Add pause/cancel status if task manager available
        if self.task_state_manager:
            task = self.task_state_manager.get_task_status(self.document_id)
            if task:
                status["celery_task_status"] = {
                    "task_status": task.task_status,
                    "can_pause": task.can_pause,
                    "can_resume": task.can_resume,
                    "can_cancel": task.can_cancel,
                    "pause_requested": task.pause_requested,
                    "cancel_requested": task.cancel_requested,
                    "percent_complete": task.percent_complete,
                    "worker_id": task.worker_id,
                    "celery_task_id": task.celery_task_id
                }
        
        return status
    
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
            
            # Clean up Celery integration data for this stage
            celery_integration = self.processing_state.get("celery_integration", {})
            for data_type in ["task_ids", "checkpoints", "pause_points"]:
                if data_type in celery_integration and stage in celery_integration[data_type]:
                    del celery_integration[data_type][stage]
            
            return self._save_processing_state()
        except Exception as e:
            logger.error(f"Error cleaning up stage data for {self.document_id}: {str(e)}")
            return False
    
    def mark_workflow_complete(self, celery_task_id: Optional[str] = None) -> bool:
        """
        Mark the entire workflow as completed.
        
        Args:
            celery_task_id: Optional Celery task ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Mark the final completion stage
            self.processing_state["current_stage"] = ProcessingStage.COMPLETED.value
            self.processing_state["workflow_completed_at"] = datetime.now().isoformat()
            
            # Ensure all expected stages are marked as complete
            expected_stages = [
                ProcessingStage.UPLOAD.value,
                ProcessingStage.EXTRACTION.value,
                ProcessingStage.CHUNKING.value,
                ProcessingStage.EMBEDDING.value,
                ProcessingStage.TREE_BUILDING.value,
                ProcessingStage.VECTOR_STORAGE.value
            ]
            
            completed_stages = set(self.processing_state.get("completed_stages", []))
            for stage in expected_stages:
                if stage not in completed_stages:
                    logger.warning(f"Stage {stage} not marked as complete, adding to completed stages")
                    completed_stages.add(stage)
            
            self.processing_state["completed_stages"] = list(completed_stages)
            
            success = self._save_processing_state()
            
            # Update task state manager
            if success and self.task_state_manager:
                try:
                    # Update current stage to completed
                    self.task_state_manager.advance_stage(self.document_id, ProcessingStage.COMPLETED.value)
                    
                    # Mark task as successfully completed
                    self.task_state_manager.complete_task(self.document_id, success=True)
                    
                    logger.info(f"Marked workflow complete in task manager for {self.document_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to update task manager for workflow completion: {str(e)}")
            
            # Update document repository
            if success and self.doc_repository:
                try:
                    self.doc_repository.update_document(self.document_id, {
                        "status": "processed",
                        "processing_completed_at": datetime.now().isoformat(),
                        "processing_state": self.processing_state
                    })
                    
                    logger.info(f"Updated document repository for workflow completion: {self.document_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to update document repository: {str(e)}")
            
            if success:
                logger.info(f"Workflow marked as complete for document {self.document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error marking workflow complete for {self.document_id}: {str(e)}")
            return False