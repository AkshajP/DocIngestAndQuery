# services/celery/task_utils.py
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TaskCheckpointManager:
    """Manages checkpoints for pausable/resumable tasks"""
    
    def __init__(self, task_state_manager=None):
        self.task_state_manager = task_state_manager
    
    def save_checkpoint(
        self, 
        document_id: str, 
        stage: str, 
        checkpoint_data: Dict[str, Any]
    ) -> bool:
        """Save checkpoint data for resuming later"""
        try:
            if self.task_state_manager:
                return self.task_state_manager.update_progress(
                    document_id=document_id,
                    percent_complete=checkpoint_data.get("percent_complete", 0),
                    checkpoint_data=checkpoint_data
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {document_id}: {str(e)}")
            return False
    
    def load_checkpoint(self, document_id: str) -> Dict[str, Any]:
        """Load checkpoint data for resuming"""
        try:
            if self.task_state_manager:
                return self.task_state_manager.get_checkpoint_data(document_id)
            return {}
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {document_id}: {str(e)}")
            return {}

def check_task_control_signals(task_state_manager, document_id: str) -> Optional[str]:
    """
    Check for pause/cancel signals and return the requested action.
    
    Returns:
        'pause' if pause requested
        'cancel' if cancel requested  
        None if no control signals
    """
    if not task_state_manager:
        return None
    
    try:
        if task_state_manager.check_cancel_requested(document_id):
            logger.info(f"Cancel signal detected for document {document_id}")
            return 'cancel'
        
        if task_state_manager.check_pause_requested(document_id):
            logger.info(f"Pause signal detected for document {document_id}")
            return 'pause'
        
        return None
    except Exception as e:
        logger.error(f"Error checking control signals for {document_id}: {str(e)}")
        return None

def controlled_sleep(duration: float, task_state_manager, document_id: str, check_interval: float = 0.1) -> Optional[str]:
    """
    Sleep with periodic checks for control signals.
    
    Returns:
        Control signal if detected, None if sleep completed normally
    """
    elapsed = 0.0
    while elapsed < duration:
        signal = check_task_control_signals(task_state_manager, document_id)
        if signal:
            return signal
        
        sleep_time = min(check_interval, duration - elapsed)
        time.sleep(sleep_time)
        elapsed += sleep_time
    
    return None

def execute_with_control_checks(
    func,
    task_state_manager,
    document_id: str,
    check_interval: int = 10,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute a function with periodic control signal checks.
    
    Args:
        func: Function to execute
        task_state_manager: Task state manager instance
        document_id: Document ID
        check_interval: Seconds between control checks
        
    Returns:
        Result from function or control signal response
    """
    start_time = time.time()
    last_check = start_time
    
    try:
        # Check control signals before starting
        signal = check_task_control_signals(task_state_manager, document_id)
        if signal:
            return {"status": signal, "message": f"Task {signal}ed before execution"}
        
        # Execute the function - this will be the actual processing work
        result = func(*args, **kwargs)
        
        # Check control signals after completion
        signal = check_task_control_signals(task_state_manager, document_id)
        if signal:
            return {"status": signal, "message": f"Task {signal}ed after execution", "partial_result": result}
        
        return result
        
    except Exception as e:
        logger.error(f"Error in controlled execution for {document_id}: {str(e)}")
        return {"status": "error", "error": str(e)}

def update_task_progress(
    task_state_manager,
    document_id: str,
    stage: str,
    percent_complete: int,
    status_message: str = None
) -> bool:
    """Update task progress with optional status message"""
    try:
        if task_state_manager:
            checkpoint_data = {
                "stage": stage,
                "percent_complete": percent_complete,
                "timestamp": datetime.now().isoformat(),
                "status_message": status_message
            }
            
            return task_state_manager.update_progress(
                document_id=document_id,
                percent_complete=percent_complete,
                checkpoint_data=checkpoint_data
            )
        return True
    except Exception as e:
        logger.error(f"Failed to update progress for {document_id}: {str(e)}")
        return False

class ControlledProcessor:
    """Wrapper for stage processors that adds control signal checking"""
    
    def __init__(self, processor, task_state_manager, document_id: str):
        self.processor = processor
        self.task_state_manager = task_state_manager
        self.document_id = document_id
    
    def execute_with_checkpoints(self, state_manager, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute processor with control signal checkpoints"""
        
        # Pre-execution check
        signal = check_task_control_signals(self.task_state_manager, self.document_id)
        if signal == 'cancel':
            return {"status": "cancelled", "message": "Task cancelled before execution"}
        elif signal == 'pause':
            return {"status": "paused", "message": "Task paused before execution"}
        
        # Execute the actual processor
        result = self.processor.execute(state_manager, context)
        
        # Post-execution check
        signal = check_task_control_signals(self.task_state_manager, self.document_id)
        if signal == 'cancel':
            return {"status": "cancelled", "message": "Task cancelled after execution", "partial_result": result}
        elif signal == 'pause':
            return {"status": "paused", "message": "Task paused after execution", "stage_result": result}
        
        return result