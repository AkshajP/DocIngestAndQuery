# api/routes/task_management_routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import logging

from api.routes.access_control import get_current_user, get_current_case, validate_user_case_access
from core.service_manager import get_initialized_service_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/tasks", tags=["task_management"])

@router.post("/{document_id}/pause")
async def pause_document_processing(
    document_id: str,
    case_id: str = Depends(get_current_case),
    _: bool = Depends(validate_user_case_access)
) -> Dict[str, Any]:
    """
    Pause document processing.
    
    Args:
        document_id: Document ID to pause
        case_id: Case ID for access control
        
    Returns:
        Dictionary with pause status
    """
    try:
        service_manager = get_initialized_service_manager()
        
        # Verify document exists and belongs to case
        document = service_manager.document_repository.get_document(document_id, case_id=case_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if document can be paused
        if document.get("status") != "processing":
            raise HTTPException(
                status_code=400, 
                detail=f"Document is not in processing state (current: {document.get('status')})"
            )
        
        if not document.get("can_pause", False):
            raise HTTPException(status_code=400, detail="Document cannot be paused at this time")
        
        # Request pause via upload service
        result = service_manager.upload_service.pause_document_processing(document_id)
        
        if result["status"] == "success":
            return {
                "status": "success",
                "message": f"Pause requested for document {document_id}",
                "document_id": document_id
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to pause document"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.post("/{document_id}/resume")
async def resume_document_processing(
    document_id: str,
    background_tasks: BackgroundTasks,
    case_id: str = Depends(get_current_case),
    _: bool = Depends(validate_user_case_access)
) -> Dict[str, Any]:
    """
    Resume paused document processing.
    
    Args:
        document_id: Document ID to resume
        background_tasks: FastAPI background tasks
        case_id: Case ID for access control
        
    Returns:
        Dictionary with resume status
    """
    try:
        service_manager = get_initialized_service_manager()
        
        # Verify document exists and belongs to case
        document = service_manager.document_repository.get_document(document_id, case_id=case_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if document can be resumed
        if document.get("status") != "paused":
            raise HTTPException(
                status_code=400, 
                detail=f"Document is not in paused state (current: {document.get('status')})"
            )
        
        if not document.get("can_resume", False):
            raise HTTPException(status_code=400, detail="Document cannot be resumed at this time")
        
        # Generate new task ID for resumed processing
        import uuid
        new_celery_task_id = f"resume_{document_id}_{uuid.uuid4().hex[:8]}"
        
        # Resume via upload service (this will be handled in background)
        background_tasks.add_task(
            _resume_document_background,
            service_manager,
            document_id,
            new_celery_task_id
        )
        
        # Update document status immediately
        service_manager.document_repository.update_document(document_id, {
            "status": "processing",
            "resume_requested_at": "now"
        })
        
        return {
            "status": "success",
            "message": f"Resume initiated for document {document_id}",
            "document_id": document_id,
            "new_task_id": new_celery_task_id
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.post("/{document_id}/cancel")
async def cancel_document_processing(
    document_id: str,
    case_id: str = Depends(get_current_case),
    _: bool = Depends(validate_user_case_access)
) -> Dict[str, Any]:
    """
    Cancel document processing.
    
    Args:
        document_id: Document ID to cancel
        case_id: Case ID for access control
        
    Returns:
        Dictionary with cancel status
    """
    try:
        service_manager = get_initialized_service_manager()
        
        # Verify document exists and belongs to case
        document = service_manager.document_repository.get_document(document_id, case_id=case_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if document can be cancelled
        current_status = document.get("status")
        if current_status in ["completed", "cancelled", "failed"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Document is already in terminal state: {current_status}"
            )
        
        if not document.get("can_cancel", True):  # Default to True for cancellation
            raise HTTPException(status_code=400, detail="Document cannot be cancelled at this time")
        
        # Request cancellation via upload service
        result = service_manager.upload_service.cancel_document_processing(document_id)
        
        if result["status"] == "success":
            return {
                "status": "success",
                "message": f"Cancellation requested for document {document_id}",
                "document_id": document_id
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to cancel document"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/{document_id}/status")
async def get_document_task_status(
    document_id: str,
    case_id: str = Depends(get_current_case),
    _: bool = Depends(validate_user_case_access)
) -> Dict[str, Any]:
    """
    Get detailed task status for a document.
    
    Args:
        document_id: Document ID
        case_id: Case ID for access control
        
    Returns:
        Dictionary with detailed task status
    """
    try:
        service_manager = get_initialized_service_manager()
        
        # Verify document exists and belongs to case
        document = service_manager.document_repository.get_document(document_id, case_id=case_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get comprehensive processing status
        status_result = service_manager.upload_service.get_document_processing_status(document_id)
        
        if status_result["status"] == "error":
            raise HTTPException(status_code=500, detail=status_result["error"])
        
        # Format response for API
        processing_status = status_result["processing_status"]
        task_control_status = status_result.get("task_control_status", {})
        
        return {
            "document_id": document_id,
            "current_stage": processing_status.get("current_stage"),
            "task_status": task_control_status.get("task_status", document.get("status")),
            "percent_complete": task_control_status.get("percent_complete", 0),
            "can_pause": task_control_status.get("can_pause", document.get("can_pause", False)),
            "can_resume": task_control_status.get("can_resume", document.get("can_resume", False)),
            "can_cancel": task_control_status.get("can_cancel", document.get("can_cancel", True)),
            "pause_requested": task_control_status.get("pause_requested", False),
            "cancel_requested": task_control_status.get("cancel_requested", False),
            "completed_stages": processing_status.get("completed_stages", []),
            "stage_completion_times": processing_status.get("stage_completion_times", {}),
            "stage_error_details": processing_status.get("stage_error_details", {}),
            "retry_counts": processing_status.get("retry_counts", {}),
            "last_updated": processing_status.get("last_updated"),
            "worker_info": task_control_status.get("worker_info", {}),
            "celery_task_id": task_control_status.get("celery_task_id")
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status for document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/{document_id}/control-status")
async def get_document_control_capabilities(
    document_id: str,
    case_id: str = Depends(get_current_case),
    _: bool = Depends(validate_user_case_access)
) -> Dict[str, Any]:
    """
    Get control capabilities (pause/resume/cancel) for a document.
    
    Args:
        document_id: Document ID
        case_id: Case ID for access control
        
    Returns:
        Dictionary with control capabilities
    """
    try:
        service_manager = get_initialized_service_manager()
        
        # Verify document exists and belongs to case
        document = service_manager.document_repository.get_document(document_id, case_id=case_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get task control status
        if service_manager.task_state_manager:
            task = service_manager.task_state_manager.get_task_status(document_id)
            if task:
                return {
                    "document_id": document_id,
                    "can_pause": task.can_pause,
                    "can_resume": task.can_resume,
                    "can_cancel": task.can_cancel,
                    "pause_requested": task.pause_requested,
                    "cancel_requested": task.cancel_requested,
                    "task_status": task.task_status,
                    "percent_complete": task.percent_complete
                }
        
        # Fallback to document metadata
        current_status = document.get("status", "unknown")
        return {
            "document_id": document_id,
            "can_pause": current_status == "processing",
            "can_resume": current_status == "paused",
            "can_cancel": current_status not in ["completed", "cancelled", "failed"],
            "pause_requested": False,
            "cancel_requested": False,
            "task_status": current_status,
            "percent_complete": 0
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting control status for document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Background task functions
async def _resume_document_background(
    service_manager,
    document_id: str,
    celery_task_id: str
):
    """Background task to resume document processing"""
    try:
        result = service_manager.upload_service.resume_document_processing(
            document_id=document_id,
            celery_task_id=celery_task_id
        )
        logger.info(f"Resume processing completed for document {document_id}: {result['status']}")
    except Exception as e:
        logger.error(f"Error in background resume for document {document_id}: {str(e)}")
        # Update document with error status
        try:
            service_manager.document_repository.update_document(document_id, {
                "status": "failed",
                "error_message": f"Resume failed: {str(e)}",
                "failure_time": "now"
            })
        except:
            pass  # Don't fail if we can't update the status