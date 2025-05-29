from fastapi import APIRouter
from services.celery.tasks.document_tasks import test_document_task
from core.celery_app import test_task
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai/test", tags=["test"])

@router.post("/celery")
async def test_celery():
    """Test basic Celery functionality"""
    try:
        # Submit test task
        task = test_task.delay()
        
        return {
            "status": "success",
            "message": "Test task submitted",
            "task_id": task.id
        }
    except Exception as e:
        logger.error(f"Error submitting test task: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@router.post("/celery/document")
async def test_document_processing():
    """Test document processing queue"""
    try:
        # Submit document test task
        task = test_document_task.delay("Testing document processing queue!")
        
        return {
            "status": "success", 
            "message": "Document test task submitted",
            "task_id": task.id
        }
    except Exception as e:
        logger.error(f"Error submitting document test task: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@router.get("/celery/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a Celery task"""
    from core.celery_app import celery_app
    
    try:
        task_result = celery_app.AsyncResult(task_id)
        
        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": task_result.result,
            "info": task_result.info
        }
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }