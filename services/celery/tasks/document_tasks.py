from celery import current_task
from core.celery_app import celery_app
import logging
import time

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, queue='document_processing')
def test_document_task(self, message="Hello from document processing!"):
    """Test task for document processing queue"""
    logger.info(f"Executing test document task: {self.request.id}")
    
    # Simulate some work
    for i in range(5):
        time.sleep(1)
        logger.info(f"Task {self.request.id} progress: {i+1}/5")
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'current': i+1, 'total': 5, 'message': f'Processing step {i+1}'}
        )
    
    result = {
        'task_id': self.request.id,
        'message': message,
        'status': 'completed',
        'steps_completed': 5
    }
    
    logger.info(f"Task {self.request.id} completed successfully")
    return result