from celery import Celery
from core.celery_config import get_celery_config
import logging

logger = logging.getLogger(__name__)

# Create Celery instance
celery_app = Celery('docrag')

# Configure Celery
celery_config = get_celery_config()
celery_app.conf.update(celery_config)

# Auto-discover tasks
celery_app.autodiscover_tasks([
    'services.celery.tasks',
])

@celery_app.task(bind=True)
def test_task(self):
    """Test task to verify Celery is working"""
    logger.info(f"Test task executed: {self.request.id}")
    return {"task_id": self.request.id, "status": "success", "message": "Celery is working!"}

if __name__ == '__main__':
    celery_app.start()