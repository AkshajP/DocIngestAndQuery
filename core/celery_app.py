from celery import Celery
from core.config import get_config
import logging

logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Create Celery app
celery_app = Celery('docrag')

# Configure Celery
celery_app.conf.update(
    broker_url=config.celery.broker_url,
    result_backend=config.celery.result_backend,
    task_serializer=config.celery.task_serializer,
    accept_content=config.celery.accept_content,
    result_serializer=config.celery.result_serializer,
    timezone=config.celery.timezone,
    enable_utc=config.celery.enable_utc,
    task_track_started=config.celery.task_track_started,
    task_time_limit=config.celery.task_time_limit,
    task_soft_time_limit=config.celery.task_soft_time_limit,
    worker_prefetch_multiplier=config.celery.worker_prefetch_multiplier,
    worker_max_tasks_per_child=config.celery.worker_max_tasks_per_child,
    
    # Task routing
    task_routes={
        'services.celery.tasks.document_tasks.*': {'queue': 'document_processing'},
        'services.celery.tasks.query_tasks.*': {'queue': 'query_processing'},
    },
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Worker settings
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
)

# Auto-discover tasks
celery_app.autodiscover_tasks([
    'services.celery.tasks',
])

# Test task for basic connectivity
@celery_app.task(bind=True)
def test_task(self):
    """Basic test task to verify Celery is working"""
    try:
        logger.info(f"Test task executed on worker {self.request.hostname}")
        return {
            'status': 'success',
            'message': 'Celery is working!',
            'task_id': self.request.id,
            'worker': self.request.hostname
        }
    except Exception as e:
        logger.error(f"Test task failed: {str(e)}")
        raise

# Configure logging
def setup_celery_logging():
    """Setup logging for Celery"""
    import logging.config
    
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '[%(asctime)s: %(levelname)s/%(name)s] %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
            },
        },
        'loggers': {
            'celery': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False,
            },
        },
    }
    
    logging.config.dictConfig(LOGGING_CONFIG)

# Initialize logging
setup_celery_logging()

logger.info("Celery app initialized successfully")
logger.info(f"Broker URL: {config.celery.broker_url}")
logger.info(f"Result Backend: {config.celery.result_backend}")