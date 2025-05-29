from core.config import get_config

def get_celery_config():
    """Get Celery configuration from app config"""
    config = get_config()
    
    return {
        'broker_url': config.celery.broker_url,
        'result_backend': config.celery.result_backend,
        'task_serializer': config.celery.task_serializer,
        'accept_content': config.celery.accept_content,
        'result_serializer': config.celery.result_serializer,
        'timezone': config.celery.timezone,
        'enable_utc': config.celery.enable_utc,
        'task_track_started': config.celery.task_track_started,
        'task_time_limit': config.celery.task_time_limit,
        'task_soft_time_limit': config.celery.task_soft_time_limit,
        'worker_prefetch_multiplier': config.celery.worker_prefetch_multiplier,
        'worker_max_tasks_per_child': config.celery.worker_max_tasks_per_child,
        
        # Task routing
        'task_routes': {
            'services.celery.tasks.document_tasks.*': {'queue': 'document_processing'},
            'services.celery.tasks.query_tasks.*': {'queue': 'query_processing'},
        },
        
        # Queue definitions
        'task_queues': {
            'document_processing': {
                'exchange': 'document_processing',
                'routing_key': 'document_processing',
            },
            'query_processing': {
                'exchange': 'query_processing', 
                'routing_key': 'query_processing',
            },
        },
    }