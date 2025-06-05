#!/usr/bin/env python3
"""
Celery worker entry point for document processing.
"""

import os
import sys
import logging
from core.celery_app import celery_app

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for Celery worker"""
    
    # Set environment variable for worker identification
    os.environ.setdefault('CELERY_WORKER_NAME', 'docrag-worker')
    
    # Worker arguments
    worker_args = [
        'worker',
        '--loglevel=info',
        '--concurrency=3',
        '--queues=document_processing,celery',
        '--hostname=docrag-worker@%h',
        '--max-tasks-per-child=1000',
        '--prefetch-multiplier=1',
    ]
    
    # Add additional arguments from environment if provided
    additional_args = os.environ.get('CELERY_WORKER_ARGS', '').split()
    if additional_args:
        worker_args.extend(additional_args)
    
    logger.info(f"Starting Celery worker with args: {' '.join(worker_args)}")
    
    try:
        # Start the worker
        celery_app.worker_main(worker_args)
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Worker failed to start: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()