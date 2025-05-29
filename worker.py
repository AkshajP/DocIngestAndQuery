#!/usr/bin/env python
"""
Celery worker entry point for document processing.
"""

import os
import sys
import logging

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.celery_app import celery_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting Celery worker...")
    
    # Start worker with specific queues
    celery_app.start([
        'worker',
        '--loglevel=INFO',
        '--queues=document_processing,query_processing',
        '--concurrency=2',  # Start with 2 worker processes
    ])