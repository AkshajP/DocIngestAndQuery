#!/usr/bin/env python3
"""
Celery worker entry point for document processing.
Enhanced for Docker environment compatibility.
"""

import os
import sys
import signal
import logging
from core.celery_app import celery_app

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def setup_docker_environment():
    """Setup environment for Docker-based execution"""
    
    # Detect if running in Docker
    is_docker = os.path.exists('/.dockerenv') or os.getenv('CONTAINER_ENV') == 'docker'
    
    if is_docker:
        logger.info("Detected Docker environment - applying Docker-specific configurations")
        
        # Set container environment marker
        os.environ['CONTAINER_ENV'] = 'docker'
        
        # Configure for single-threaded ML libraries if requested
        if os.getenv('MINERU_DISABLE_MULTIPROCESSING', '').lower() in ('1', 'true', 'yes'):
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            logger.info("Configured for single-threaded ML execution")
        
        # Ensure proper signal handling
        try:
            signal.signal(signal.SIGTERM, signal.default_int_handler)
            signal.signal(signal.SIGINT, signal.default_int_handler)
            logger.info("Configured signal handlers for Docker")
        except Exception as e:
            logger.warning(f"Could not configure signal handlers: {e}")
    
    else:
        logger.info("Running in native environment")

def main():
    """Main entry point for Celery worker"""
    
    # Setup Docker environment if needed
    setup_docker_environment()
    
    # Set environment variable for worker identification
    os.environ.setdefault('CELERY_WORKER_NAME', 'docrag-worker')
    
    # Log environment info
    logger.info(f"Worker starting with Python {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Container environment: {os.getenv('CONTAINER_ENV', 'native')}")
    
    # Worker arguments - adjusted for Docker
    worker_args = [
        'worker',
        '--loglevel=debug',
        '--concurrency=3',
        '--queues=document_processing,celery',
        '--hostname=docrag-worker@%h',
        '--max-tasks-per-child=50',  # Reduced to prevent memory issues
        '--prefetch-multiplier=1',
        '--without-gossip',  # Disable gossip for Docker
        '--without-mingle',  # Disable mingle for Docker
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