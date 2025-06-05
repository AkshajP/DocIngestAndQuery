#!/usr/bin/env python3
"""
Script to verify Celery setup is working correctly.
Run this after setting up all Celery infrastructure.
"""

import sys
import time
import logging
from core.celery_app import celery_app, ping, test_db_connection, health_check
from db.document_store.document_tasks_repository import DocumentTasksRepository, DocumentTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_redis_connection():
    """Test Redis broker connectivity"""
    try:
        import redis
        from core.config import get_config
        
        config = get_config()
        r = redis.from_url(config.celery.broker_url)
        r.ping()
        logger.info("✅ Redis connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False

def test_celery_ping():
    """Test basic Celery task execution"""
    try:
        logger.info("Testing Celery ping task...")
        result = ping.delay()
        response = result.get(timeout=30)
        
        if response == "pong":
            logger.info("✅ Celery ping task successful")
            return True
        else:
            logger.error(f"❌ Unexpected ping response: {response}")
            return False
    except Exception as e:
        logger.error(f"❌ Celery ping task failed: {e}")
        return False

def test_database_task():
    """Test database connectivity from Celery worker"""
    try:
        logger.info("Testing database connectivity task...")
        result = test_db_connection.delay()
        response = result.get(timeout=30)
        
        if response['status'] == 'success':
            logger.info(f"✅ Database connection successful ({response['total_documents']} documents found)")
            return True
        else:
            logger.error(f"❌ Database connection failed: {response}")
            return False
    except Exception as e:
        logger.error(f"❌ Database connection task failed: {e}")
        return False

def test_document_tasks_table():
    """Test document_tasks table operations"""
    try:
        logger.info("Testing document_tasks table...")
        
        # Try to create repository with better error handling
        try:
            repo = DocumentTasksRepository()
            logger.info("✅ DocumentTasksRepository created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create DocumentTasksRepository: {e}")
            logger.info("💡 Make sure PostgreSQL is running and accessible:")
            logger.info("   - For Docker: docker-compose up postgres")
            logger.info("   - Check DATABASE_URL or POSTGRES_* environment variables")
            return False
        
        # Create a test task
        test_task = DocumentTask(
            document_id="test_doc_123",
            current_stage="upload",
            task_status="PENDING",
        )
        
        try:
            task_id = repo.create_task(test_task)
            logger.info(f"✅ Created test task with ID: {task_id}")
        except Exception as e:
            logger.error(f"❌ Failed to create test task: {e}")
            logger.info("💡 Make sure the document_tasks table exists:")
            logger.info("   - Run: psql -d yourdb -f db/migrations/add_document_tasks_table.sql")
            return False
        
        # Retrieve the task
        try:
            retrieved_task = repo.get_task_by_document_id("test_doc_123")
            if retrieved_task and retrieved_task.document_id == "test_doc_123":
                logger.info("✅ Document task retrieval successful")
            else:
                logger.error("❌ Failed to retrieve created task")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to retrieve task: {e}")
            return False
        
        # Update task status
        try:
            success = repo.update_task_status("test_doc_123", "STARTED")
            if success:
                logger.info("✅ Task status update successful")
            else:
                logger.error("❌ Task status update failed")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to update task status: {e}")
            return False
        
        # Clean up test task
        try:
            repo.delete_task("test_doc_123")
            logger.info("✅ Test task cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up test task: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Document tasks table test failed: {e}")
        return False

def test_worker_health():
    """Test worker health check"""
    try:
        logger.info("Testing worker health check...")
        result = health_check.delay()
        response = result.get(timeout=30)
        
        if response['status'] == 'healthy':
            logger.info(f"✅ Worker health check successful (Host: {response['worker_host']})")
            return True
        else:
            logger.error(f"❌ Worker health check failed: {response}")
            return False
    except Exception as e:
        logger.error(f"❌ Worker health check failed: {e}")
        return False

def check_celery_worker_status():
    """Check if Celery workers are running"""
    try:
        inspect = celery_app.control.inspect()
        active_queues = inspect.active_queues()
        
        if active_queues:
            logger.info(f"✅ Found {len(active_queues)} active Celery worker(s)")
            for worker, queues in active_queues.items():
                queue_names = [q['name'] for q in queues]
                logger.info(f"  Worker: {worker}, Queues: {queue_names}")
            return True
        else:
            logger.warning("⚠️ No active Celery workers found")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to check worker status: {e}")
        return False

def main():
    """Run all verification tests"""
    logger.info("🚀 Starting Celery setup verification...")
    
    # Show environment info for debugging
    import os
    logger.info("\n📋 Environment Configuration:")
    logger.info(f"  CELERY_BROKER_URL: {os.getenv('CELERY_BROKER_URL', 'Not set')}")
    logger.info(f"  DATABASE_URL: {os.getenv('DATABASE_URL', 'Not set')}")
    logger.info(f"  POSTGRES_HOST: {os.getenv('DOCRAG_DATABASE_HOST', os.getenv('POSTGRES_HOST', 'localhost'))}")
    logger.info(f"  POSTGRES_PORT: {os.getenv('DOCRAG_DATABASE_PORT', os.getenv('POSTGRES_PORT', '5432'))}")
    logger.info(f"  POSTGRES_USER: {os.getenv('POSTGRES_USER', 'youruser')}")
    logger.info(f"  POSTGRES_DB: {os.getenv('POSTGRES_DB', 'yourdb')}")
    
    tests = [
        ("Redis Connection", test_redis_connection),
        ("Document Tasks Table", test_document_tasks_table),
        ("Celery Worker Status", check_celery_worker_status),
        ("Celery Ping Task", test_celery_ping),
        ("Database Connectivity Task", test_database_task),
        ("Worker Health Check", test_worker_health),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Celery setup is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the configuration.")
        logger.info("\n💡 Troubleshooting tips:")
        logger.info("1. Make sure all services are running: docker-compose up -d")
        logger.info("2. Check database migration: psql -d yourdb -f db/migrations/add_document_tasks_table.sql")
        logger.info("3. Verify environment variables are set correctly")
        logger.info("4. Check Redis connectivity: redis-cli ping")
        return 1

if __name__ == "__main__":
    sys.exit(main())