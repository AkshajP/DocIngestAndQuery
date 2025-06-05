import pytest
import time
import os
from unittest.mock import patch, MagicMock
from core.celery_app import celery_app, ping, test_db_connection, health_check

# Configure Celery for testing
celery_app.conf.update(
    task_always_eager=True,  # Execute tasks synchronously for testing
    task_eager_propagates=True,
    broker_url='memory://',
    result_backend='cache+memory://',
)

class TestCelerySetup:
    """Test suite for Celery infrastructure setup"""
    
    def test_celery_app_configuration(self):
        """Test that Celery app is properly configured"""
        assert celery_app.conf.task_serializer == 'json'
        assert celery_app.conf.result_serializer == 'json'
        assert celery_app.conf.accept_content == ['json']
        assert 'document_processing' in celery_app.conf.task_routes.values()[0]['queue']
    
    def test_ping_task(self):
        """Test the basic ping task functionality"""
        result = ping.delay()
        assert result.get(timeout=10) == "pong"
        assert result.successful()
    
    def test_ping_task_direct_call(self):
        """Test ping task when called directly"""
        result = ping()
        assert result == "pong"
    
    def test_health_check_task(self):
        """Test the health check task"""
        result = health_check.delay()
        response = result.get(timeout=10)
        
        assert response['status'] == 'healthy'
        assert 'timestamp' in response
        assert 'worker_host' in response
        assert 'celery_version' in response
        assert result.successful()
    
    @patch('db.document_store.repository.DocumentMetadataRepository')
    def test_db_connection_task_success(self, mock_repo_class):
        """Test database connection task success scenario"""
        # Mock the repository
        mock_repo = MagicMock()
        mock_repo.get_statistics.return_value = {'total_documents': 5}
        mock_repo_class.return_value = mock_repo
        
        result = test_db_connection.delay()
        response = result.get(timeout=10)
        
        assert response['status'] == 'success'
        assert response['total_documents'] == 5
        assert result.successful()
    
    @patch('db.document_store.repository.DocumentMetadataRepository')
    def test_db_connection_task_failure(self, mock_repo_class):
        """Test database connection task failure scenario"""
        # Mock repository to raise exception
        mock_repo_class.side_effect = Exception("Database connection failed")
        
        result = test_db_connection.delay()
        
        with pytest.raises(Exception) as exc_info:
            result.get(timeout=10)
        
        assert "Database connection failed" in str(exc_info.value)
        assert result.failed()
    
    @patch('db.vector_store.adapter.VectorStoreAdapter')
    def test_vector_db_connection_task_success(self, mock_adapter_class):
        """Test vector database connection task success scenario"""
        # Mock the vector store adapter
        mock_adapter = MagicMock()
        mock_adapter.list_collections.return_value = ['collection1', 'collection2']
        mock_adapter_class.return_value = mock_adapter
        
        result = test_vector_db_connection.delay()
        response = result.get(timeout=10)
        
        assert response['status'] == 'success'
        assert response['collections_count'] == 2
        assert result.successful()
    
    def test_task_routing_configuration(self):
        """Test that task routing is properly configured"""
        routes = celery_app.conf.task_routes
        
        # Check that document tasks are routed to document_processing queue
        assert 'services.celery.tasks.document_tasks.*' in routes
        assert routes['services.celery.tasks.document_tasks.*']['queue'] == 'document_processing'
    
    def test_worker_configuration(self):
        """Test worker-specific configuration"""
        assert celery_app.conf.worker_prefetch_multiplier == 1
        assert celery_app.conf.task_acks_late == True
        assert celery_app.conf.result_expires == 3600
        assert celery_app.conf.task_max_retries == 3

class TestCeleryIntegration:
    """Integration tests for Celery setup"""
    
    @pytest.mark.integration
    def test_redis_connection(self):
        """Test Redis connection (requires Redis to be running)"""
        # This test requires actual Redis instance
        redis_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
        
        try:
            import redis
            r = redis.from_url(redis_url)
            r.ping()
            assert True  # If we reach here, connection is successful
        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")
    
    @pytest.mark.integration 
    def test_celery_worker_discovery(self):
        """Test that tasks can be discovered by Celery worker"""
        # Check that our tasks are registered
        registered_tasks = celery_app.tasks.keys()
        
        assert 'core.celery_app.ping' in registered_tasks
        assert 'core.celery_app.health_check' in registered_tasks
        assert 'core.celery_app.test_db_connection' in registered_tasks
        assert 'core.celery_app.test_vector_db_connection' in registered_tasks

class TestTaskStateTracking:
    """Test task state and result tracking"""
    
    def test_task_result_storage(self):
        """Test that task results are properly stored"""
        result = ping.delay()
        
        # Test result properties
        assert result.task_id is not None
        assert result.state in ['PENDING', 'SUCCESS']
        
        # Get result
        response = result.get(timeout=10)
        assert response == "pong"
        assert result.successful()
    
    def test_task_failure_handling(self):
        """Test task failure scenarios"""
        
        @celery_app.task
        def failing_task():
            raise ValueError("Test error")
        
        result = failing_task.delay()
        
        with pytest.raises(ValueError):
            result.get(timeout=10)
        
        assert result.failed()
        assert result.state == 'FAILURE'

# Pytest fixtures for setup/teardown
@pytest.fixture(scope="session")
def celery_config():
    """Celery configuration for testing"""
    return {
        'broker_url': 'memory://',
        'result_backend': 'cache+memory://',
        'task_always_eager': True,
        'task_eager_propagates': True,
    }

@pytest.fixture
def celery_worker_parameters():
    """Celery worker parameters for testing"""
    return {
        'queues': ('celery', 'document_processing'),
        'perform_ping_check': False,
    }