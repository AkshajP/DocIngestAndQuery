# core/service_manager.py

import logging
from typing import Optional, Dict, Any
from functools import lru_cache

from core.config import get_config
from db.chat_store.repository import ChatRepository, UserCaseRepository
from db.document_store.repository import DocumentMetadataRepository
from services.chat.manager import ChatManager
from services.chat.history import ChatHistoryService
from services.retrieval.query_engine import QueryEngine
from services.ml.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class ServiceManager:
    """
    Centralized service manager that initializes and manages all application services.
    Services are created once at startup and reused across requests.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self._services: Dict[str, Any] = {}
        self._initialized = False
        
    def initialize(self):
        """Initialize all services at application startup"""
        if self._initialized:
            return
            
        logger.info("Initializing application services...")
        
        try:
            # Initialize core repositories with shared database configuration
            self._services['chat_repository'] = ChatRepository(self.config.database)
            self._services['user_case_repository'] = UserCaseRepository(self.config.database)
            self._services['document_repository'] = DocumentMetadataRepository()
            
            # Initialize higher-level services (pass repositories to avoid re-initialization)
            self._services['chat_history_service'] = ChatHistoryService(
                config=self.config,
                chat_repo=self._services['chat_repository']
            )
            
            self._services['chat_manager'] = ChatManager(
                config=self.config,
                chat_repo=self._services['chat_repository'],
                doc_repo=self._services['document_repository'],
                history_service=self._services['chat_history_service']
            )
            
            # Initialize ML services
            self._services['embedding_service'] = EmbeddingService(
                model_name=self.config.ollama.embed_model,
                base_url=self.config.ollama.base_url
            )
            
            self._services['query_engine'] = QueryEngine(
                config=self.config,
                embeddings=self._services['embedding_service']
            )
            
            self._initialized = True
            logger.info("All application services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {str(e)}")
            raise
    
    def get_service(self, service_name: str):
        """Get a service by name"""
        if not self._initialized:
            raise RuntimeError("Services not initialized. Call initialize() first.")
            
        service = self._services.get(service_name)
        if service is None:
            raise ValueError(f"Service '{service_name}' not found")
            
        return service
    
    @property
    def chat_manager(self) -> ChatManager:
        return self.get_service('chat_manager')
    
    @property 
    def chat_history_service(self) -> ChatHistoryService:
        return self.get_service('chat_history_service')
    
    @property
    def chat_repository(self) -> ChatRepository:
        return self.get_service('chat_repository')
    
    @property
    def user_case_repository(self) -> UserCaseRepository:
        return self.get_service('user_case_repository')
    
    @property
    def document_repository(self) -> DocumentMetadataRepository:
        return self.get_service('document_repository')
    
    @property
    def query_engine(self) -> QueryEngine:
        return self.get_service('query_engine')
    
    @property
    def embedding_service(self) -> EmbeddingService:
        return self.get_service('embedding_service')
    
    def shutdown(self):
        """Cleanup services on application shutdown"""
        logger.info("Shutting down application services...")
        
        # Close database connections
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'close'):
                    service.close()
                elif hasattr(service, 'release'):
                    service.release()
            except Exception as e:
                logger.error(f"Error shutting down {service_name}: {str(e)}")
        
        self._services.clear()
        self._initialized = False
        logger.info("Application services shutdown complete")

# Global service manager instance
_service_manager: Optional[ServiceManager] = None

def get_service_manager() -> ServiceManager:
    """Get the global service manager instance"""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager

@lru_cache(maxsize=1)
def get_initialized_service_manager() -> ServiceManager:
    """Get initialized service manager (cached)"""
    manager = get_service_manager()
    if not manager._initialized:
        manager.initialize()
    return manager