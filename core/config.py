import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, model_validator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class VectorDBConfig(BaseModel):
    host: str = "localhost"
    port: str = "19530"
    username: Optional[str] = None
    password: Optional[str] = None
    collection_name: str = "document_store"
    dimension: int = 3072
    
    # Hybrid search settings
    enable_hybrid_search: bool = True
    default_vector_weight: float = 0.65  # Slightly favor vector search by default
    fusion_method: str = "weighted"


class OllamaConfig(BaseModel):
    """Configuration for Ollama service"""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    embed_model: str = "llama3.2"
    temperature: float = 0.1
    max_tokens: int = 2000

@dataclass
class CeleryConfig:
    """Celery configuration settings"""
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    task_always_eager: bool = False  # Set to True for synchronous testing
    worker_concurrency: int = 3
    max_tasks_per_child: int = 1000
    task_time_limit: int = 3600  # 1 hour max per task
    task_soft_time_limit: int = 3000  # 50 minutes soft limit



class StorageConfig(BaseModel):
    """Configuration for document storage"""
    storage_type: str = "local"  # "local" or "s3"
    storage_dir: str = "document_store"
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = ""
    aws_region: str = "us-east-1"
    
    @model_validator(mode='after')
    def validate_s3_config(self):
        """Validate S3 configuration if storage_type is s3"""
        if self.storage_type == "s3" and not self.s3_bucket:
            raise ValueError("S3 bucket must be specified when using S3 storage")
        return self


class ProcessingConfig(BaseModel):
    """Configuration for document processing"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tree_levels: int = 3
    language: str = "en"


class DatabaseConfig(BaseModel):
    """Configuration for PostgreSQL database"""
    host: str = "localhost"
    port: str = "5433"
    dbname: str = "yourdb"
    user: str = "youruser"
    password: str = "yourpassword"
    connection_timeout: int = 30


class AppConfig(BaseModel):
    """Main application configuration"""
    app_name: str = "Document RAG API"
    debug: bool = False
    log_level: str = "INFO"
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    @classmethod
    def from_json(cls, file_path: str) -> "AppConfig":
        """Load configuration from JSON file"""
        if not os.path.exists(file_path):
            logger.warning(f"Config file not found: {file_path}, using default configuration")
            return cls()
        
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            
            return cls(**config_data)
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {str(e)}")
            logger.warning("Falling back to default configuration")
            return cls()
    
    @property
    def celery(self) -> CeleryConfig:
        """Get Celery configuration"""
        if not hasattr(self, '_celery_config'):
            self._celery_config = self._load_celery_config()
        return self._celery_config

    def _load_celery_config(self) -> CeleryConfig:
        """Load Celery configuration from environment variables"""
        return CeleryConfig(
            broker_url=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
            result_backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
            task_always_eager=os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true",
            worker_concurrency=int(os.getenv("CELERY_WORKER_CONCURRENCY", "3")),
            max_tasks_per_child=int(os.getenv("CELERY_MAX_TASKS_PER_CHILD", "1000")),
            task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "3600")),
            task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "3000")),
        )
    
    def to_json(self, file_path: str) -> None:
        """Save configuration to JSON file"""
        try:
            config_data = self.dict()
            
            # Create directory if needed (only if directory path is not empty)
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
                
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {str(e)}")
    
    def update_from_env(self) -> "AppConfig":
        """Update configuration from environment variables"""
        
        # Standard environment variable mappings (commonly used conventions)
        standard_env_mappings = {
            # Ollama configuration
            'OLLAMA_BASE_URL': ('ollama', 'base_url'),
            'OLLAMA_HOST': ('ollama', 'host'),  # Will construct base_url from host:port
            'OLLAMA_PORT': ('ollama', 'port'),  # Will construct base_url from host:port
            'OLLAMA_MODEL': ('ollama', 'model'),
            'OLLAMA_EMBED_MODEL': ('ollama', 'embed_model'),
            'OLLAMA_TEMPERATURE': ('ollama', 'temperature'),
            'OLLAMA_MAX_TOKENS': ('ollama', 'max_tokens'),
            
            # Vector DB configuration  
            'MILVUS_HOST': ('vector_db', 'host'),
            'MILVUS_PORT': ('vector_db', 'port'),
            'MILVUS_URI': ('vector_db', 'uri'),  # Special handling needed
            'VECTOR_DB_HOST': ('vector_db', 'host'),
            'VECTOR_DB_PORT': ('vector_db', 'port'),
            
            # Database configuration
            'DATABASE_URL': ('database', 'url'),  # Special handling needed
            'POSTGRES_HOST': ('database', 'host'),
            'POSTGRES_PORT': ('database', 'port'), 
            'POSTGRES_DB': ('database', 'dbname'),
            'POSTGRES_USER': ('database', 'user'),
            'POSTGRES_PASSWORD': ('database', 'password'),
        }
        
        # Handle standard environment variables
        ollama_host = None
        ollama_port = None
        
        for env_var, (section, prop) in standard_env_mappings.items():
            value = os.getenv(env_var)
            if value:
                if hasattr(self, section):
                    section_obj = getattr(self, section)
                    
                    # Special handling for certain variables
                    if env_var == 'OLLAMA_HOST':
                        ollama_host = value
                        continue
                    elif env_var == 'OLLAMA_PORT':
                        ollama_port = value
                        continue
                    elif env_var == 'MILVUS_URI' and value.startswith('http://'):
                        # Parse Milvus URI like "http://standalone:19530"
                        from urllib.parse import urlparse
                        parsed = urlparse(value)
                        if parsed.hostname:
                            setattr(section_obj, 'host', parsed.hostname)
                        if parsed.port:
                            setattr(section_obj, 'port', str(parsed.port))
                        logger.info(f"Updated Milvus config from URI: {value}")
                        continue
                    elif env_var == 'DATABASE_URL':
                        # Parse PostgreSQL URL like "postgresql://user:pass@host:port/db"
                        self._parse_database_url(value)
                        continue
                    
                    # Regular property mapping
                    if hasattr(section_obj, prop):
                        # Convert value to the appropriate type
                        current_value = getattr(section_obj, prop)
                        converted_value = self._convert_env_value(value, current_value)
                        setattr(section_obj, prop, converted_value)
                        logger.info(f"Updated config {section}.{prop} from {env_var}: {converted_value}")
        
        # Construct base_url from host and port if both are provided
        if ollama_host and ollama_port:
            base_url = f"http://{ollama_host}:{ollama_port}"
            self.ollama.base_url = base_url
            logger.info(f"Constructed Ollama base_url from host/port: {base_url}")
        elif ollama_host and not ollama_port:
            # Use default port
            base_url = f"http://{ollama_host}:11434"
            self.ollama.base_url = base_url
            logger.info(f"Constructed Ollama base_url with default port: {base_url}")
        
        # Handle DOCRAG_ prefixed environment variables (existing format)
        prefix = "DOCRAG_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                parts = key[len(prefix):].lower().split('_')
                
                # Handle nested configuration
                if len(parts) >= 2:
                    section = parts[0]
                    prop = '_'.join(parts[1:])
                    
                    if hasattr(self, section) and hasattr(getattr(self, section), prop):
                        section_obj = getattr(self, section)
                        current_value = getattr(section_obj, prop)
                        converted_value = self._convert_env_value(value, current_value)
                        setattr(section_obj, prop, converted_value)
                        logger.debug(f"Updated config {section}.{prop} from {key}: {converted_value}")
        
        # Log final configuration for debugging
        logger.info(f"Final Ollama base_url: {self.ollama.base_url}")
        logger.info(f"Final Vector DB host: {self.vector_db.host}:{self.vector_db.port}")
        
        return self
    
    def _convert_env_value(self, value: str, current_value: Any) -> Any:
        """Convert environment variable string to appropriate type"""
        if isinstance(current_value, bool):
            return value.lower() in ('true', 'yes', '1', 'on')
        elif isinstance(current_value, int):
            return int(value)
        elif isinstance(current_value, float):
            return float(value)
        else:
            return value
    
    def _parse_database_url(self, database_url: str) -> None:
        """Parse DATABASE_URL and update database configuration"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            
            if parsed.scheme in ['postgresql', 'postgres']:
                if parsed.hostname:
                    self.database.host = parsed.hostname
                if parsed.port:
                    self.database.port = str(parsed.port)
                if parsed.username:
                    self.database.user = parsed.username
                if parsed.password:
                    self.database.password = parsed.password
                if parsed.path and len(parsed.path) > 1:
                    self.database.dbname = parsed.path[1:]  # Remove leading /
                    
                logger.info(f"Updated database config from DATABASE_URL")
        except Exception as e:
            logger.error(f"Error parsing DATABASE_URL: {e}")


# Global configuration instance
_config_instance = None


def get_config(config_path: str = "config.json") -> AppConfig:
    """Get the global configuration instance"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = AppConfig.from_json(config_path)
        # Update from environment variables after loading from file
        _config_instance = _config_instance.update_from_env()
    
    return _config_instance


def reset_config() -> None:
    """Reset the global configuration instance"""
    global _config_instance
    _config_instance = None