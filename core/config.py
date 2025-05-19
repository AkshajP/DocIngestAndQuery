import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, model_validator

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
        # Example environment variables: DOCRAG_VECTOR_DB_HOST, DOCRAG_OLLAMA_MODEL, etc.
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
                        
                        # Convert value to the appropriate type
                        current_value = getattr(section_obj, prop)
                        if isinstance(current_value, bool):
                            value = value.lower() in ('true', 'yes', '1')
                        elif isinstance(current_value, int):
                            value = int(value)
                        elif isinstance(current_value, float):
                            value = float(value)
                        
                        setattr(section_obj, prop, value)
                        logger.debug(f"Updated config {section}.{prop} from environment: {value}")
        
        return self


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