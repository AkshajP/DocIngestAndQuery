import sys
import os
import logging

# Add parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_config, reset_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("config_example")

def main():
    """Example of using the configuration system"""
    logger.info("Loading configuration...")
    
    # Get configuration (loads from config.json or uses defaults)
    config = get_config()
    
    # Display current configuration
    logger.info(f"App name: {config.app_name}")
    logger.info(f"Debug mode: {config.debug}")
    logger.info(f"Vector DB host: {config.vector_db.host}")
    logger.info(f"Ollama model: {config.ollama.model}")
    logger.info(f"Storage type: {config.storage.storage_type}")
    logger.info(f"Chunk size: {config.processing.chunk_size}")
    
    # Example of modifying configuration
    logger.info("\nModifying configuration...")
    config.debug = True
    config.vector_db.host = "milvus-server"
    config.ollama.model = "deepseek-r1"
    
    # Display modified configuration
    logger.info(f"Updated debug mode: {config.debug}")
    logger.info(f"Updated Vector DB host: {config.vector_db.host}")
    logger.info(f"Updated Ollama model: {config.ollama.model}")
    
    # Save modified configuration
    modified_config_path = "modified_config.json"
    config.to_json(modified_config_path)
    logger.info(f"Saved modified configuration to {modified_config_path}")
    
    # Reset and load the modified configuration
    reset_config()
    modified_config = get_config(modified_config_path)
    logger.info(f"\nLoaded modified configuration:")
    logger.info(f"Debug mode: {modified_config.debug}")
    logger.info(f"Vector DB host: {modified_config.vector_db.host}")
    logger.info(f"Ollama model: {modified_config.ollama.model}")
    
    # Example of environment variable override
    logger.info("\nOverriding with environment variables...")
    os.environ["DOCRAG_OLLAMA_MODEL"] = "mixtral"
    os.environ["DOCRAG_VECTOR_DB_PORT"] = "19531"
    
    # Reset and get config with environment overrides
    reset_config()
    env_config = get_config()
    logger.info(f"Ollama model (from env): {env_config.ollama.model}")
    logger.info(f"Vector DB port (from env): {env_config.vector_db.port}")

if __name__ == "__main__":
    main()